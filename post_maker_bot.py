import os
import re
import sys
import logging
import asyncio
import random
import subprocess
import trafilatura
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, CallbackQueryHandler, filters
from litellm import completion

# Fun status messages while processing
STATUS_MESSAGES = [
    "🧠 Читаю статью...",
    "🔍 Ищу инсайты...",
    "✍️ Пишу черновик...",
    "🎨 Добавляю огонька...",
    "💡 Формулирую мысль...",
    "🚀 Почти готово...",
    "🔥 Делаю пост виральным...",
    "📝 Редактирую текст...",
    "🎯 Ловлю суть...",
    "⚡ Генерирую контент...",
    "🤖 Нейросеть думает...",
    "💭 Анализирую историю...",
    "🏗️ Собираю пост...",
    "✨ Полирую формулировки...",
]

# Maximum number of stored generations per user
MAX_STORED_GENERATIONS = 10

# Prompt modifiers for regeneration
PROMPT_MODIFIERS = {
    "shorter": """ВАЖНО: Сделай пост ЗНАЧИТЕЛЬНО короче. Максимум 800-1000 символов.
Сохрани главную мысль, убери всё второстепенное. Будь максимально лаконичен.""",

    "regenerate": """ВАЖНО: Напиши ДРУГОЙ вариант поста с ДРУГИМ углом подачи.
Используй другую структуру, другой хук, другие акценты. Не повторяй предыдущий вариант.""",

    "custom": """ВАЖНО: Учти эти правки при переписывании поста:
{edits}"""
}

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/google/gemini-3-pro-preview")
ADMIN_ID = os.getenv("ADMIN_ID")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def markdown_to_html(text):
    """Convert Markdown formatting to Telegram HTML."""
    import html

    # Replace guillemets (« ») with regular quotes
    text = text.replace('«', '"').replace('»', '"')

    # Replace ё with е
    text = text.replace('ё', 'е').replace('Ё', 'Е')

    # First, protect URLs from being modified
    url_pattern = r'(https?://[^\s]+)'
    urls = re.findall(url_pattern, text)
    url_placeholders = {}
    for i, url in enumerate(urls):
        placeholder = f"__URL_PLACEHOLDER_{i}__"
        url_placeholders[placeholder] = html.escape(url)
        text = text.replace(url, placeholder)

    # Escape HTML special characters (except our placeholders)
    text = html.escape(text)

    # Restore URL placeholders
    for placeholder, url in url_placeholders.items():
        text = text.replace(html.escape(placeholder), url)

    # Convert **bold** to <b>bold</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

    # Convert *italic* to <i>italic</i> (but not inside URLs)
    text = re.sub(r'(?<![/:])\*([^*\n]+?)\*(?![/])', r'<i>\1</i>', text)

    # Convert `code` to <code>code</code>
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)

    # Convert [text](url) to <a href="url">text</a>
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    return text

def extract_content(text):
    """Extract content from URL or return original text. Returns None if URL extraction fails."""
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)

    # If text is substantial (>300 chars), use it as-is even if it contains URLs
    if len(text) > 300:
        logger.info(f"Using provided text as content ({len(text)} chars)")
        return text

    if urls:
        url = urls[0]
        logger.info(f"Extracting content from URL: {url}")
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, output_format='markdown', include_links=True)
            if extracted and len(extracted) > 100:
                logger.info(f"Successfully extracted {len(extracted)} chars from {url}")
                return extracted
            else:
                logger.warning(f"Trafilatura returned empty/short content for {url}")
                return None
        else:
            logger.warning(f"Failed to download content from {url}")
            return None
    return text

def get_prompt():
    """Read prompt from prompt.md file."""
    try:
        with open("prompt.md", "r", encoding="utf-8") as f:
            content = f.read().strip()
            logger.info(f"Loaded prompt from prompt.md ({len(content)} chars)")
            return content
    except Exception as e:
        logger.error(f"Error reading prompt.md: {e}")
        return "You are a helpful assistant."

def create_post_keyboard(message_id: int) -> InlineKeyboardMarkup:
    """Create inline keyboard with post modification buttons."""
    keyboard = [
        [
            InlineKeyboardButton("✂️ Короче", callback_data=f"shorter:{message_id}"),
            InlineKeyboardButton("🔄 Другой", callback_data=f"regenerate:{message_id}"),
        ],
        [
            InlineKeyboardButton("📝 Правки", callback_data=f"edits:{message_id}"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def save_generation(context: ContextTypes.DEFAULT_TYPE, message_id: int, original_content: str, generated_post: str):
    """Save generation data to user_data for later regeneration."""
    if 'generations' not in context.user_data:
        context.user_data['generations'] = {}

    # Store generation data
    context.user_data['generations'][str(message_id)] = {
        'original_content': original_content,
        'generated_post': generated_post,
    }

    # Clean up old generations if too many
    generations = context.user_data['generations']
    if len(generations) > MAX_STORED_GENERATIONS:
        # Remove oldest entries (smallest message_ids)
        sorted_ids = sorted(generations.keys(), key=int)
        for old_id in sorted_ids[:len(generations) - MAX_STORED_GENERATIONS]:
            del generations[old_id]

def get_generation(context: ContextTypes.DEFAULT_TYPE, message_id: int) -> dict | None:
    """Get saved generation data by message_id."""
    if 'generations' not in context.user_data:
        return None
    return context.user_data['generations'].get(str(message_id))

async def update_status_periodically(status_message, stop_event):
    """Update status message with fun phrases until stopped."""
    messages = STATUS_MESSAGES.copy()
    random.shuffle(messages)
    index = 0

    while not stop_event.is_set():
        await asyncio.sleep(2.5)
        if stop_event.is_set():
            break
        try:
            await status_message.edit_text(messages[index % len(messages)])
            index += 1
        except Exception:
            pass

async def regenerate_post(update: Update, context: ContextTypes.DEFAULT_TYPE, gen_data: dict, modifier: str, custom_edits: str = None):
    """Regenerate post with a modifier."""
    chat_id = update.effective_chat.id

    # Send status message
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    status_message = await context.bot.send_message(chat_id=chat_id, text="🔄 Перегенерирую пост...")

    # Start status updater
    stop_event = asyncio.Event()
    status_task = asyncio.create_task(update_status_periodically(status_message, stop_event))

    try:
        system_prompt = get_prompt()

        # Add modifier to the prompt
        if modifier == "custom" and custom_edits:
            modifier_text = PROMPT_MODIFIERS["custom"].format(edits=custom_edits)
        else:
            modifier_text = PROMPT_MODIFIERS.get(modifier, "")

        # Include the previous generated post for context
        user_content = f"""Оригинальный контент:
{gen_data['original_content']}

---
Предыдущий сгенерированный пост:
{gen_data['generated_post']}

---
{modifier_text}"""

        logger.info(f"Regenerating post with modifier: {modifier}")
        response = await asyncio.to_thread(
            completion,
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            api_key=OPENROUTER_API_KEY
        )

        reply_text = response.choices[0].message.content
        logger.info(f"Regenerated post ({len(reply_text)} chars)")

        # Stop status updater
        stop_event.set()
        status_task.cancel()

        # Delete status and send new post
        await status_message.delete()

        try:
            html_text = markdown_to_html(reply_text)
            sent_message = await context.bot.send_message(
                chat_id=chat_id,
                text=html_text,
                parse_mode='HTML',
                reply_markup=create_post_keyboard(0)  # Placeholder, will update
            )
        except Exception as parse_error:
            logger.warning(f"HTML parse failed, sending as plain text: {parse_error}")
            sent_message = await context.bot.send_message(
                chat_id=chat_id,
                text=reply_text,
                reply_markup=create_post_keyboard(0)
            )

        # Update keyboard with correct message_id and save generation
        await sent_message.edit_reply_markup(reply_markup=create_post_keyboard(sent_message.message_id))
        save_generation(context, sent_message.message_id, gen_data['original_content'], reply_text)

        return sent_message

    except Exception as e:
        stop_event.set()
        status_task.cancel()
        logger.error(f"Error in regenerate_post: {str(e)}", exc_info=True)
        await status_message.edit_text(f"❌ Ошибка при перегенерации: {str(e)[:200]}")
        return None

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard button presses."""
    query = update.callback_query
    await query.answer()

    user_id = str(query.from_user.id)

    # Filter by Admin ID if set
    if ADMIN_ID and user_id != str(ADMIN_ID):
        logger.warning(f"Unauthorized callback attempt by {user_id}")
        return

    data = query.data
    action, message_id = data.split(":", 1)
    logger.info(f"Callback: action={action}, message_id={message_id}")

    # Get generation data
    gen_data = get_generation(context, int(message_id))
    if not gen_data:
        await query.message.reply_text("❌ Данные о генерации не найдены. Попробуйте сгенерировать пост заново.")
        return

    if action == "shorter":
        await regenerate_post(update, context, gen_data, "shorter")
    elif action == "regenerate":
        await regenerate_post(update, context, gen_data, "regenerate")
    elif action == "edits":
        # Set waiting for edits state
        context.user_data['waiting_for_edits'] = int(message_id)
        await query.message.reply_text("✏️ Напишите, какие правки внести в пост:")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = str(user.id)
    user_text = update.message.text

    logger.info(f"Received message from {user.username} ({user_id}): {user_text[:50]}...")

    # Filter by Admin ID if set
    if ADMIN_ID and user_id != str(ADMIN_ID):
        logger.warning(f"Unauthorized access attempt by {user_id}")
        # Optionally reply to unauthorized users, or just ignore
        # await update.message.reply_text("⛔️ У вас нет доступа к этому боту.")
        return

    if not user_text:
        return

    # Check if user is providing custom edits after pressing "📝 Правки" button
    if context.user_data.get('waiting_for_edits'):
        message_id = context.user_data.pop('waiting_for_edits')
        gen_data = get_generation(context, message_id)
        if gen_data:
            await regenerate_post(update, context, gen_data, "custom", custom_edits=user_text)
            return
        else:
            await update.message.reply_text("❌ Данные о генерации не найдены.")
            return

    # Check if this is a reply to a generated post
    if update.message.reply_to_message:
        replied_message_id = update.message.reply_to_message.message_id
        gen_data = get_generation(context, replied_message_id)
        if gen_data:
            # Check for reply commands
            text_lower = user_text.lower().strip()
            if text_lower in ['короче', 'shorter']:
                await regenerate_post(update, context, gen_data, "shorter")
                return
            elif text_lower in ['другой', 'еще', 'ещe', 'еще раз', 'другой вариант']:
                await regenerate_post(update, context, gen_data, "regenerate")
                return
            else:
                # Treat as custom edits
                await regenerate_post(update, context, gen_data, "custom", custom_edits=user_text)
                return

    # Send typing action and status message
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    status_message = await update.message.reply_text("🧠 Читаю статью...")

    # Start status updater
    stop_event = asyncio.Event()
    status_task = asyncio.create_task(update_status_periodically(status_message, stop_event))

    try:
        # Extract content
        content = extract_content(user_text)

        # Check if extraction failed
        if content is None:
            stop_event.set()
            status_task.cancel()
            await status_message.edit_text("❌ Не удалось извлечь контент из ссылки. Сайт может блокировать парсинг или использовать JS-рендеринг.")
            return

        system_prompt = get_prompt()

        # Save extracted content to debug file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)

        extracted_file = debug_dir / f"{user_id}_{timestamp}_extracted.md"
        with open(extracted_file, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved extracted content to {extracted_file}")

        # Call LLM in thread to not block
        logger.info(f"Calling LLM ({LLM_MODEL})...")
        response = await asyncio.to_thread(
            completion,
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            api_key=OPENROUTER_API_KEY
        )

        reply_text = response.choices[0].message.content
        logger.info(f"Successfully received response from LLM ({len(reply_text)} chars)")
        logger.info(f"LLM Response:\n{reply_text}")

        # Stop status updater
        stop_event.set()
        status_task.cancel()

        # Save LLM response to debug file
        response_file = debug_dir / f"{user_id}_{timestamp}_response.md"
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(reply_text)
        logger.info(f"Saved LLM response to {response_file}")

        # Delete status message and send response as reply to original message with keyboard
        await status_message.delete()
        try:
            html_text = markdown_to_html(reply_text)
            sent_message = await update.message.reply_text(
                html_text,
                parse_mode='HTML',
                do_quote=True,
                reply_markup=create_post_keyboard(0)  # Placeholder, will update
            )
        except Exception as parse_error:
            logger.warning(f"HTML parse failed, sending as plain text: {parse_error}")
            sent_message = await update.message.reply_text(
                reply_text,
                do_quote=True,
                reply_markup=create_post_keyboard(0)
            )

        # Update keyboard with correct message_id and save generation
        await sent_message.edit_reply_markup(reply_markup=create_post_keyboard(sent_message.message_id))
        save_generation(context, sent_message.message_id, content, reply_text)

    except Exception as e:
        # Stop status updater
        stop_event.set()
        status_task.cancel()

        logger.error(f"Error in handle_message: {str(e)}", exc_info=True)
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            friendly_error = "🔔 Лимит запросов исчерпан (Quota Exceeded). Пожалуйста, подождите или проверьте биллинг в OpenRouter."
        else:
            friendly_error = f"❌ Произошла ошибка: {error_msg[:200]}"

        await status_message.edit_text(friendly_error)

async def fetch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = str(user.id)
    
    logger.info(f"Received /fetch command from {user.username} ({user_id})")

    # Filter by Admin ID if set
    if ADMIN_ID and user_id != str(ADMIN_ID):
        logger.warning(f"Unauthorized /fetch attempt by {user_id}")
        return

    status_message = await update.message.reply_text("⏳ Запускаю парсинг историй... (это может занять время)")

    try:
        # Run the fetch script using subprocess
        # Assumes the script is at scripts/fetch_stories.py relative to current dir
        script_path = Path("scripts/fetch_stories.py")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Fetch script output: {result.stdout}")
        
        # The script saves to output/stories.xlsx
        output_file = Path("output/stories.xlsx")
        
        if output_file.exists():
            await status_message.edit_text("✅ Парсинг завершен. Отправляю файл...")
            await update.message.reply_document(
                document=open(output_file, "rb"),
                filename="stories.xlsx",
                caption="Вот свежий список историй 📂"
            )
        else:
            await status_message.edit_text("⚠️ Скрипт завершился, но файл не найден.")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running fetch script: {e.stderr}")
        await status_message.edit_text(f"❌ Ошибка при выполнении скрипта:\n{e.stderr[:200]}")
    except Exception as e:
        logger.error(f"Error in fetch_command: {e}", exc_info=True)
        await status_message.edit_text(f"❌ Неизвестная ошибка: {str(e)}")

if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Error: TELEGRAM_BOT_TOKEN not found in .env")
        exit(1)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("fetch", fetch_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    logger.info(f"Bot is starting... (Admin: {ADMIN_ID if ADMIN_ID else 'None'})")
    app.run_polling()
