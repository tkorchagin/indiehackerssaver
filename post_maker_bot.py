import os
import re
import sys
import logging
import subprocess
import trafilatura
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from litellm import completion

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

    return text

def extract_content(text):
    """Extract content from URL or return original text."""
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    
    if urls:
        url = urls[0]
        logger.info(f"Extracting content from URL: {url}")
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, output_format='markdown')
            if extracted:
                logger.info(f"Successfully extracted {len(extracted)} chars from {url}")
                return extracted
            else:
                logger.warning(f"Trafilatura returned empty content for {url}")
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

    # Send typing action and status message
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    status_message = await update.message.reply_text("⏳ Обрабатываю...")

    try:
        # Extract content
        content = extract_content(user_text)
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

        # Call LLM
        logger.info(f"Calling LLM ({LLM_MODEL})...")
        response = completion(
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
        
        # Save LLM response to debug file
        response_file = debug_dir / f"{user_id}_{timestamp}_response.md"
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(reply_text)
        logger.info(f"Saved LLM response to {response_file}")
        
        # Delete status message and send response as reply to original message
        await status_message.delete()
        try:
            html_text = markdown_to_html(reply_text)
            await update.message.reply_text(html_text, parse_mode='HTML', do_quote=True)
        except Exception as parse_error:
            logger.warning(f"HTML parse failed, sending as plain text: {parse_error}")
            await update.message.reply_text(reply_text, do_quote=True)

    except Exception as e:
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
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    logger.info(f"Bot is starting... (Admin: {ADMIN_ID if ADMIN_ID else 'None'})")
    app.run_polling()
