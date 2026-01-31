# 🚀 Быстрый старт - Настройка мокапов

## Основные настройки (mockup_config.py)

```python
MAX_IMAGE_SIZE = 1024  # Макс. размер картинки
MOCKUP_PADDING = 80    # Отступы в пикселях
BACKGROUND_IMAGE = None  # Фон (None = градиент)
```

## Примеры:

### Instagram Stories:
```python
MAX_IMAGE_SIZE = 1080
MOCKUP_PADDING = 100
```

### Большие посты:
```python
MAX_IMAGE_SIZE = 1920
MOCKUP_PADDING = 150
```

### Быстрая генерация:
```python
MAX_IMAGE_SIZE = 800
MOCKUP_PADDING = 50
```

## Кастомный фон:

1. Положи картинку в `backgrounds/`
2. В `mockup_config.py`:
   ```python
   BACKGROUND_IMAGE = "backgrounds/твой-фон.png"
   ```
3. Перезапусти бота

## Тест без бота:
```bash
source venv/bin/activate
python test_new_mockup.py
```

Подробнее: [MOCKUP_CONFIG_README.md](MOCKUP_CONFIG_README.md)
