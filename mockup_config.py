"""Configuration for screenshot mockup generation."""

# Image sizing
MAX_IMAGE_SIZE = 1024  # Maximum size for the longest side of the image (ALWAYS)

# Padding around the image in pixels (fixed on all sides)
MOCKUP_PADDING = 50  # Fixed: 50px on all sides

# Background image (optional)
# Set to None to use default gradient, or provide path to custom background
BACKGROUND_IMAGE = "backgrounds/photo_2026-01-31 11.37.23.jpeg"  # Your custom background

# You can create different background images and switch between them
# Examples:
# BACKGROUND_IMAGE = "backgrounds/sunset.png"
# BACKGROUND_IMAGE = "backgrounds/dark_gradient.png"
# BACKGROUND_IMAGE = None  # Use default gradient
