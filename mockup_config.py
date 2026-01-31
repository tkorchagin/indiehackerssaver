"""Configuration for screenshot mockup generation."""

# Image sizing
MAX_IMAGE_SIZE = 1024  # Maximum size for the longest side of the image

# Padding around the image in pixels
MOCKUP_PADDING = 80  # Default: 80px

# Background image (optional)
# Set to None to use default gradient, or provide path to custom background
BACKGROUND_IMAGE = None  # Example: "backgrounds/gradient1.png"

# You can create different background images and switch between them
# Examples:
# BACKGROUND_IMAGE = "backgrounds/sunset.png"
# BACKGROUND_IMAGE = "backgrounds/dark_gradient.png"
# BACKGROUND_IMAGE = None  # Use default gradient
