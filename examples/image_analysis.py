"""Image analysis with multi-modal support and optimization.

Demonstrates:
- Image type for vision-capable models
- Automatic image optimization for API limits
- Loading images from paths and URLs
- Asking questions about images
"""

from spellcrafting import spell, Image, Config


# --- Spells ---


@spell
def describe(image: Image) -> str:
    """Describe what you see in this image in detail."""
    ...


@spell
def extract_text(image: Image) -> str:
    """Extract any text visible in this image (OCR)."""
    ...


@spell
def answer_about(image: Image, question: str) -> str:
    """Answer a question about the image."""
    ...


# --- Demo ---

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Configure with a vision-capable model
    config = Config(
        models={
            "default": {"model": "anthropic:claude-sonnet-4-5"},
        }
    )

    with config:
        print("=" * 60)
        print("SPELLCRAFTING IMAGE ANALYSIS DEMO")
        print("=" * 60)

        # Check if user provided an image path
        if len(sys.argv) > 1:
            image_path = Path(sys.argv[1])
            if not image_path.exists():
                print(f"Error: Image not found: {image_path}")
                sys.exit(1)

            # Load image with automatic optimization (default)
            # Large images are resized to fit API limits (1.15MP, 1568px max)
            print(f"\nLoading image: {image_path}")
            img = Image.from_path(image_path)
            print(f"Image loaded: {img}")

            # Describe the image
            print("\n" + "-" * 40)
            print("DESCRIPTION:")
            print("-" * 40)
            description = describe(img)
            print(description)

            # Extract text (if any)
            print("\n" + "-" * 40)
            print("TEXT EXTRACTION (OCR):")
            print("-" * 40)
            text = extract_text(img)
            print(text if text.strip() else "(No text found)")

            # Answer a question
            print("\n" + "-" * 40)
            print("Q&A:")
            print("-" * 40)
            answer = answer_about(img, "What colors are most prominent in this image?")
            print(f"Q: What colors are most prominent?\nA: {answer}")

        else:
            # Demo with a public image URL
            print("\nNo image path provided. Using a public test image URL.")
            print("Usage: python examples/image_analysis.py <image_path>")
            print("\n" + "-" * 40)
            print("URL EXAMPLE:")
            print("-" * 40)

            # Note: URL images are not optimized locally - sent as-is to API
            img = Image.from_url(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
                "PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
            )
            print(f"Image: {img}")

            description = describe(img)
            print(f"\nDescription:\n{description}")

        # Show optimization examples (informational)
        print("\n" + "=" * 60)
        print("IMAGE OPTIMIZATION OPTIONS")
        print("=" * 60)
        print("""
# Default: auto-optimize for API limits
img = Image.from_path("large_photo.jpg")

# Custom optimization parameters
img = Image.from_path(
    "photo.jpg",
    max_pixels=1_150_000,    # Total pixel budget (Anthropic's 1.15MP limit)
    max_dimension=1568,       # Max width OR height
    quality=85,               # JPEG quality (1-100)
)

# Disable optimization for precise work (e.g., diagrams, screenshots)
img = Image.from_path("diagram.png", optimize=False)

# OpenAI detail parameter (passed through to API)
img = Image.from_path("photo.jpg", detail="high")

# From raw bytes
with open("image.png", "rb") as f:
    img = Image.from_bytes(f.read())
""")
