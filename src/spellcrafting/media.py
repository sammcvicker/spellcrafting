"""Multi-modal input support for spellcrafting.

This module provides Image, Audio, Document, and Video types that can be passed
directly to spells for use with vision and multi-modal models.

Example:
    from spellcrafting import spell
    from spellcrafting.media import Image

    @spell
    def describe(image: Image) -> str:
        '''Describe what you see in this image.'''
        ...

    img = Image.from_path("photo.png")
    result = describe(img)
"""

from __future__ import annotations

import mimetypes
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    VideoUrl,
)

# Type alias for what pydantic_ai accepts as multimodal content
PydanticAIContent = ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent


# ---------------------------------------------------------------------------
# Magic bytes for format detection
# ---------------------------------------------------------------------------

# Image magic bytes
_IMAGE_MAGIC = {
    b"\x89PNG\r\n\x1a\n": ("image/png", "png"),
    b"\xff\xd8\xff": ("image/jpeg", "jpeg"),
    b"GIF87a": ("image/gif", "gif"),
    b"GIF89a": ("image/gif", "gif"),
    # WebP: RIFF....WEBP (bytes 0-3 and 8-11)
}

# Audio magic bytes
_AUDIO_MAGIC = {
    b"ID3": ("audio/mpeg", "mp3"),  # MP3 with ID3 tag
    b"\xff\xfb": ("audio/mpeg", "mp3"),  # MP3 without ID3
    b"\xff\xfa": ("audio/mpeg", "mp3"),  # MP3 variant
    b"\xff\xf3": ("audio/mpeg", "mp3"),  # MP3 variant
    b"\xff\xf2": ("audio/mpeg", "mp3"),  # MP3 variant
    b"OggS": ("audio/ogg", "ogg"),
    b"fLaC": ("audio/flac", "flac"),
    b"RIFF": ("audio/wav", "wav"),  # Could be WAV (check for WAVE)
    b"FORM": ("audio/aiff", "aiff"),  # AIFF
}

# Document magic bytes
_DOCUMENT_MAGIC = {
    b"%PDF": ("application/pdf", "pdf"),
    b"PK\x03\x04": ("application/zip", "zip"),  # Could be docx/xlsx
}

# Video magic bytes
_VIDEO_MAGIC = {
    b"\x1aE\xdf\xa3": ("video/x-matroska", "mkv"),  # Matroska/WebM
    b"\x00\x00\x00\x1cftyp": ("video/mp4", "mp4"),
    b"\x00\x00\x00\x18ftyp": ("video/mp4", "mp4"),
    b"\x00\x00\x00\x20ftyp": ("video/mp4", "mp4"),
    b"FLV\x01": ("video/x-flv", "flv"),
}


def _detect_format_from_bytes(data: bytes) -> tuple[str, str] | None:
    """Detect media type and format from magic bytes.

    Args:
        data: The raw bytes to analyze (at least 32 bytes recommended)

    Returns:
        Tuple of (media_type, format) if detected, None otherwise
    """
    if len(data) < 4:
        return None

    # Check WebP (RIFF + WEBP at offset 8)
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return ("image/webp", "webp")

    # Check WAV (RIFF + WAVE at offset 8)
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WAVE":
        return ("audio/wav", "wav")

    # Check MP4/MOV (ftyp box)
    if len(data) >= 8 and data[4:8] == b"ftyp":
        # Check brand
        if len(data) >= 12:
            brand = data[8:12]
            if brand in (b"qt  ", b"mqt "):
                return ("video/quicktime", "mov")
            if brand in (b"isom", b"iso2", b"mp41", b"mp42", b"M4V ", b"avc1"):
                return ("video/mp4", "mp4")
            if brand == b"M4A ":
                return ("audio/mp4", "m4a")
            # Default to mp4 for unknown ftyp
            return ("video/mp4", "mp4")

    # Check Office Open XML formats (docx, xlsx, pptx)
    if data[:4] == b"PK\x03\x04":
        # These are ZIP files, need deeper inspection
        # For now, return generic application/zip
        # Actual format detection would require looking at [Content_Types].xml
        return ("application/zip", "zip")

    # Check image formats
    for magic, result in _IMAGE_MAGIC.items():
        if data.startswith(magic):
            return result

    # Check audio formats (skip RIFF as handled above)
    for magic, result in _AUDIO_MAGIC.items():
        if magic == b"RIFF":
            continue  # Already handled
        if data.startswith(magic):
            return result

    # Check document formats
    for magic, result in _DOCUMENT_MAGIC.items():
        if magic == b"PK\x03\x04":
            continue  # Already handled
        if data.startswith(magic):
            return result

    # Check video formats
    for magic, result in _VIDEO_MAGIC.items():
        if data.startswith(magic):
            return result

    return None


def _detect_format_from_extension(path: str | Path) -> tuple[str, str] | None:
    """Detect media type from file extension.

    Args:
        path: File path to analyze

    Returns:
        Tuple of (media_type, format) if detected, None otherwise
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Common mappings
    extension_map = {
        # Images
        ".png": ("image/png", "png"),
        ".jpg": ("image/jpeg", "jpeg"),
        ".jpeg": ("image/jpeg", "jpeg"),
        ".gif": ("image/gif", "gif"),
        ".webp": ("image/webp", "webp"),
        # Audio
        ".mp3": ("audio/mpeg", "mp3"),
        ".wav": ("audio/wav", "wav"),
        ".ogg": ("audio/ogg", "ogg"),
        ".flac": ("audio/flac", "flac"),
        ".aiff": ("audio/aiff", "aiff"),
        ".aif": ("audio/aiff", "aiff"),
        ".aac": ("audio/aac", "aac"),
        ".m4a": ("audio/mp4", "m4a"),
        # Documents
        ".pdf": ("application/pdf", "pdf"),
        ".txt": ("text/plain", "txt"),
        ".csv": ("text/csv", "csv"),
        ".html": ("text/html", "html"),
        ".htm": ("text/html", "html"),
        ".md": ("text/markdown", "md"),
        ".docx": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        ),
        ".xlsx": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xlsx",
        ),
        ".doc": ("application/msword", "doc"),
        ".xls": ("application/vnd.ms-excel", "xls"),
        # Video
        ".mp4": ("video/mp4", "mp4"),
        ".mov": ("video/quicktime", "mov"),
        ".webm": ("video/webm", "webm"),
        ".mkv": ("video/x-matroska", "mkv"),
        ".avi": ("video/x-msvideo", "avi"),
        ".flv": ("video/x-flv", "flv"),
        ".wmv": ("video/x-ms-wmv", "wmv"),
        ".mpeg": ("video/mpeg", "mpeg"),
        ".mpg": ("video/mpeg", "mpeg"),
        ".3gp": ("video/3gpp", "3gp"),
    }

    return extension_map.get(suffix)


# ---------------------------------------------------------------------------
# Image optimization
# ---------------------------------------------------------------------------

# Default optimization parameters (optimal for Anthropic models)
DEFAULT_MAX_PIXELS = 1_150_000  # Anthropic's recommended max: 1.15 megapixels
DEFAULT_MAX_DIMENSION = 1568  # Max width or height
DEFAULT_QUALITY = 85  # JPEG quality (1-100)

# Type for OpenAI detail parameter
DetailLevel = Literal["auto", "low", "high"]


def _check_pillow_available() -> bool:
    """Check if Pillow is available for image optimization.

    Returns:
        True if Pillow is installed, False otherwise
    """
    try:
        from PIL import Image as PILImage

        return True
    except ImportError:
        return False


def _optimize_image(
    data: bytes,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    quality: int = DEFAULT_QUALITY,
) -> tuple[bytes, str]:
    """Resize and re-encode image for optimal API usage.

    This function resizes large images to fit within API limits and reduce
    token costs. Images are resized using high-quality Lanczos resampling.

    Args:
        data: Raw image bytes
        max_pixels: Maximum total pixels (width * height). Default 1,150,000.
        max_dimension: Maximum width or height in pixels. Default 1568.
        quality: JPEG quality for re-encoding (1-100). Default 85.

    Returns:
        Tuple of (optimized_bytes, media_type)

    Raises:
        ImportError: If Pillow is not installed
    """
    from PIL import Image as PILImage

    img = PILImage.open(BytesIO(data))
    original_format = img.format

    # Get original dimensions
    width, height = img.size
    current_pixels = width * height

    # Calculate resize ratio
    scale = 1.0

    # Check dimension limits
    if width > max_dimension:
        scale = min(scale, max_dimension / width)
    if height > max_dimension:
        scale = min(scale, max_dimension / height)

    # Check pixel limits (after dimension scaling)
    if current_pixels * (scale**2) > max_pixels:
        scale = min(scale, (max_pixels / current_pixels) ** 0.5)

    # Only resize if needed
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Ensure minimum size of 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

    # Re-encode the image
    buffer = BytesIO()

    # Convert RGBA to RGB for JPEG (JPEG doesn't support alpha)
    if original_format in ("JPEG", "JPG") or (
        original_format is None and img.mode in ("RGB", "L")
    ):
        # Ensure RGB mode for JPEG
        if img.mode == "RGBA":
            # Create white background and paste image
            background = PILImage.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha as mask
            img = background
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    else:
        # For PNG, GIF, WebP, etc., preserve format or use PNG
        if img.mode == "P" and "transparency" in img.info:
            # Preserve palette transparency
            img = img.convert("RGBA")
        img.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"


# ---------------------------------------------------------------------------
# MediaType Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MediaType(Protocol):
    """Protocol for all media types.

    All media types (Image, Audio, Document, Video) implement this protocol,
    allowing spellcrafting to detect and handle them uniformly.
    """

    def to_pydantic_ai(self) -> PydanticAIContent:
        """Convert to a pydantic_ai content type.

        Returns:
            A pydantic_ai content type (ImageUrl, AudioUrl, DocumentUrl,
            VideoUrl, or BinaryContent)
        """
        ...


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


class Image:
    """Image input for vision-capable models.

    Use the class methods to create Image instances:
    - Image.from_path() - Load from a local file
    - Image.from_url() - Reference an image URL
    - Image.from_bytes() - Create from raw bytes

    By default, images are automatically optimized to fit within API limits
    (1.15 megapixels, 1568px max dimension) to reduce token costs and improve
    reliability. This requires Pillow to be installed (`pip install spellcrafting[images]`).

    Example:
        # Auto-optimized for API limits (default)
        img = Image.from_path("huge_photo.jpg")  # 4000x3000 -> auto-resized

        # Explicit control over optimization
        img = Image.from_path(
            "photo.jpg",
            max_pixels=1_150_000,    # Total pixel budget
            max_dimension=1568,       # Max width OR height
            quality=85,               # JPEG quality (1-100)
        )

        # Disable optimization
        img = Image.from_path("photo.jpg", optimize=False)

        # OpenAI detail level (passed through to API)
        img = Image.from_path("photo.jpg", detail="high")
    """

    __slots__ = ("_content", "_media_type", "_identifier", "_detail")

    def __init__(
        self,
        content: PydanticAIContent,
        media_type: str | None = None,
        identifier: str | None = None,
        detail: DetailLevel | None = None,
    ) -> None:
        """Internal constructor. Use from_path, from_url, or from_bytes instead."""
        self._content = content
        self._media_type = media_type
        self._identifier = identifier
        self._detail = detail

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        media_type: str | None = None,
        optimize: bool = True,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
        quality: int = DEFAULT_QUALITY,
        detail: DetailLevel | None = None,
    ) -> Image:
        """Create an Image from a local file path.

        By default, images are automatically optimized to fit within API limits.
        This requires Pillow to be installed. If Pillow is not available, a warning
        is issued and the original image is used.

        Args:
            path: Path to the image file
            media_type: Optional media type override (auto-detected if not provided)
            optimize: Whether to optimize the image for API limits. Default True.
            max_pixels: Maximum total pixels (width * height). Default 1,150,000.
            max_dimension: Maximum width or height in pixels. Default 1568.
            quality: JPEG quality for re-encoding (1-100). Default 85.
            detail: OpenAI detail level ("auto", "low", "high"). Passed through to API.

        Returns:
            Image instance

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the format cannot be detected
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        # Read raw bytes for optimization
        data = path.read_bytes()

        # Attempt optimization if enabled
        if optimize:
            if _check_pillow_available():
                try:
                    data, detected_media_type = _optimize_image(
                        data,
                        max_pixels=max_pixels,
                        max_dimension=max_dimension,
                        quality=quality,
                    )
                    if media_type is None:
                        media_type = detected_media_type
                except Exception as e:
                    # If optimization fails, fall back to original
                    warnings.warn(
                        f"Image optimization failed: {e}. Using original image.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    "Pillow not installed. Install with: pip install spellcrafting[images] "
                    "for automatic image optimization. Sending original image.",
                    UserWarning,
                    stacklevel=2,
                )

        # Determine media type if not set
        if media_type is None:
            result = _detect_format_from_bytes(data)
            if result:
                media_type = result[0]
            else:
                # Fallback to extension-based detection
                result = _detect_format_from_extension(path)
                if result:
                    media_type = result[0]

        content = BinaryContent(data=data, media_type=media_type, identifier=str(path))
        return cls(content, media_type=media_type, identifier=str(path), detail=detail)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        media_type: str | None = None,
        force_download: bool = False,
        detail: DetailLevel | None = None,
    ) -> Image:
        """Create an Image from a URL.

        Note: URL-based images are not optimized locally. The image is sent
        as-is to the API, which handles any resizing. Use `force_download=True`
        if you need local processing.

        Args:
            url: URL of the image
            media_type: Optional media type override
            force_download: If True, download the image before sending to API
            detail: OpenAI detail level ("auto", "low", "high"). Passed through to API.

        Returns:
            Image instance
        """
        content = ImageUrl(url=url, media_type=media_type, force_download=force_download)
        return cls(content, media_type=media_type, identifier=url, detail=detail)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: str | None = None,
        identifier: str | None = None,
        optimize: bool = True,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
        quality: int = DEFAULT_QUALITY,
        detail: DetailLevel | None = None,
    ) -> Image:
        """Create an Image from raw bytes.

        By default, images are automatically optimized to fit within API limits.
        This requires Pillow to be installed. If Pillow is not available, a warning
        is issued and the original image is used.

        Args:
            data: Raw image bytes
            media_type: Media type (auto-detected if not provided)
            identifier: Optional identifier for the image
            optimize: Whether to optimize the image for API limits. Default True.
            max_pixels: Maximum total pixels (width * height). Default 1,150,000.
            max_dimension: Maximum width or height in pixels. Default 1568.
            quality: JPEG quality for re-encoding (1-100). Default 85.
            detail: OpenAI detail level ("auto", "low", "high"). Passed through to API.

        Returns:
            Image instance

        Raises:
            ValueError: If media_type is not provided and cannot be detected
        """
        # Attempt optimization if enabled
        if optimize:
            if _check_pillow_available():
                try:
                    data, detected_media_type = _optimize_image(
                        data,
                        max_pixels=max_pixels,
                        max_dimension=max_dimension,
                        quality=quality,
                    )
                    if media_type is None:
                        media_type = detected_media_type
                except Exception as e:
                    # If optimization fails, fall back to original
                    warnings.warn(
                        f"Image optimization failed: {e}. Using original image.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    "Pillow not installed. Install with: pip install spellcrafting[images] "
                    "for automatic image optimization. Sending original image.",
                    UserWarning,
                    stacklevel=2,
                )

        # Determine media type if not set
        if media_type is None:
            result = _detect_format_from_bytes(data)
            if result:
                media_type = result[0]
            else:
                raise ValueError(
                    "Could not detect image format from bytes. "
                    "Please provide media_type explicitly."
                )

        content = BinaryContent(data=data, media_type=media_type, identifier=identifier)
        return cls(content, media_type=media_type, identifier=identifier, detail=detail)

    def to_pydantic_ai(self) -> PydanticAIContent:
        """Convert to pydantic_ai content type.

        Returns:
            ImageUrl or BinaryContent for pydantic_ai
        """
        return self._content

    @property
    def media_type(self) -> str | None:
        """The media type (e.g., 'image/png')."""
        return self._media_type

    @property
    def identifier(self) -> str | None:
        """Optional identifier (file path or URL)."""
        return self._identifier

    @property
    def detail(self) -> DetailLevel | None:
        """OpenAI detail level (auto, low, high)."""
        return self._detail

    def __repr__(self) -> str:
        if self._identifier:
            return f"Image({self._identifier!r})"
        if self._media_type:
            return f"Image(media_type={self._media_type!r})"
        return "Image(...)"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class Audio:
    """Audio input for audio-capable models.

    Use the class methods to create Audio instances:
    - Audio.from_path() - Load from a local file
    - Audio.from_url() - Reference an audio URL
    - Audio.from_bytes() - Create from raw bytes

    Example:
        audio = Audio.from_path("recording.mp3")
        audio = Audio.from_url("https://example.com/audio.wav")
        audio = Audio.from_bytes(raw_bytes, media_type="audio/mpeg")
    """

    __slots__ = ("_content", "_media_type", "_identifier")

    def __init__(
        self,
        content: PydanticAIContent,
        media_type: str | None = None,
        identifier: str | None = None,
    ) -> None:
        """Internal constructor. Use from_path, from_url, or from_bytes instead."""
        self._content = content
        self._media_type = media_type
        self._identifier = identifier

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        media_type: str | None = None,
    ) -> Audio:
        """Create an Audio from a local file path.

        Args:
            path: Path to the audio file
            media_type: Optional media type override (auto-detected if not provided)

        Returns:
            Audio instance

        Raises:
            FileNotFoundError: If the file does not exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        content = BinaryContent.from_path(path)

        if media_type is None:
            if hasattr(content, "media_type"):
                media_type = content.media_type
            else:
                result = _detect_format_from_extension(path)
                if result:
                    media_type = result[0]

        return cls(content, media_type=media_type, identifier=str(path))

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        media_type: str | None = None,
        force_download: bool = False,
    ) -> Audio:
        """Create an Audio from a URL.

        Args:
            url: URL of the audio
            media_type: Optional media type override
            force_download: If True, download the audio before sending to API

        Returns:
            Audio instance
        """
        content = AudioUrl(url=url, media_type=media_type, force_download=force_download)
        return cls(content, media_type=media_type, identifier=url)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: str | None = None,
        identifier: str | None = None,
    ) -> Audio:
        """Create an Audio from raw bytes.

        Args:
            data: Raw audio bytes
            media_type: Media type (auto-detected if not provided)
            identifier: Optional identifier for the audio

        Returns:
            Audio instance

        Raises:
            ValueError: If media_type is not provided and cannot be detected
        """
        if media_type is None:
            result = _detect_format_from_bytes(data)
            if result:
                media_type = result[0]
            else:
                raise ValueError(
                    "Could not detect audio format from bytes. "
                    "Please provide media_type explicitly."
                )

        content = BinaryContent(data=data, media_type=media_type, identifier=identifier)
        return cls(content, media_type=media_type, identifier=identifier)

    def to_pydantic_ai(self) -> PydanticAIContent:
        """Convert to pydantic_ai content type.

        Returns:
            AudioUrl or BinaryContent for pydantic_ai
        """
        return self._content

    @property
    def media_type(self) -> str | None:
        """The media type (e.g., 'audio/mpeg')."""
        return self._media_type

    @property
    def identifier(self) -> str | None:
        """Optional identifier (file path or URL)."""
        return self._identifier

    def __repr__(self) -> str:
        if self._identifier:
            return f"Audio({self._identifier!r})"
        if self._media_type:
            return f"Audio(media_type={self._media_type!r})"
        return "Audio(...)"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class Document:
    """Document input for document-capable models.

    Supports PDF, text, CSV, HTML, Markdown, and Office documents.

    Use the class methods to create Document instances:
    - Document.from_path() - Load from a local file
    - Document.from_url() - Reference a document URL
    - Document.from_bytes() - Create from raw bytes

    Example:
        doc = Document.from_path("report.pdf")
        doc = Document.from_url("https://example.com/doc.pdf")
        doc = Document.from_bytes(raw_bytes, media_type="application/pdf")
    """

    __slots__ = ("_content", "_media_type", "_identifier")

    def __init__(
        self,
        content: PydanticAIContent,
        media_type: str | None = None,
        identifier: str | None = None,
    ) -> None:
        """Internal constructor. Use from_path, from_url, or from_bytes instead."""
        self._content = content
        self._media_type = media_type
        self._identifier = identifier

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        media_type: str | None = None,
    ) -> Document:
        """Create a Document from a local file path.

        Args:
            path: Path to the document file
            media_type: Optional media type override (auto-detected if not provided)

        Returns:
            Document instance

        Raises:
            FileNotFoundError: If the file does not exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        content = BinaryContent.from_path(path)

        if media_type is None:
            if hasattr(content, "media_type"):
                media_type = content.media_type
            else:
                result = _detect_format_from_extension(path)
                if result:
                    media_type = result[0]

        return cls(content, media_type=media_type, identifier=str(path))

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        media_type: str | None = None,
        force_download: bool = False,
    ) -> Document:
        """Create a Document from a URL.

        Args:
            url: URL of the document
            media_type: Optional media type override
            force_download: If True, download the document before sending to API

        Returns:
            Document instance
        """
        content = DocumentUrl(
            url=url, media_type=media_type, force_download=force_download
        )
        return cls(content, media_type=media_type, identifier=url)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: str | None = None,
        identifier: str | None = None,
    ) -> Document:
        """Create a Document from raw bytes.

        Args:
            data: Raw document bytes
            media_type: Media type (auto-detected if not provided)
            identifier: Optional identifier for the document

        Returns:
            Document instance

        Raises:
            ValueError: If media_type is not provided and cannot be detected
        """
        if media_type is None:
            result = _detect_format_from_bytes(data)
            if result:
                media_type = result[0]
            else:
                raise ValueError(
                    "Could not detect document format from bytes. "
                    "Please provide media_type explicitly."
                )

        content = BinaryContent(data=data, media_type=media_type, identifier=identifier)
        return cls(content, media_type=media_type, identifier=identifier)

    def to_pydantic_ai(self) -> PydanticAIContent:
        """Convert to pydantic_ai content type.

        Returns:
            DocumentUrl or BinaryContent for pydantic_ai
        """
        return self._content

    @property
    def media_type(self) -> str | None:
        """The media type (e.g., 'application/pdf')."""
        return self._media_type

    @property
    def identifier(self) -> str | None:
        """Optional identifier (file path or URL)."""
        return self._identifier

    def __repr__(self) -> str:
        if self._identifier:
            return f"Document({self._identifier!r})"
        if self._media_type:
            return f"Document(media_type={self._media_type!r})"
        return "Document(...)"


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


class Video:
    """Video input for video-capable models.

    Use the class methods to create Video instances:
    - Video.from_path() - Load from a local file
    - Video.from_url() - Reference a video URL
    - Video.from_bytes() - Create from raw bytes

    Example:
        video = Video.from_path("clip.mp4")
        video = Video.from_url("https://example.com/video.mp4")
        video = Video.from_bytes(raw_bytes, media_type="video/mp4")

    Note:
        Video support varies by provider. Currently only Google/Gemini
        supports video inputs natively.
    """

    __slots__ = ("_content", "_media_type", "_identifier")

    def __init__(
        self,
        content: PydanticAIContent,
        media_type: str | None = None,
        identifier: str | None = None,
    ) -> None:
        """Internal constructor. Use from_path, from_url, or from_bytes instead."""
        self._content = content
        self._media_type = media_type
        self._identifier = identifier

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        media_type: str | None = None,
    ) -> Video:
        """Create a Video from a local file path.

        Args:
            path: Path to the video file
            media_type: Optional media type override (auto-detected if not provided)

        Returns:
            Video instance

        Raises:
            FileNotFoundError: If the file does not exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        content = BinaryContent.from_path(path)

        if media_type is None:
            if hasattr(content, "media_type"):
                media_type = content.media_type
            else:
                result = _detect_format_from_extension(path)
                if result:
                    media_type = result[0]

        return cls(content, media_type=media_type, identifier=str(path))

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        media_type: str | None = None,
        force_download: bool = False,
    ) -> Video:
        """Create a Video from a URL.

        Args:
            url: URL of the video
            media_type: Optional media type override
            force_download: If True, download the video before sending to API

        Returns:
            Video instance
        """
        content = VideoUrl(url=url, media_type=media_type, force_download=force_download)
        return cls(content, media_type=media_type, identifier=url)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: str | None = None,
        identifier: str | None = None,
    ) -> Video:
        """Create a Video from raw bytes.

        Args:
            data: Raw video bytes
            media_type: Media type (auto-detected if not provided)
            identifier: Optional identifier for the video

        Returns:
            Video instance

        Raises:
            ValueError: If media_type is not provided and cannot be detected
        """
        if media_type is None:
            result = _detect_format_from_bytes(data)
            if result:
                media_type = result[0]
            else:
                raise ValueError(
                    "Could not detect video format from bytes. "
                    "Please provide media_type explicitly."
                )

        content = BinaryContent(data=data, media_type=media_type, identifier=identifier)
        return cls(content, media_type=media_type, identifier=identifier)

    def to_pydantic_ai(self) -> PydanticAIContent:
        """Convert to pydantic_ai content type.

        Returns:
            VideoUrl or BinaryContent for pydantic_ai
        """
        return self._content

    @property
    def media_type(self) -> str | None:
        """The media type (e.g., 'video/mp4')."""
        return self._media_type

    @property
    def identifier(self) -> str | None:
        """Optional identifier (file path or URL)."""
        return self._identifier

    def __repr__(self) -> str:
        if self._identifier:
            return f"Video({self._identifier!r})"
        if self._media_type:
            return f"Video(media_type={self._media_type!r})"
        return "Video(...)"


# ---------------------------------------------------------------------------
# Helper to check if a value is a media type
# ---------------------------------------------------------------------------


def is_media_type(value: Any) -> bool:
    """Check if a value is a spellcrafting media type.

    Args:
        value: Value to check

    Returns:
        True if value is an Image, Audio, Document, or Video instance
    """
    return isinstance(value, (Image, Audio, Document, Video))


__all__ = [
    # Protocol
    "MediaType",
    # Media types
    "Image",
    "Audio",
    "Document",
    "Video",
    # Helper
    "is_media_type",
    # Image optimization constants
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_MAX_DIMENSION",
    "DEFAULT_QUALITY",
    "DetailLevel",
]
