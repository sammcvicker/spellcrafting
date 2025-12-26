"""Tests for the spellcrafting.media module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, VideoUrl

from spellcrafting.media import (
    Audio,
    Document,
    Image,
    MediaType,
    Video,
    _detect_format_from_bytes,
    _detect_format_from_extension,
    is_media_type,
)


# ---------------------------------------------------------------------------
# Test fixtures - minimal valid file bytes for each format
# ---------------------------------------------------------------------------


@pytest.fixture
def png_bytes():
    """Minimal valid PNG file bytes (1x1 transparent pixel)."""
    return bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D,  # IHDR chunk length
        0x49, 0x48, 0x44, 0x52,  # IHDR
        0x00, 0x00, 0x00, 0x01,  # width = 1
        0x00, 0x00, 0x00, 0x01,  # height = 1
        0x08, 0x02,              # bit depth = 8, color type = 2 (RGB)
        0x00, 0x00, 0x00,        # compression, filter, interlace
        0x90, 0x77, 0x53, 0xDE,  # CRC
        0x00, 0x00, 0x00, 0x0C,  # IDAT chunk length
        0x49, 0x44, 0x41, 0x54,  # IDAT
        0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0xFF, 0x00, 0x05, 0xFE, 0x02, 0xFE,
        0xA3, 0x6C, 0xD4, 0xF5,  # CRC
        0x00, 0x00, 0x00, 0x00,  # IEND chunk length
        0x49, 0x45, 0x4E, 0x44,  # IEND
        0xAE, 0x42, 0x60, 0x82   # CRC
    ])


@pytest.fixture
def jpeg_bytes():
    """Minimal JPEG file bytes (just header)."""
    return bytes([0xFF, 0xD8, 0xFF, 0xE0])


@pytest.fixture
def gif_bytes():
    """GIF89a header bytes."""
    return b"GIF89a"


@pytest.fixture
def webp_bytes():
    """Minimal WebP file header."""
    return b"RIFF\x00\x00\x00\x00WEBP"


@pytest.fixture
def mp3_bytes():
    """MP3 with ID3 tag."""
    return b"ID3\x04\x00\x00"


@pytest.fixture
def mp3_no_id3_bytes():
    """MP3 without ID3 tag (sync word with extra bytes for detection)."""
    # Need at least 4 bytes for reliable detection
    return bytes([0xFF, 0xFB, 0x90, 0x00])


@pytest.fixture
def wav_bytes():
    """WAV file header."""
    return b"RIFF\x00\x00\x00\x00WAVE"


@pytest.fixture
def pdf_bytes():
    """PDF header."""
    return b"%PDF-1.4"


@pytest.fixture
def mp4_bytes():
    """MP4 file header (ftyp box)."""
    return bytes([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70,
                  0x69, 0x73, 0x6F, 0x6D])  # isom brand


# ---------------------------------------------------------------------------
# Format detection tests
# ---------------------------------------------------------------------------


class TestFormatDetection:
    """Tests for magic byte and extension-based format detection."""

    def test_detect_png_from_bytes(self, png_bytes):
        result = _detect_format_from_bytes(png_bytes)
        assert result == ("image/png", "png")

    def test_detect_jpeg_from_bytes(self, jpeg_bytes):
        result = _detect_format_from_bytes(jpeg_bytes)
        assert result == ("image/jpeg", "jpeg")

    def test_detect_gif_from_bytes(self, gif_bytes):
        result = _detect_format_from_bytes(gif_bytes)
        assert result == ("image/gif", "gif")

    def test_detect_webp_from_bytes(self, webp_bytes):
        result = _detect_format_from_bytes(webp_bytes)
        assert result == ("image/webp", "webp")

    def test_detect_mp3_with_id3_from_bytes(self, mp3_bytes):
        result = _detect_format_from_bytes(mp3_bytes)
        assert result == ("audio/mpeg", "mp3")

    def test_detect_mp3_without_id3_from_bytes(self, mp3_no_id3_bytes):
        result = _detect_format_from_bytes(mp3_no_id3_bytes)
        assert result == ("audio/mpeg", "mp3")

    def test_detect_wav_from_bytes(self, wav_bytes):
        result = _detect_format_from_bytes(wav_bytes)
        assert result == ("audio/wav", "wav")

    def test_detect_pdf_from_bytes(self, pdf_bytes):
        result = _detect_format_from_bytes(pdf_bytes)
        assert result == ("application/pdf", "pdf")

    def test_detect_mp4_from_bytes(self, mp4_bytes):
        result = _detect_format_from_bytes(mp4_bytes)
        assert result == ("video/mp4", "mp4")

    def test_detect_unknown_returns_none(self):
        result = _detect_format_from_bytes(b"unknown format data here")
        assert result is None

    def test_detect_short_bytes_returns_none(self):
        result = _detect_format_from_bytes(b"abc")
        assert result is None

    def test_detect_from_extension_png(self):
        result = _detect_format_from_extension("photo.png")
        assert result == ("image/png", "png")

    def test_detect_from_extension_jpg(self):
        result = _detect_format_from_extension("photo.jpg")
        assert result == ("image/jpeg", "jpeg")

    def test_detect_from_extension_jpeg(self):
        result = _detect_format_from_extension("photo.jpeg")
        assert result == ("image/jpeg", "jpeg")

    def test_detect_from_extension_mp3(self):
        result = _detect_format_from_extension("song.mp3")
        assert result == ("audio/mpeg", "mp3")

    def test_detect_from_extension_pdf(self):
        result = _detect_format_from_extension("doc.pdf")
        assert result == ("application/pdf", "pdf")

    def test_detect_from_extension_mp4(self):
        result = _detect_format_from_extension("video.mp4")
        assert result == ("video/mp4", "mp4")

    def test_detect_from_extension_unknown(self):
        result = _detect_format_from_extension("file.xyz")
        assert result is None

    def test_detect_from_extension_path_object(self):
        result = _detect_format_from_extension(Path("image.gif"))
        assert result == ("image/gif", "gif")


# ---------------------------------------------------------------------------
# Image tests
# ---------------------------------------------------------------------------


class TestImage:
    """Tests for the Image class."""

    def test_from_path_creates_image(self, png_bytes, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(png_bytes)

        img = Image.from_path(img_path)

        assert isinstance(img, Image)
        assert img.identifier == str(img_path)
        assert img.media_type == "image/png"

    def test_from_path_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            Image.from_path("/nonexistent/path.png")

    def test_from_path_with_explicit_media_type(self, png_bytes, tmp_path):
        img_path = tmp_path / "test.data"
        img_path.write_bytes(png_bytes)

        img = Image.from_path(img_path, media_type="image/png")

        assert img.media_type == "image/png"

    def test_from_url_creates_image(self):
        url = "https://example.com/image.jpg"
        img = Image.from_url(url)

        assert isinstance(img, Image)
        assert img.identifier == url

    def test_from_url_with_force_download(self):
        url = "https://example.com/image.jpg"
        img = Image.from_url(url, force_download=True)

        content = img.to_pydantic_ai()
        assert isinstance(content, ImageUrl)
        assert content.force_download is True

    def test_from_bytes_with_auto_detection(self, png_bytes):
        img = Image.from_bytes(png_bytes)

        assert isinstance(img, Image)
        assert img.media_type == "image/png"

    def test_from_bytes_with_explicit_media_type(self):
        # Use arbitrary bytes with explicit media type
        data = b"some image data"
        img = Image.from_bytes(data, media_type="image/png")

        assert img.media_type == "image/png"

    def test_from_bytes_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Could not detect image format"):
            Image.from_bytes(b"unknown data")

    def test_to_pydantic_ai_returns_content(self, png_bytes, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(png_bytes)

        img = Image.from_path(img_path)
        content = img.to_pydantic_ai()

        # from_path returns BinaryContent (via BinaryImage)
        assert hasattr(content, "is_image")
        assert content.is_image

    def test_to_pydantic_ai_url_returns_image_url(self):
        url = "https://example.com/image.jpg"
        img = Image.from_url(url)
        content = img.to_pydantic_ai()

        assert isinstance(content, ImageUrl)
        assert content.url == url

    def test_repr_with_identifier(self, png_bytes, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(png_bytes)
        img = Image.from_path(img_path)

        assert "test.png" in repr(img)

    def test_repr_with_media_type(self, png_bytes):
        img = Image.from_bytes(png_bytes)

        assert "image/png" in repr(img)

    def test_image_is_media_type(self, png_bytes):
        img = Image.from_bytes(png_bytes)
        assert is_media_type(img)

    def test_image_protocol_compliance(self, png_bytes):
        img = Image.from_bytes(png_bytes)
        # Check protocol compliance
        assert isinstance(img, MediaType)
        assert callable(getattr(img, "to_pydantic_ai", None))


# ---------------------------------------------------------------------------
# Audio tests
# ---------------------------------------------------------------------------


class TestAudio:
    """Tests for the Audio class."""

    def test_from_path_creates_audio(self, mp3_bytes, tmp_path):
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(mp3_bytes)

        audio = Audio.from_path(audio_path)

        assert isinstance(audio, Audio)
        assert audio.identifier == str(audio_path)

    def test_from_path_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            Audio.from_path("/nonexistent/path.mp3")

    def test_from_url_creates_audio(self):
        url = "https://example.com/audio.mp3"
        audio = Audio.from_url(url)

        assert isinstance(audio, Audio)
        assert audio.identifier == url

    def test_from_url_with_force_download(self):
        url = "https://example.com/audio.mp3"
        audio = Audio.from_url(url, force_download=True)

        content = audio.to_pydantic_ai()
        assert isinstance(content, AudioUrl)
        assert content.force_download is True

    def test_from_bytes_with_auto_detection(self, mp3_bytes):
        audio = Audio.from_bytes(mp3_bytes)

        assert isinstance(audio, Audio)
        assert audio.media_type == "audio/mpeg"

    def test_from_bytes_with_explicit_media_type(self):
        data = b"some audio data"
        audio = Audio.from_bytes(data, media_type="audio/mpeg")

        assert audio.media_type == "audio/mpeg"

    def test_from_bytes_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Could not detect audio format"):
            Audio.from_bytes(b"unknown data")

    def test_to_pydantic_ai_url_returns_audio_url(self):
        url = "https://example.com/audio.mp3"
        audio = Audio.from_url(url)
        content = audio.to_pydantic_ai()

        assert isinstance(content, AudioUrl)
        assert content.url == url

    def test_audio_is_media_type(self, mp3_bytes):
        audio = Audio.from_bytes(mp3_bytes)
        assert is_media_type(audio)


# ---------------------------------------------------------------------------
# Document tests
# ---------------------------------------------------------------------------


class TestDocument:
    """Tests for the Document class."""

    def test_from_path_creates_document(self, pdf_bytes, tmp_path):
        doc_path = tmp_path / "test.pdf"
        doc_path.write_bytes(pdf_bytes)

        doc = Document.from_path(doc_path)

        assert isinstance(doc, Document)
        assert doc.identifier == str(doc_path)

    def test_from_path_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Document file not found"):
            Document.from_path("/nonexistent/path.pdf")

    def test_from_url_creates_document(self):
        url = "https://example.com/doc.pdf"
        doc = Document.from_url(url)

        assert isinstance(doc, Document)
        assert doc.identifier == url

    def test_from_url_with_force_download(self):
        url = "https://example.com/doc.pdf"
        doc = Document.from_url(url, force_download=True)

        content = doc.to_pydantic_ai()
        assert isinstance(content, DocumentUrl)
        assert content.force_download is True

    def test_from_bytes_with_auto_detection(self, pdf_bytes):
        doc = Document.from_bytes(pdf_bytes)

        assert isinstance(doc, Document)
        assert doc.media_type == "application/pdf"

    def test_from_bytes_with_explicit_media_type(self):
        data = b"some document data"
        doc = Document.from_bytes(data, media_type="application/pdf")

        assert doc.media_type == "application/pdf"

    def test_from_bytes_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Could not detect document format"):
            Document.from_bytes(b"unknown data")

    def test_to_pydantic_ai_url_returns_document_url(self):
        url = "https://example.com/doc.pdf"
        doc = Document.from_url(url)
        content = doc.to_pydantic_ai()

        assert isinstance(content, DocumentUrl)
        assert content.url == url

    def test_document_is_media_type(self, pdf_bytes):
        doc = Document.from_bytes(pdf_bytes)
        assert is_media_type(doc)


# ---------------------------------------------------------------------------
# Video tests
# ---------------------------------------------------------------------------


class TestVideo:
    """Tests for the Video class."""

    def test_from_path_creates_video(self, mp4_bytes, tmp_path):
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(mp4_bytes)

        video = Video.from_path(video_path)

        assert isinstance(video, Video)
        assert video.identifier == str(video_path)

    def test_from_path_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            Video.from_path("/nonexistent/path.mp4")

    def test_from_url_creates_video(self):
        url = "https://example.com/video.mp4"
        video = Video.from_url(url)

        assert isinstance(video, Video)
        assert video.identifier == url

    def test_from_url_with_force_download(self):
        url = "https://example.com/video.mp4"
        video = Video.from_url(url, force_download=True)

        content = video.to_pydantic_ai()
        assert isinstance(content, VideoUrl)
        assert content.force_download is True

    def test_from_bytes_with_auto_detection(self, mp4_bytes):
        video = Video.from_bytes(mp4_bytes)

        assert isinstance(video, Video)
        assert video.media_type == "video/mp4"

    def test_from_bytes_with_explicit_media_type(self):
        data = b"some video data"
        video = Video.from_bytes(data, media_type="video/mp4")

        assert video.media_type == "video/mp4"

    def test_from_bytes_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Could not detect video format"):
            Video.from_bytes(b"unknown data")

    def test_to_pydantic_ai_url_returns_video_url(self):
        url = "https://example.com/video.mp4"
        video = Video.from_url(url)
        content = video.to_pydantic_ai()

        assert isinstance(content, VideoUrl)
        assert content.url == url

    def test_video_is_media_type(self, mp4_bytes):
        video = Video.from_bytes(mp4_bytes)
        assert is_media_type(video)


# ---------------------------------------------------------------------------
# is_media_type tests
# ---------------------------------------------------------------------------


class TestIsMediaType:
    """Tests for the is_media_type helper function."""

    def test_image_is_media_type(self):
        img = Image.from_url("https://example.com/img.png")
        assert is_media_type(img) is True

    def test_audio_is_media_type(self):
        audio = Audio.from_url("https://example.com/audio.mp3")
        assert is_media_type(audio) is True

    def test_document_is_media_type(self):
        doc = Document.from_url("https://example.com/doc.pdf")
        assert is_media_type(doc) is True

    def test_video_is_media_type(self):
        video = Video.from_url("https://example.com/video.mp4")
        assert is_media_type(video) is True

    def test_string_is_not_media_type(self):
        assert is_media_type("hello") is False

    def test_int_is_not_media_type(self):
        assert is_media_type(42) is False

    def test_none_is_not_media_type(self):
        assert is_media_type(None) is False

    def test_dict_is_not_media_type(self):
        assert is_media_type({"key": "value"}) is False


# ---------------------------------------------------------------------------
# Integration tests with spell module
# ---------------------------------------------------------------------------


class TestMediaIntegration:
    """Tests for media types integration with the spell module."""

    def test_build_user_prompt_with_image(self, png_bytes):
        """Test that _build_user_prompt handles Image arguments."""
        from spellcrafting.spell import _build_user_prompt

        img = Image.from_bytes(png_bytes)

        def describe(image: Image) -> str:
            """Describe the image."""
            ...

        result = _build_user_prompt(describe, (img,), {})

        # Should return a list when media is present
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == "image:"
        # Second element should be the pydantic_ai content
        assert hasattr(result[1], "is_image")

    def test_build_user_prompt_with_mixed_args(self, png_bytes):
        """Test that _build_user_prompt handles mixed text and media."""
        from spellcrafting.spell import _build_user_prompt

        img = Image.from_bytes(png_bytes)

        def analyze(question: str, image: Image) -> str:
            """Analyze the image."""
            ...

        result = _build_user_prompt(analyze, ("What is this?", img), {})

        assert isinstance(result, list)
        # Should have: question repr, image label, image content
        assert "question:" in result[0]
        assert "What is this?" in result[0]
        assert "image:" in result[1]

    def test_build_user_prompt_text_only_returns_string(self):
        """Test that text-only args still return a string."""
        from spellcrafting.spell import _build_user_prompt

        def greet(name: str) -> str:
            """Greet."""
            ...

        result = _build_user_prompt(greet, ("Alice",), {})

        assert isinstance(result, str)
        assert "name:" in result
        assert "Alice" in result

    def test_rebuild_user_prompt_with_image(self, png_bytes):
        """Test _rebuild_user_prompt_from_args with media."""
        from spellcrafting.spell import _rebuild_user_prompt_from_args

        img = Image.from_bytes(png_bytes)
        input_args = {"image": img, "question": "What is this?"}

        result = _rebuild_user_prompt_from_args(input_args)

        assert isinstance(result, list)

    def test_rebuild_user_prompt_text_only(self):
        """Test _rebuild_user_prompt_from_args with text only."""
        from spellcrafting.spell import _rebuild_user_prompt_from_args

        input_args = {"name": "Alice", "age": 30}

        result = _rebuild_user_prompt_from_args(input_args)

        assert isinstance(result, str)
        assert "name:" in result
        assert "age:" in result


# ---------------------------------------------------------------------------
# Top-level import tests
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    """Test that media types are properly exported from the package."""

    def test_image_importable(self):
        from spellcrafting import Image
        assert Image is not None

    def test_audio_importable(self):
        from spellcrafting import Audio
        assert Audio is not None

    def test_document_importable(self):
        from spellcrafting import Document
        assert Document is not None

    def test_video_importable(self):
        from spellcrafting import Video
        assert Video is not None

    def test_media_type_importable(self):
        from spellcrafting import MediaType
        assert MediaType is not None

    def test_is_media_type_importable(self):
        from spellcrafting import is_media_type
        assert is_media_type is not None

    def test_media_submodule_importable(self):
        from spellcrafting.media import Image, Audio, Document, Video
        assert all([Image, Audio, Document, Video])
