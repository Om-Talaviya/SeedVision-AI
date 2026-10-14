"""Input validation for live web inference and user uploads."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import io
from PIL import Image

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION = 32
MAX_DIMENSION = 4096


@dataclass
class UploadValidationResult:
    is_valid: bool
    image: Optional[Image.Image] = None
    error_message: Optional[str] = None
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0


class InferenceInputValidator:
    """Validates user-uploaded image files before model processing."""

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        min_dim: int = MIN_DIMENSION,
        max_dim: int = MAX_DIMENSION,
    ):
        self.max_file_size = max_file_size
        self.min_dim = min_dim
        self.max_dim = max_dim

    def validate_upload(
        self,
        file_bytes: Union[bytes, io.BytesIO],
        filename: str = "upload.jpg",
    ) -> UploadValidationResult:
        """Validate raw uploaded bytes for safety, integrity, and dimensions."""
        if isinstance(file_bytes, io.BytesIO):
            data = file_bytes.getvalue()
        else:
            data = file_bytes

        size = len(data)

        if size == 0:
            return UploadValidationResult(
                is_valid=False,
                error_message="Uploaded file is empty (0 bytes).",
                file_size_bytes=0,
            )

        if size > self.max_file_size:
            return UploadValidationResult(
                is_valid=False,
                error_message=f"File size ({size / 1024 / 1024:.1f} MB) exceeds maximum allowed ({self.max_file_size / 1024 / 1024:.1f} MB).",
                file_size_bytes=size,
            )

        # Decompress and verify image
        try:
            stream = io.BytesIO(data)
            with Image.open(stream) as img:
                img.verify()

            # Re-read image data
            stream.seek(0)
            img = Image.open(stream)
            img.load()

            w, h = img.size

            if w < self.min_dim or h < self.min_dim:
                return UploadValidationResult(
                    is_valid=False,
                    error_message=f"Image resolution {w}x{h} is too low for seed grading (minimum {self.min_dim}x{self.min_dim}).",
                    width=w,
                    height=h,
                    file_size_bytes=size,
                )

            if w > self.max_dim or h > self.max_dim:
                return UploadValidationResult(
                    is_valid=False,
                    error_message=f"Image resolution {w}x{h} exceeds maximum supported ({self.max_dim}x{self.max_dim}).",
                    width=w,
                    height=h,
                    file_size_bytes=size,
                )

            # Ensure clean RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")

            return UploadValidationResult(
                is_valid=True,
                image=img,
                width=w,
                height=h,
                file_size_bytes=size,
            )

        except Exception as e:
            return UploadValidationResult(
                is_valid=False,
                error_message=f"Corrupted or invalid image file: {str(e)}",
                file_size_bytes=size,
            )
