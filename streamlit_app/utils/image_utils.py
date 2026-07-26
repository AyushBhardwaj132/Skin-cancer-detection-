"""Image loading, validation, and lesion center crop processing."""
from __future__ import annotations
import io
import cv2
import numpy as np
from PIL import Image
from streamlit_app.utils.logger import get_app_logger

logger = get_app_logger("ImageUtils")


def validate_image_file(uploaded_file) -> tuple[bool, str, Image.Image | None]:
    """Validates uploaded image file format, size, and integrity."""
    if uploaded_file is None:
        return False, "No file uploaded.", None

    try:
        # Check size limit (max 20MB)
        if uploaded_file.size > 20 * 1024 * 1024:
            return False, "File size exceeds 20MB limit.", None

        image_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        if w < 50 or h < 50:
            return False, f"Image resolution ({w}x{h}) is too small. Minimum 50x50 required.", None

        return True, "Valid image.", image
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return False, f"Invalid or corrupted image file: {str(e)}", None


def crop_lesion_centered(img_np: np.ndarray, margin: float = 0.20) -> np.ndarray:
    """Detects lesion ROI using OpenCV thresholding and crops square region around lesion with margin.
    
    Reused from src/data/dataset.py to preserve exact preprocessing logic.
    """
    h, w, _ = img_np.shape
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_np

    c_max = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c_max)
    if area < (h * w * 0.01) or area > (h * w * 0.95):
        return img_np

    x, y, bw, bh = cv2.boundingRect(c_max)
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * (1.0 + margin)

    x1 = max(0, int(cx - side / 2.0))
    y1 = max(0, int(cy - side / 2.0))
    x2 = min(w, int(cx + side / 2.0))
    y2 = min(h, int(cy + side / 2.0))

    cropped = img_np[y1:y2, x1:x2]
    if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return img_np
    return cropped
