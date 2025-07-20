import httpx
import cv2
import numpy as np
from typing import Tuple

async def fetch_image(image_url: str, token: str, timeout: float = 30.0) -> Tuple[bytes, np.ndarray]:
    """
    Fetch an image from URL and return both raw bytes and numpy array.

    Args:
        image_url: URL to fetch the image from
        token: Bearer token for authentication
        timeout: Request timeout in seconds

    Returns:
        Tuple of (image_bytes, numpy_image) where numpy_image is dtype=np.uint8

    Raises:
        httpx.HTTPError: If request fails
        cv2.error: If image cannot be decoded
    """
    headers = {
        "Authorization": f"Bearer {token}"
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(image_url, headers=headers)
        response.raise_for_status()

        image_bytes = response.content

        # Convert bytes to numpy array and decode with OpenCV
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Ensure the image is in the correct format
        if image is None:
            raise ValueError("Failed to decode image")

        # OpenCV loads images in BGR format, convert to RGB if needed
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image_bytes, image_rgb
