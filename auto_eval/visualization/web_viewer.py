"""
host a website locally to view the robot's video feed and status
See the website under auto_eval/visualization/template/index.html
"""
import io
import logging

import requests
from PIL import Image as PILImage

# Set the logging level for urllib3 to WARNING
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Hardcoded hyperparameters
WEB_VIEWER_IP = "0.0.0.0"
WEB_VIEWER_PORT = 8080
IMAGE_TYPES = ["observation", "goal"]


def upload_to_webviewer(
    img, language_instruction=None, robot_id=0, episode=0, timestep=0, session=None
) -> None:
    """Upload the processed image and language instruction to the web server."""
    if img is None:
        return

    # make image compatible
    try:
        assert type(img) == PILImage.Image
    except AssertionError:
        # convert to PIL
        img = PILImage.fromarray(img)
    # assert shape is 256x 256, else reshape
    try:
        assert img.size == (256, 256)
    except AssertionError:
        img = img.resize((256, 256))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    files = {"file": ("image.jpg", buffer.getvalue(), "image/jpeg")}

    # Use provided session or create a new one
    should_close_session = False
    if session is None:
        session = requests.Session()
        should_close_session = True

    try:
        # upload image
        url = f"http://{WEB_VIEWER_IP}:{WEB_VIEWER_PORT}/upload/{robot_id}?type=observation"
        response = session.post(url, files=files)
        response.raise_for_status()  # Raise an error for bad responses

        # Only update status if at least one status parameter is not None
        if (
            language_instruction is not None
            or episode is not None
            or timestep is not None
        ):
            status_url = (
                f"http://{WEB_VIEWER_IP}:{WEB_VIEWER_PORT}/update_status/{robot_id}"
            )
            status_data = {
                "language_instruction": language_instruction
                if language_instruction is not None
                else "N/A",
                "episode": episode if episode is not None else 0,
                "timestep": timestep if timestep is not None else 0,
            }
            status_response = session.post(status_url, json=status_data)
            status_response.raise_for_status()  # Raise an error for bad responses
    except requests.RequestException as e:
        print(f"Error uploading image: {e}")
    finally:
        buffer.close()
        if should_close_session:
            session.close()
