import threading
import time

import requests
from PIL import Image

from auto_eval.visualization.local import display_img
from auto_eval.visualization.web_viewer import upload_to_webviewer


def visualize_image(
    method,
    img,
    language_instruction=None,
    robot_id=0,
    episode=0,
    timestep=0,
    wait_time=1,
    session=None,
):
    """
    Visualize the image in different ways
    """
    if method == "display":
        # wait time in miliseconds
        # Image.fromarray(img).save("image.png")
        display_img(img, language_instruction, wait_time=wait_time)
    elif method == "web_viewer":
        upload_to_webviewer(
            img,
            language_instruction=language_instruction,
            robot_id=robot_id,
            episode=episode,
            timestep=timestep,
            session=session,
        )
    elif method == "none":
        # Don't do anything
        pass
    else:
        raise ValueError(f"Unknown visualization method: {method}")


class ImageStreamer:
    """Thread-safe image streaming with proper resource management"""

    def __init__(self, manipulator_interface, visualization_method, robot_id):
        self.manipulator_interface = manipulator_interface
        self.visualization_method = visualization_method
        self.robot_id = robot_id
        self._stop_event = threading.Event()
        self._session = (
            None if visualization_method != "web_viewer" else requests.Session()
        )

    def start(self):
        """Start the image streaming thread"""
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the image streaming thread and cleanup resources"""
        self._stop_event.set()
        if self._session:
            self._session.close()

    def _stream_loop(self):
        """Main streaming loop with proper error handling"""
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                img = self.manipulator_interface.primary_img
                if img is not None:
                    visualize_image(
                        self.visualization_method,
                        img,
                        None,  # No language instruction for continuous view
                        robot_id=self.robot_id,
                        episode=None,
                        timestep=None,
                        session=self._session,  # Pass the reusable session
                    )
                consecutive_errors = 0  # Reset error counter on success
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            except Exception as e:
                consecutive_errors += 1
                print(f"Error in visualization thread: {e}")
                if consecutive_errors > 10:
                    print(
                        "Too many consecutive errors in visualization thread, stopping"
                    )
                    break
                time.sleep(min(consecutive_errors, 5))  # Exponential backoff up to 5s


def stream_images(manipulator_interface, visualization_method, robot_id):
    """Create and return an ImageStreamer instance"""
    streamer = ImageStreamer(manipulator_interface, visualization_method, robot_id)
    streamer.start()
    return streamer  # Return the streamer so it can be stopped later
