import io
import os
import time

import numpy as np
from PIL import Image
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from auto_eval.utils.info import print_red


def img_to_bytes(img):
    """
    Convert an image to bytes and send it to Slack.
    """
    if type(img) == bytes:
        return img
    elif type(img) == np.ndarray:
        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(img.astype("uint8"))

        # Save the image to a bytes buffer
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")  # You can change the format if needed
        img_byte_arr.seek(0)  # Move to the beginning of the BytesIO buffer

        return img_byte_arr
    else:
        raise ValueError(f"Image type {type(img)} not supported")


class SlackMessenger:
    def __init__(self):
        # get slack token from environment variable
        try:
            self.slack_token = os.environ["SLACK_BOT_TOKEN"]
            self.channel_id = os.environ["SLACK_CHANNEL_ID"]
        except KeyError:
            raise KeyError(
                "Please set the environment variable SLACK_BOT_TOKEN and SLACK_CHANNEL_ID"
            )
        self.client = WebClient(token=self.slack_token)

    def send(self, message, image=None, get_response=True):
        """
        Send a message to the slack channel, with an optional image
        """
        try:
            if image is not None:
                response = self.client.files_upload_v2(
                    channel=self.channel_id,
                    file=img_to_bytes(image),
                    initial_comment=message,
                )
                time.sleep(1)  # delay so file is uploaded before sending message
            else:
                response = self.client.chat_postMessage(
                    channel=self.channel_id, text=message
                )

            print_red(f"Message sent to slack: {message}")
            if get_response:
                response = self.client.chat_postMessage(
                    channel=self.channel_id,
                    text="Please reply in this thread with 'y' to continue robot operation.",
                )
                self.most_recent_text_ts = response["ts"]  # timestamp

        except SlackApiError as e:
            assert e.response["ok"] is False
            assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
            print(f"Got an error: {e.response['error']}")

    def check_for_response(self):
        """
        Check for a response in the slack channel.
        Returns the message if 'y' is found, otherwise None.
        """
        try:
            response = self.client.conversations_replies(
                channel=self.channel_id, ts=self.most_recent_text_ts
            )
            messages = response["messages"]
            if len(messages) > 1 and messages[-1]["text"].lower() in ("y", "Y"):
                return True
        except SlackApiError as e:
            print(f"Error fetching messages: {e.response['error']}")
        return False


class DummyBot:
    def __init__(self):
        pass

    def send(self, message, image=None, get_response=False):
        # image, get_response are ignored, just keeping API consistent
        print(f"Message sent to terminal: {message}")
        if image is not None:
            print(
                "Unable to display image in terminal. Please use SlackMessenger instead."
            )

    def check_for_response(self):
        """
        Always return False as this is a dummy bot.
        """
        return False


if __name__ == "__main__":
    # testing the slack messenger
    img_path = "goal_images/open the drawer.png"
    with open(img_path, "rb") as f:
        image = f.read()

    bot = SlackMessenger()

    # sending messages
    # bot.send("Hello from the slack bot!")
    bot.send("Testing the image upload", image=image)

    # checking for messages
    for i in range(100):
        response = bot.check_for_response()
        print(response)
