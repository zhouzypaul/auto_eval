import os
from typing import Optional, Union

import numpy as np
from PIL import Image


class BaseSuccessDetector:
    def __init__(
        self,
        save_data: bool = False,
        save_dir: Optional[str] = None,
    ):
        """
        Optionally save the input/output data for the success detector.
        """
        self.save_data = save_data
        self.save_counter = 0
        self.save_dir = save_dir
        if save_dir is not None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(
        self,
        prompt: str,
        image: Union[np.ndarray, Image.Image],
        answer: Optional[str] = None,
    ) -> Union[str, bool]:
        """
        Args:
            prompt (str): The prompt for the model.
            image (Union[np.ndarray, Image.Image]): The image for the model.
            answer (str): The answer for the model. Default is None.

        Returns:
            Union[str, bool]: The output of the model

        This function is a wrapper around the main detection function: self.__call_main__
        """
        result = self.__call_main__(prompt, image, answer)
        self.__call_post__(image, result)
        return result

    def __call_main__(
        self,
        prompt: str,
        image: Union[np.ndarray, Image.Image],
        answer: Optional[str] = None,
    ) -> bool:
        """main meat of the call function. Returns the image with the success result."""
        raise NotImplementedError

    def __call_post__(
        self,
        image: Union[np.ndarray, Image.Image],
        success_result: bool,
    ) -> Union[str, bool]:
        """post processing of the call function: save data, etc."""
        if self.save_data:
            assert self.save_dir is not None
            self.save_image_and_result(image, success_result)

    def save_image_and_result(self, image, result):
        """
        Save the image alongside the result. Both are indexed by `step`.
        """
        img_path = os.path.join(self.save_dir, "{}_img.png".format(self.save_counter))
        result_path = os.path.join(
            self.save_dir, "{}_success.txt".format(self.save_counter)
        )
        self.save_counter += 1
        # save to disk
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(img_path)
        with open(result_path, "w") as f:
            f.write(str(result))
