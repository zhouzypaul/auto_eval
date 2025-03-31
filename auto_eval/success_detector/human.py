from typing import Optional, Union

import numpy as np
from PIL import Image

from auto_eval.success_detector.base_detector import BaseSuccessDetector


class HumanHandDetector(BaseSuccessDetector):
    """success detection by hand"""

    def __init__(
        self,
        # saving data
        save_data=False,
        save_dir=None,
        *args,
        **kwargs
    ):
        super().__init__(save_data=save_data, save_dir=save_dir)
        pass

    def __call_main__(
        self,
        prompt: str,
        image: Union[np.ndarray, Image.Image],
        answer: Optional[str] = None,
    ):
        # TODO: ask for human input from command line and parse it
        return False
