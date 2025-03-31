from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

from auto_eval.success_detector.base_detector import BaseSuccessDetector
from auto_eval.utils.info import print_yellow


class PaligemmaDetector(BaseSuccessDetector):
    def __init__(
        self,
        processor_id="google/paligemma-3b-pt-224",
        model_id="google/paligemma-3b-pt-224",
        device="cuda",
        quantize=False,
        # saving data
        save_data=False,
        save_dir=None,
    ):
        super().__init__(save_data=save_data, save_dir=save_dir)
        self.processor = PaliGemmaProcessor.from_pretrained(processor_id)
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            )
        else:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            ).to(device)
        self.device = device
        self.model.eval()

    def __call_main__(
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
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        inputs = self.processor(image, "<image>" + prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        decoded_output = self.processor.decode(outputs[0], skip_special_tokens=True)
        decoded_output = decoded_output[len(prompt) :].strip("\n")
        print_yellow(
            f"Raw Paligemma output: {decoded_output.strip().lower()}. ((Prompt is : {prompt}))"
        )
        if answer is None:
            return decoded_output
        return decoded_output.strip().lower() == answer
