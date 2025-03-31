"""
MiniVLA installation:

NOTE(YL): create a separate conda environment for minivla install, as it is forked from
OpenVLA and the dependencies might conflict if installed in the same environment as OpenVLA.

https://github.com/Stanford-ILIAD/openvla-mini


# 1. Set up environment variables, they are not used in the script, to avoid error, set them to None:
export HF_TOKEN=None
export PRISMATIC_DATA_ROOT=None

# 2. Download Checkpoitns from huggingface:

cd auto_eval/auto_eval/policy_server/

git lfs install
# download bridge minivla checkpoint to current directory and rename it to minivla
git clone https://huggingface.co/Stanford-ILIAD/minivla-vq-bridge-prismatic

# Download vq checkpoint to current directory and rename it to vq, this should be in the current running directory
# which is in auto_eval/auto_eval/policy_server/
git clone https://huggingface.co/Stanford-ILIAD/pretrain_vq vq

# 3. install dependencies:
pip install uvicorn fastapi json-numpy draccus

# install vqvae:
git clone https://github.com/jayLEE0301/vq_bet_official
cd vq_bet_official
pip install -r requirements.txt
pip install -e .

# 4. run the server:
# NOTE: update your checkpoint path accordingly
python minivla_server.py \
    --hf_token HF_TOKEN \
    --pretrained_checkpoint /home/youliang/rail/prism-qwen25-dinosiglip-224px+0_5b+mx-bridge+n0+b8+x7/checkpoints/step-362600-epoch-604-loss=0.1017.pt

# 5. run server with different checkpoint and unnorm_key:
python minivla_server.py \
    --hf_token HF_TOKEN \
    --unnorm_key expert_demos \
    --pretrained_checkpoint /home/youliang/rail/minivla-vq-bridge-prismatic/checkpoints/step-362500-epoch-21-loss\=0.2259.pt


NOTE(YL) This script is adapted from:
  https://github.com/Stanford-ILIAD/openvla-mini/blob/main/experiments/robot/simpler/run_simpler_eval.py
"""

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from collections import deque

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_action, get_model, set_seed_everywhere


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Server Configuration
    #################################################################################################################
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8004                                                    # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "prismatic"                    # Model family
    hf_token: str = Path(".hf_token")                # Model family
    # vq_checkpoint: Union[str, Path] = ""               # VQ-VAE checkpoint path # THIS IS HARD CODED IN THE SCRIPT
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #NOTE: >1 giving me error ;( need further debug https://github.com/Stanford-ILIAD/openvla-mini/tree/main?tab=readme-ov-file#multi-image
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)
    seed: int = 7                                    # Random Seed (for reproducibility)

    unnorm_key: str = "bridge_dataset"                # Action un-normalization key


class MiniVLAServer:
    def __init__(self, cfg: GenerateConfig):
        assert (
            cfg.pretrained_checkpoint is not None
        ), "cfg.pretrained_checkpoint must not be None!"
        if "image_aug" in cfg.pretrained_checkpoint:
            assert (
                cfg.center_crop
            ), "Expecting `center_crop==True` because model was trained with image augmentations!"
        assert not (
            cfg.load_in_8bit and cfg.load_in_4bit
        ), "Cannot use both 8-bit and 4-bit quantization!"

        # Set random seed
        set_seed_everywhere(cfg.seed)

        # [OpenVLA] Set action un-normalization key to default bridge dataset
        # if cfg.model_family == "prismatic":
        #     cfg.unnorm_key = "bridge_dataset"
        # else:
        #     cfg.unnorm_key = "bridge_orig"

        # Load model
        self.model = get_model(cfg)

        # [OpenVLA] Check that the model contains the action un-normalization key
        if cfg.model_family in ["openvla", "prismatic"]:
            # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            if (
                cfg.unnorm_key not in self.model.norm_stats
                and f"{cfg.unnorm_key}_no_noops" in self.model.norm_stats
            ):
                cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
            print(f"Using un-norm key: {cfg.unnorm_key}")
            assert (
                cfg.unnorm_key in self.model.norm_stats
            ), f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # [OpenVLA] Get Hugging Face processor
        self.processor = None
        if cfg.model_family == "openvla":
            self.processor = get_processor(cfg)
        self.cfg = cfg
        self.image_history = deque(maxlen=cfg.obs_history)

    def predict_action(self, payload: dict) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # checks all keys in payload
            assert {"image", "instruction"} <= set(
                payload.keys()
            ), "Missing keys in payload!"

            self.image_history.append(payload["image"])

            # convert deque to list of Image
            images = list(self.image_history)

            observation = {
                "full_image": images,
            }

            # run inference
            action = get_action(
                self.cfg,
                self.model,
                observation,
                payload["instruction"],
                processor=self.processor,
            )

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8004) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@draccus.wrap()
def deploy(cfg: GenerateConfig) -> None:
    server = MiniVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
