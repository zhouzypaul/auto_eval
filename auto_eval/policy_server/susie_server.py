import json
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict

import draccus
import json_numpy
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

json_numpy.patch()


class GoalImageGeneratorServer:
    """SuSIE: https://arxiv.org/abs/2310.10639"""

    def __init__(self, config):
        # lazy import
        from susie.model import create_sample_fn

        self.diffusion_sample_func = create_sample_fn(
            config["diffusion_checkpoint"],
            config["diffusion_wandb"],
            config["diffusion_num_steps"],
            config["prompt_w"],
            config["context_w"],
            0.0,
            config["diffusion_pretrained_path"],
        )
        self.image_size = config["image_size"]

    def generate_subgoal(self, payload: Dict[str, Any] = Body(...)):
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            image_obs = np.array(payload["image"])
            prompt = payload["text"]

            assert image_obs.shape == (
                self.image_size,
                self.image_size,
                3,
            ), f"Bad input image shape {image_obs.shape}"

            result = self.diffusion_sample_func(image_obs, prompt)

            if double_encode:
                return JSONResponse(json_numpy.dumps(result))
            else:
                return JSONResponse(result.tolist())
        except Exception as e:  # Be more specific about the error
            logging.error(traceback.format_exc())
            logging.warning(
                f"Error processing request: {str(e)}\n"
                "Make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'text': str}"
            )
            return JSONResponse(
                {"error": f"Failed to process request: {str(e)}"}, status_code=400
            )

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/susie")(self.generate_subgoal)
        uvicorn.run(self.app, host=host, port=port)


class SOARPolicyServer:
    """
    A server for the SOARPolicy; exposes `/act` to predict an action for a given image + instruction.
        => Takes in {"image_primary": np.ndarray, "instruction": str, "proprio": Optional[np.ndarray]}
        => Returns  {"action": np.ndarray}
    """

    def __init__(self, config):
        from auto_eval.robot.policy import SOARPolicy

        self.policy = SOARPolicy(config)

    def predict_action(self, payload: Dict[str, Any] = Body(...)) -> JSONResponse:
        """
        Predict a 7-dim action given an image + proprio + instruction
        """
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            if "image" not in payload or "instruction" not in payload:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required fields: image and instruction",
                )
            image, instruction = payload["image"], payload["instruction"]
            proprio = payload.get("proprio", None)  # proprio is optional

            action = self.policy(
                {"image_primary": image, "proprio": proprio}, instruction
            )

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except HTTPException:
            raise
        except Exception as e:
            logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=(
                    "Error processing request."
                    "Make sure your request complies with the expected format:\n"
                    "{'image': np.ndarray, 'instruction': str}\n"
                ),
            )

    def reset(self):
        self.policy.reset()

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # Add health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}

        self.app.post("/act")(self.predict_action)

        # Add reset endpoint
        @self.app.post("/reset")
        async def reset():
            self.reset()
            return {"status": "reset successful"}

        # Configure server with increased timeout and request size limits
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            timeout_keep_alive=120,
            limit_concurrency=2,
        )
        server = uvicorn.Server(config)
        server.run()


@dataclass
class DeployConfig:
    # fmt: off
    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8001                                                    # Host Port
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    from scripts.configs.eval_config import soar_policy_kwargs as default_configs

    server = SOARPolicyServer(default_configs)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
