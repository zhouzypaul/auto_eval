"""
Provide a lightweight server/client implementation template for deploying your generalist policy over a
REST API. This template implements *just* the server.
See auto_eval/robot/policy_clients.py:OpenWebClient for an example of how the client is handled.

Dependencies:
pip install uvicorn fastapi json-numpy draccus

Usage:
python policy_server.py --port 8000

To make your server accessible on the open web, you can use ngrok or bore.pub
With ngrok:
  ngrok http 8000
With bore.pub:
  bore local 8000 --to bore.pub

Note that if you aren't able to resolve bore.pub's DNS (test this with `ping bore.pub`), you can use their actual IP: 159.223.171.199

Adapted from: https://github.com/openvla/openvla/blob/main/vla-scripts/deploy.py
"""

import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import draccus
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# === Server Interface ===
class PolicyServer:
    """
    A simple server for your robot policy; exposes `/act` to predict an action for a given image + instruction.
        => Takes in {"image": np.ndarray, "instruction": str, "proprio": Optional[np.ndarray]}
        => Returns  {"action": np.ndarray}
    """

    def __init__(
        self,
    ) -> Path:
        """
        Load your policy model
        """
        ####################### TODO #######################
        ##                 Load model here                ##
        ####################################################
        self.model = ...
        pass

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
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

            ####################### TODO #######################
            ##              Run model inference               ##
            ####################################################
            action = ...  # e.g. self.model(image, instruction)

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
    # Server Configuration
    host: str = "0.0.0.0"  # Host IP Address
    port: int = 8000  # Host Port


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = PolicyServer()
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
