"""
The advanced policy server provides an example of serving a policy that needs:
    action chunking (with temporal ensembling)
    observation history
You can modify this example to make your policy track other internal states.

The policy server needs to provide an endpoint .reset() that resets these internal states.
The .reset() endpoint will be called by AutoEval at the start of every evaluation trajectory.

-------------------------------------------------------------------------------------------

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
"""

import json_numpy

json_numpy.patch()

import json
import logging
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import draccus
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# === Server Interface ===
class ActionChunkingObsHistoryPolicyServer:
    """
    A simple server for your robot policy; exposes `/act` to predict an action for a given image + instruction.
        => Takes in {"image": np.ndarray, "instruction": str, "proprio": Optional[np.ndarray]}
        => Returns  {"action": np.ndarray}

    Features:
    - Optional observation history: Maintain a history of past observations
    - Optional action chunking: Predict multiple actions at once
    - Optional temporal ensembling: Combine multiple predictions for the same timestep
    """

    def __init__(
        self,
        obs_horizon: int = 1,
        action_pred_horizon: int = 1,
        action_temporal_ensemble: bool = False,
        action_exp_weight: float = 0.0,
    ):
        """
        Initialize the policy server with optional history and action chunking support.

        Args:
            obs_horizon: Number of observations to keep in history (1 means no history)
            pred_horizon: Number of actions to predict at once (1 means no chunking)
            temporal_ensemble: Whether to use temporal ensembling for actions
            exp_weight: Exponential weight for temporal ensembling
        """
        ####################### TODO #######################
        ##                 Load model here                ##
        ####################################################
        self.model = ...

        # History tracking
        self.obs_horizon = max(1, obs_horizon)
        self.observation_history = deque(maxlen=self.obs_horizon)

        # Action chunking
        self.action_pred_horizon = max(1, action_pred_horizon)
        # Temporal ensembling
        self.action_temporal_ensemble = action_temporal_ensemble
        self.action_exp_weight = action_exp_weight
        self.action_history = (
            deque(maxlen=self.action_pred_horizon) if action_temporal_ensemble else None
        )

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        """
        Predict action(s) given an image + proprio + instruction.

        With history and chunking support:
        - If obs_horizon > 1: Uses stacked observations
        - If pred_horizon > 1: Returns multiple actions
        - If temporal_ensemble: Applies temporal ensembling
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

            # Extract observation components
            observation = {
                "image": payload["image"],
                "instruction": payload["instruction"],
            }

            # Add optional proprio if provided
            if "proprio" in payload:
                observation["proprio"] = payload["proprio"]

            # Update observation history
            if self.obs_horizon > 1:
                observation = self._update_observation_history(observation)

            ####################### TODO #######################
            ##              Run model inference               ##
            ####################################################
            # Predict action chunks (pred_horizon actions at once)
            # Replace this with your actual model inference
            action = ...  # self.model(observation)

            # check if actions are chunked
            if len(action.shape) > 1:
                assert action.shape[0] == self.action_pred_horizon

                # Apply temporal ensembling if enabled
                if self.action_temporal_ensemble:
                    action = self._apply_temporal_ensembling(action)

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
                    "{'image': np.ndarray, 'instruction': str, 'proprio': Optional[np.ndarray]}\n"
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

        # Add reset endpoint
        @self.app.post("/reset")
        async def reset_server():
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

    def _update_observation_history(
        self, observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update the observation history and return the stacked observations.

        Args:
            observation: Current observation

        Returns:
            Stacked observations with history
        """
        # If this is the first observation, initialize history with copies
        if len(self.observation_history) == 0:
            self.observation_history.extend([observation] * self.obs_horizon)
        else:
            self.observation_history.append(observation)

        # If using history (obs_horizon > 1), stack and pad
        if self.obs_horizon > 1:
            return stack_history(self.observation_history)
        else:
            return observation

    def _apply_temporal_ensembling(self, action_chunks: np.ndarray) -> np.ndarray:
        """
        Apply temporal ensembling to the predicted action chunks.

        Args:
            action_chunks: Predicted action chunks of shape (pred_horizon, action_dim)

        Returns:
            First action after temporal ensembling
        """
        # Add current prediction to history
        self.action_history.append(action_chunks)
        num_actions = len(self.action_history)

        # Select the predicted action for the current step from history of action chunk predictions
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.action_history
                )
            ]
        )

        # More recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.action_exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()

        # Compute the weighted average across all predictions for this timestep
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)
        assert action.shape == (7,), action.shape

        return action

    def reset(self) -> None:
        """
        Reset the server state (observation history and action history).
        """
        self.observation_history.clear()
        if self.action_pred_horizon > 1:
            self.action_history.clear()


def stack_history(
    history: deque,
):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values.
    """
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    return full_obs


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "0.0.0.0"  # Host IP Address
    port: int = 8000  # Host Port
    pred_horizon: int = 1  # action chunk to predict
    obs_horizon: int = 1  # observation history
    temporal_ensemble: bool = False  # whether to use temporal ensembling
    exp_weight: float = 0.0  # exponential weight for temporal ensembling


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = PolicyServer(
        obs_horizon=cfg.obs_horizon,
        action_pred_horizon=cfg.pred_horizon,
        action_temporal_ensemble=cfg.temporal_ensemble,
        exp_weight=cfg.exp_weight,
    )
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
