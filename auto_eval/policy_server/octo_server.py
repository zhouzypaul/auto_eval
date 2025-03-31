import json_numpy

json_numpy.patch()

import json
import logging
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import draccus
import jax
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import stack_and_pad


# === Server Interface ===
class OctoServer:
    """
    Server implementation for the Octo policy. This implementation follows template_advanced.py.
    Exposes `/act` to predict an action for a given image + instruction.
        => Takes in {"image_primary": np.ndarray, "instruction": str, "proprio": np.ndarray}
        => Returns  {"action": np.ndarray}

    Features:
    - Uses observation history (obs_horizon=2)
    - Uses action chunking (pred_horizon=4)
    - Optional temporal ensembling for actions
    """

    def __init__(
        self,
        obs_horizon: int = 2,
        action_pred_horizon: int = 4,
        action_temporal_ensemble: bool = True,
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
        ####################### IMPLEMENTED ####################
        ##                 Load Octo model                    ##
        ########################################################
        # Load the Octo model
        self.agent = OctoModel.load_pretrained(
            "hf://rail-berkeley/octo-small"
        )  # don't use 1.5
        self.task = None  # Created later when we get the first instruction

        # History tracking
        self.obs_horizon = max(1, obs_horizon)
        self.observation_history = deque(maxlen=self.obs_horizon)
        self.num_obs = 0  # Track number of observations for stack_and_pad

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
            obs_dict = {
                "image_primary": payload["image"],
                # "language_instruction": payload["instruction"],
            }

            # Add optional proprio if provided
            if "proprio" in payload:
                obs_dict["proprio"] = payload["proprio"]

            # Update observation history
            if self.obs_horizon > 1:
                obs_dict = self._update_observation_history(obs_dict)

            ####################### IMPLEMENTED ####################
            ##              Run Octo model inference             ##
            ########################################################
            # Create task if this is the first call
            if self.task is None:
                # Assumes that each Octo model doesn't receive different tasks
                self.task = self.agent.create_tasks(texts=[payload["instruction"]])
                # Alternative: self.task = self.agent.create_tasks(goals={"image_primary": img})  # for goal-conditioned

            # Sample actions
            action = self.agent.sample_actions(
                jax.tree_map(lambda x: x[None], obs_dict),
                self.task,
                unnormalization_statistics=self.agent.dataset_statistics[
                    "bridge_dataset"
                ]["action"],
                rng=jax.random.PRNGKey(0),
            )
            # Model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            action = action[0]  # Note that actions here are chunked

            # Convert JAX array to NumPy array for JSON serialization
            action = np.array(action)

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
            self.num_obs = 1
        else:
            self.observation_history.append(observation)
            self.num_obs += 1

        # If using history (obs_horizon > 1), stack and pad
        if self.obs_horizon > 1:
            return stack_and_pad(self.observation_history, self.num_obs)
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
        Reset the server state (observation history, action history, and task).
        """
        self.observation_history.clear()
        self.num_obs = 0  # Reset observation counter
        if self.action_pred_horizon > 1:
            self.action_history.clear()
        # Also reset the task when resetting the server
        self.task = None


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "0.0.0.0"  # Host IP Address
    port: int = 8000  # Host Port
    pred_horizon: int = 4  # action chunk to predict for Octo
    obs_horizon: int = 2  # observation history for Octo
    temporal_ensemble: bool = True  # whether to use temporal ensembling
    exp_weight: float = 0.0  # exponential weight for temporal ensembling


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OctoServer(
        obs_horizon=cfg.obs_horizon,
        action_pred_horizon=cfg.pred_horizon,
        action_temporal_ensemble=cfg.temporal_ensemble,
        action_exp_weight=cfg.exp_weight,
    )
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
