"""
Some policies are expensive to run locally, so we run them on a server machine,
and use these clients to communicate with them.
When the clients are called, make an HTTP request to an action server.

The default ports are:
8000 -- OpenVLA
8001 -- OpenPiZero
8002 -- SuSIE
8003 -- finetuned OpenVLA open drawer policy
8004 -- miniVLA server
"""

import getpass
import io
import os
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import requests
from PIL import Image

from auto_eval.utils.info import print_yellow


class OpenWebClient:
    """
    A Client that listens to a port on the open web, and makes HTTP requests to it.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._session = requests.Session()

        # we need to able to serialize numpy arrays to json to send over the network
        import json_numpy

        json_numpy.patch()

    def __call__(
        self, obs_dict: Dict[str, Any], language_instruction: Optional[str] = None
    ):
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3, obs_dict[
            "image_primary"
        ].shape

        # Use the session for connection reuse
        optional_kwargs = (
            {
                "proprio": obs_dict["proprio"],
            }
            if "proprio" in obs_dict
            else {}
        )
        action = self._session.post(
            f"http://{self.host}:{self.port}/act",
            json={
                "image": obs_dict["image_primary"],
                "instruction": language_instruction,
                **optional_kwargs,
            },
        ).json()

        # the original action is not modifiable, cannot clip boundaries after the fact for example
        if type(action) not in (np.ndarray, list):
            raise RuntimeError(
                "Policy server returned invalid action. It must return a numpy array or a list. Received: "
                + str(action)
            )
        return action.copy()

    def reset(
        self,
    ):
        # Post request to the policy server to reset internal states
        response = self._session.post(f"http://{self.host}:{self.port}/reset")
        # If we get a response, check if it was successful
        if response.status_code == 404:
            # if the policy server doesn't have a /reset endpoint, ignore
            print_yellow("Policy server does not have a /reset endpoint")
            pass
        elif response.status_code != 200:
            raise RuntimeError(
                f"Failed to reset policy server: {response.status_code} {response.text}"
            )

    def close(self):
        """Explicitly close connections"""
        if hasattr(self, "_session"):
            self._session.close()

    def __del__(self):
        self.close()


class PortForwardingClient:
    def __init__(self, host, port, ssh_port=None):
        self.host = host
        self.port = port  # port the policy is running
        self.ssh_port = ssh_port  # the port used to access server
        self._session = requests.Session()  # Create a session for connection reuse

        if self.host != "localhost":
            self._forward_port(port=port, ssh_port=ssh_port)

    def _forward_port(self, port, ssh_port=None):
        # run ssh [-p 2222] -L 8000:localhost:8000 ssh USER@<SERVER_IP>
        user = getpass.getuser()
        optional_ssh_port = "" if ssh_port is None else f"-p {ssh_port}"
        subprocess.run(
            f"ssh {optional_ssh_port} -L {port}:localhost:{port} -N -f -C -o ExitOnForwardFailure=yes {user}@{self.host}",
            shell=True,
        )
        # Check if the port is open
        result = subprocess.run(
            f"lsof -i :{port}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            print(f"Port {port} is open.")
        else:
            raise RuntimeError(f"Port {port} is not open.")

    def __call__(
        self, obs_dict: Dict[str, Any], language_instruction: Optional[str] = None
    ):
        raise NotImplementedError

    def reset(self):
        pass

    def close(self):
        """Explicitly close connections"""
        if hasattr(self, "_session"):
            self._session.close()
        if hasattr(self, "port") and hasattr(self, "host") and self.host != "localhost":
            try:
                # Find and kill the SSH process forwarding this port
                subprocess.run(
                    f"lsof -ti :{self.port} | xargs kill -9",
                    shell=True,
                    stderr=subprocess.PIPE,
                )
                print(f"Closed port forwarding on port {self.port}")
            except Exception as e:
                print(f"Error closing port {self.port}: {str(e)}")

    def __del__(self):
        self.close()


class OpenVLAClient(PortForwardingClient):
    def __init__(self, host="localhost", port=8000, ssh_port=2222):
        # lazy imports
        import json_numpy

        json_numpy.patch()
        super().__init__(host, port, ssh_port)

    def __call__(
        self, obs_dict: Dict[str, Any], language_instruction: Optional[str] = None
    ):
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3, obs_dict[
            "image_primary"
        ].shape

        # the server will handle adding VLA prompt to language_instruction
        # Use the session for connection reuse
        action = self._session.post(
            f"http://localhost:{self.port}/act",
            json={
                "image": obs_dict["image_primary"],
                "instruction": language_instruction,
                "unnorm_key": "bridge_orig",
            },
        ).json()

        # the original action is not modifiable, cannot clip boundaries after the fact for example
        return action.copy()


class MiniVLAClient(OpenVLAClient):
    def __init__(self, host="localhost", port=8004, ssh_port=2222):
        # override default port
        super().__init__(host, port, ssh_port)


class OpenPiZeroClient(PortForwardingClient):
    def __init__(self, host="localhost", port=8001, ssh_port=2222):
        # lazy imports
        import json_numpy

        json_numpy.patch()
        super().__init__(host, port, ssh_port)

    def __call__(
        self, obs_dict: Dict[str, Any], language_instruction: Optional[str] = None
    ):
        assert "image_primary" in obs_dict.keys()
        img = obs_dict["image_primary"]
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3, img.shape
        assert img.shape == (256, 256, 3)

        # get rid of the 0 dimension
        raw_proprio = obs_dict["proprio"]
        gripper = raw_proprio[7]
        # normalize the gripper from (0, 0.39) from manipulator_gym to (0, 1)
        # see widowx.py in manipulator_gym for the (0, 0.39) range
        gripper = gripper / 0.39
        gripper = np.clip(gripper, 0, 1)
        processed_proprio = np.append(raw_proprio[:6], gripper)

        # Use the session for connection reuse
        action = self._session.post(
            f"http://localhost:{self.port}/act",
            json={
                "image": img,
                "instruction": language_instruction,
                "unnorm_key": "bridge_orig",
                "proprio": processed_proprio,
            },
        ).json()

        # the original action is not modifiable, cannot clip boundaries after the fact for example
        r_action = action.copy()
        # normalize the gripper from (-1, 1) to (0, 1)
        r_action[:, -1] = (r_action[:, -1] + 1) / 2
        r_action[:, -1] = np.clip(r_action[:, -1], 0, 1)
        return r_action


class GoalImageGeneratorClient(PortForwardingClient):
    """
    Run SuSIE on a remote server, and this acts as a client to query the server
    and get back the goal image.
    """

    def __init__(self, config, host="localhost", port=8002, ssh_port=2222):
        # lazy imports
        import json_numpy

        json_numpy.patch()
        super().__init__(host, port, ssh_port)
        self.config = config

    def __call__(self, image_obs: np.ndarray, prompt: str):
        assert image_obs.shape == (
            self.config["image_size"],
            self.config["image_size"],
            3,
        ), f"Bad image shape {image_obs.shape}"

        # Use the session for connection reuse
        response = self._session.post(
            f"http://localhost:{self.port}/susie",
            json={
                "image": image_obs.tolist(),  # Convert numpy array to list for JSON serialization
                "text": prompt,
            },
        )

        if response.status_code == 200:
            return np.array(response.json(), dtype=np.uint8)
        else:
            error_msg = response.json() if response.text else {"error": "Unknown error"}
            print("Failed to process image", response.status_code, error_msg)
            return None


class DiffusionPolicyClient:
    def __init__(
        self, host="localhost", port=6000, precomputed_goal_images_dir="goal_images"
    ):
        self.host = host
        self.port = port

        # Find all images under precomputed_goal_images_dir
        # the file name should match the language instruction being commanded
        files = [
            f
            for f in os.listdir(precomputed_goal_images_dir)
            if os.path.isfile(os.path.join(precomputed_goal_images_dir, f))
        ]
        self.lang_to_goal_image = {}
        for file in files:
            lang_instr = file.split(".")[0]
            self.lang_to_goal_image[lang_instr] = os.path.join(
                precomputed_goal_images_dir, file
            )

    def array_to_image_bytes(self, arr):
        img = Image.fromarray(arr.astype("uint8"), "RGB")
        byte_arr = io.BytesIO()
        img.save(byte_arr, format="PNG")
        byte_arr = byte_arr.getvalue()
        return byte_arr

    def __call__(
        self, obs_dict: Dict[str, Any], language_instruction: Optional[str] = None
    ):
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3, obs_dict[
            "image_primary"
        ].shape
        assert (
            language_instruction is not None
            and language_instruction in self.lang_to_goal_image
        ), "unknown language instruction"

        goal_images_path = self.lang_to_goal_image[language_instruction]
        goal_img = Image.open(goal_images_path)
        goal_img = np.array(goal_img)
        assert goal_img.shape == (256, 256, 3), goal_img.shape

        obs_image = obs_dict["image_primary"]
        assert obs_image.shape == (256, 256, 3), obs_image.shape

        goal_bytes = self.array_to_image_bytes(goal_img)
        observation_bytes = self.array_to_image_bytes(obs_image)

        response = requests.post(
            "http://" + self.host + ":" + str(self.port) + "/query_policy",
            files={
                "observation": ("observation.png", observation_bytes, "image/png"),
                "goal": ("goal.png", goal_bytes, "image/png"),
            },
        )

        assert response.status_code == 200, "Call to policy server failed"
        action = np.array(response.json(), dtype=np.float32)

        return action.copy()

    def reset(self):
        response = requests.post(
            "http://" + self.host + ":" + str(self.port) + "/reset"
        )
        assert response.status_code == 200, "Resetting to policy server failed"


policy_clients = {
    "open_web_client": OpenWebClient,
    "openvla_client": OpenVLAClient,
    "minivla_client": MiniVLAClient,
    "pi_zero_client": OpenPiZeroClient,
    "diffusion_policy_client": DiffusionPolicyClient,
}


if __name__ == "__main__":
    # manual test
    """
    forward port:
    ssh -L 8000:localhost:8000 -N -f -C -o ExitOnForwardFailure=yes $USER@markov.ist.berkeley.edu
    """
    action = requests.post(
        f"http://0.0.0.0:8000/act",
        json={
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "instruction": "do something",
            "unnorm_key": "bridge_orig",
        },
    ).json()
    print(action)
