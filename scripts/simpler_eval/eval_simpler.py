"""
Test script to run the eval


NOTE: this uses: https://github.com/youliangtan/SimplerEnv

python eval_simpler.py --test --env widowx_open_drawer
python eval_simpler.py --test --env widowx_close_drawer
python eval_simpler.py --test --env widowx_put_eggplant_in_basket
python eval_simpler.py --test --env widowx_put_eggplant_in_sink

# Openvla api call
python eval_simpler.py --env widowx_open_drawer --vla_url http://XXX.XXX.XXX.XXX:6633/act
python eval_simpler.py --env widowx_close_drawer --vla_url http://XXX.XXX.XXX.XXX:6633/act


# octo policy
python eval_simpler.py --env widowx_open_drawer --octo
python eval_simpler.py --env widowx_close_drawer --octo

# gcbc policy
# NOTE: the config is located in eval_config.py
# this also requires $PWD/goal_images/task_name.png
python eval_simpler.py --env widowx_open_drawer --gcbc
python eval_simpler.py --env widowx_close_drawer --gcbc


Policy server:

Open-pi-zero policy: https://github.com/youliangtan/open-pi-zero

Server is located in:
    open-pi-zero$ python scripts/open_pi0_server.py  --checkpoint_path bridge_beta_step19296_2024-12-26_22-30_42.pt    --use_bf16     --use_torch_compile
"""

import argparse
import os
from collections import deque

import cv2

# import gymnasium as gym
import gym
import json_numpy
import numpy as np

# for openvla api call
import requests
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from auto_eval.robot.policy import (
    BasePolicy,
    GCPolicy,
    OctoPolicy,
    RandomPolicy,
    SOARPolicy,
)

try:
    from transforms3d import euler as te
    from transforms3d import quaternions as tq
except ImportError:
    pass

json_numpy.patch()

print_green = lambda x: print("\033[92m {}\033[00m".format(x))

# print numpy array with 2 decimal points
np.set_printoptions(precision=2)


########################################################################
class OpenVLAPolicy(BasePolicy):
    def create_agent(self):
        assert "url" in self.config, "url not found in config"
        self.url = self.config["url"]

    def __call__(self, obs_dict, language_instruction):
        """
        Openvla api call to get the action.
            obs_dict : dict
            language_instruction : str
        """
        img = obs_dict["image_primary"]
        img = cv2.resize(img, (256, 256))  # ensure size is 256x256
        action = requests.post(
            self.url,
            json={
                "image": img,
                "instruction": language_instruction,
                "proprio": obs_dict["proprio"],
                "unnorm_key": "bridge_orig",
            },
        ).json()
        return np.array(action)


########################################################################


class WrapSimplerEnv(gym.Wrapper):
    def __init__(self, env):
        super(WrapSimplerEnv, self).__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    def reset(self):
        obs, reset_info = self.env.reset()
        obs, additional_info = self._process_obs(obs)
        reset_info.update(additional_info)
        return obs, reset_info

    def step(self, action):
        """
        NOTE action is 7 dim
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
        gripper: -1 close, 1 open
        """
        obs, reward, done, truncated, info = self.env.step(action)
        obs, additional_info = self._process_obs(obs)
        info.update(additional_info)
        return obs, reward, done, truncated, info

    def _process_obs(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = self._process_proprio(obs)
        return (
            {
                "image_primary": cv2.resize(img, (256, 256)),
                "proprio": proprio,
            },
            {
                "original_image_primary": img,
            },
        )

    def _process_proprio(self, obs):
        """
        Process proprioceptive information
        """
        # TODO: should we use rxyz instead of quaternion?
        # 3 dim translation, 4 dim quaternion rotation and 1 dim gripper
        eef_pose = obs["agent"]["eef_pos"]
        # joint_angles = obs['agent']['qpos'] # 8-dim vector joint angles
        return eef_pose


class BridgeSimplerStateWrapper(gym.Wrapper):
    """
    NOTE(YL): this converts the prorio from the default
    [x, y, z, qx, qy, qz, qw, gripper (0)]

    to

    [x, y, z, roll, pitch, yaw, <PAD, default_len=0>, gripper]

    is adapted from:
    https://github.com/allenzren/open-pi-zero/blob/main/src/agent/env_adapter/simpler.py
    """

    def __init__(self, env, pad_proprio_len=0):
        super(BridgeSimplerStateWrapper, self).__init__(env)
        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203

        # NOTE: now proprio is size 7
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7 + pad_proprio_len,),
                    dtype=np.float32,
                ),
            }
        )
        self._pad_proprio_len = pad_proprio_len

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs["proprio"] = self._preprocess_proprio(obs)
        return obs, info

    def step(self, action):
        action[-1] = self._postprocess_gripper(action[-1])
        obs, reward, done, trunc, info = super().step(action)
        obs["proprio"] = self._preprocess_proprio(obs)
        return obs, reward, done, trunc, info

    def _preprocess_proprio(self, obs: dict) -> np.array:
        # convert ee rotation to the frame of top-down
        # proprio = obs["agent"]["eef_pos"]
        proprio = obs["proprio"]
        assert len(proprio) == 8, "original proprio should be size 8"
        rm_bridge = tq.quat2mat(proprio[3:7])
        rpy_bridge_converted = te.mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        )
        assert len(raw_proprio) == 7, "proprio should be size 7"
        if self._pad_proprio_len > 0:
            raw_proprio = np.concatenate(
                [
                    raw_proprio[:6],
                    np.zeros(self._pad_proprio_len),
                    [raw_proprio[6]],
                ]
            )
        assert (
            len(raw_proprio) == 7 + self._pad_proprio_len
        ), "proprio should be size 7 + pad_proprio_len"
        return raw_proprio

    def _postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler
        action_gripper = 2.0 * (action > 0.5) - 1.0
        return action_gripper


########################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="widowx_close_drawer")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--octo", action="store_true")
    parser.add_argument("--gcbc", action="store_true")
    parser.add_argument("--susie", action="store_true")
    parser.add_argument("--openvla", action="store_true")
    parser.add_argument("--minivla", action="store_true")
    parser.add_argument("--openpi0", action="store_true")
    parser.add_argument("--show_img", action="store_true")
    parser.add_argument("--server_host", type=str, default="10.110.17.183")
    parser.add_argument("--eval_count", type=int, default=10)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--output_video_dir", type=str, default=None)
    args = parser.parse_args()

    base_env = simpler_env.make(args.env)
    base_env._max_episode_steps = args.episode_length  # override the max episode length

    instruction = base_env.get_language_instruction()

    env = WrapSimplerEnv(base_env)

    print("Instruction", instruction)

    if "widowx" in args.env:
        print(
            "Wrap Simpler with bridge state wrapper for proprio and action convention"
        )
        if args.octo:
            # NOTE: octo uses a 8 dim proprio
            env = BridgeSimplerStateWrapper(env, pad_proprio_len=1)
        else:
            env = BridgeSimplerStateWrapper(env)

    if args.test:
        policy = RandomPolicy(config={})

    elif args.octo:
        policy_config = {}
        policy = OctoPolicy(policy_config)
        from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper

        env = HistoryWrapper(env, horizon=2)
        env = TemporalEnsembleWrapper(env, 4)

    elif args.gcbc:
        # from eval_config import jaxrl_gc_policy_kwargs
        from configs.eval_config import jaxrl_gc_policy_kwargs

        jaxrl_gc_policy_kwargs["goal_images_dir"] = "goal_images/simpler"
        policy = GCPolicy(jaxrl_gc_policy_kwargs)

    elif args.susie:
        from configs.eval_config import soar_policy_kwargs

        soar_policy_kwargs["susie_client"] = dict(
            host=args.server_host,
            ssh_port=None,
        )
        soar_policy_kwargs["show_goal_image"] = args.show_img
        policy = SOARPolicy(soar_policy_kwargs)

    elif args.openvla or args.minivla:
        policy_config = {"url": f"http://{args.server_host}:8000/act"}
        policy = OpenVLAPolicy(policy_config)

    elif args.openpi0:
        policy_config = {"url": f"http://{args.server_host}:8000/act"}
        from octo.utils.gym_wrappers import TemporalEnsembleWrapper

        policy = OpenVLAPolicy(policy_config)
        env = TemporalEnsembleWrapper(env, 4)

    else:
        raise ValueError("No policy specified, --help for more info")

    success_count = 0

    for i in range(args.eval_count):
        print_green(f"Evaluate Episode {i}")

        done, truncated = False, False
        obs, info = env.reset()
        print("Initial proprio", obs["proprio"])
        if args.susie:
            # susie policy is stateful
            policy.reset()

        images = []

        step_count = 0
        while not (done or truncated):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            # image = get_image_from_maniskill2_obs_dict(env, obs)
            image = obs["image_primary"]

            if args.output_video_dir:
                images.append(image)

            # show image
            full_image = info["original_image_primary"]
            if args.show_img:
                cv2.imshow("Image", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            action = policy(obs, instruction)

            # print(f"Step {step_count} Action: {action}")
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1

        # check if the episode is successful
        if done:
            success_count += 1
            print_green(f"Episode {i} Success")
        else:
            print_green(f"Episode {i} Failed")

        # save mp4 video of the current episode
        if args.output_video_dir:
            video_name = f"{args.output_video_dir}/{args.env}_{i}.mp4"
            print(f"Save video to {video_name}")
            height, width, _ = images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
            for image in images:
                out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            out.release()

        episode_stats = info.get("episode_stats", {})
        print("Episode stats", episode_stats)
        print_green(f"Success rate: {success_count}/{i + 1}")
