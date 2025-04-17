import os
import pickle
import time
from typing import Any, Dict, Optional

import cv2
import gym
import numpy as np

try:
    import jax
except ImportError:
    pass  # not every policy needs jax

from auto_eval.robot.policy_clients import GoalImageGeneratorClient


def unnormalize_actions(actions, metadata, normalization_type="normal"):
    """normalize the first 6 dimensions of the widowx actions

    Args:
        actions: actions to unnormalize
        metadata: metadata containing mean, std, q01, and q99
        normalization_type: type of normalization to use ("normal" or "bounds")
    """
    if normalization_type not in ["normal", "bounds"]:
        raise ValueError("normalization_type must be either 'normal' or 'bounds'")

    gripper_action = 1.0 if actions[6] > 0 else 0.0

    if normalization_type == "bounds":
        if "q01" not in metadata or "q99" not in metadata:
            raise ValueError(
                "metadata must contain q01 and q99 for bounds normalization"
            )
        # Denormalize using bounds
        actions_except_gripper = (actions[:6] + 1) / 2 * (
            metadata["q99"][:6] - metadata["q01"][:6]
        ) + metadata["q01"][:6]
    else:  # normal
        if "mean" not in metadata or "std" not in metadata:
            raise ValueError(
                "metadata must contain mean and std for normal normalization"
            )
        # Original gaussian normalization
        actions_except_gripper = (
            metadata["std"][:6] * actions[:6] + metadata["mean"][:6]
        )

    actions = np.concatenate([actions_except_gripper, np.array([gripper_action])])
    return actions


def normalize_actions(actions, metadata, normalization_type="normal"):
    """normalize  the actions

    Args:
        actions: actions to normalize
        metadata: metadata containing mean, std, q01, and q99
        normalization_type: type of normalization to use ("normal" or "bounds")
    """
    if normalization_type not in ["normal", "bounds"]:
        raise ValueError("normalization_type must be either 'normal' or 'bounds'")

    if normalization_type == "bounds":
        if "q01" not in metadata or "q99" not in metadata:
            raise ValueError(
                "metadata must contain q01 and q99 for bounds normalization"
            )
        # Normalize using bounds
        actions = (
            2 * (actions - metadata["q01"]) / (metadata["q99"] - metadata["q01"]) - 1
        )
    else:  # normal
        if "mean" not in metadata or "std" not in metadata:
            raise ValueError(
                "metadata must contain mean and std for normal normalization"
            )
        # Original gaussian normalization
        actions = (actions - metadata["mean"]) / metadata["std"]
    return actions


def create_bridge_example_batch(batch_size, img_size):
    """create a dummy batch of the correct shape to create the agent"""
    example_batch = {
        "observations": {
            "proprio": np.zeros(
                (
                    batch_size,
                    7,
                ),
            ),
            "image": np.zeros(
                (batch_size, img_size, img_size, 3),
            ),
        },
        "goals": {
            "image": np.zeros(
                (batch_size, img_size, img_size, 3),
            ),
        },
        "actions": np.zeros(
            (
                batch_size,
                7,
            ),
        ),
    }
    return example_batch


class BasePolicy:
    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = device
        self.agent = self.create_agent()

    def create_agent(self):
        raise NotImplementedError

    def __call__(
        self, obs_dict: Dict[str, Any], language_instruction: Optional[str] = None
    ):
        raise NotImplementedError

    def reset(self):
        pass


class RandomPolicy(BasePolicy):
    def create_agent(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    def __call__(self, obs_dict, language_instruction):
        """
        Random action
        """
        action = self.action_space.sample()
        action[-1] = np.random.choice([0, 1])
        return action


class OctoPolicy(BasePolicy):
    def create_agent(self):
        # lazy imports
        import jax
        from octo.model.octo_model import OctoModel

        agent = OctoModel.load_pretrained(
            "hf://rail-berkeley/octo-small"
        )  # don't use 1.5
        self.task = None  # created later
        return agent

    def __call__(self, obs_dict, language_instruction):
        assert all(key in obs_dict.keys() for key in ["image_primary", "proprio"])
        pose = obs_dict["proprio"]

        if self.task is None:
            # assumes that each Octo model doesn't receive different tasks
            self.task = self.agent.create_tasks(texts=[language_instruction])
            # self.task = self.agent.create_tasks(goals={"image_primary": img})   # for goal-conditioned

        actions = self.agent.sample_actions(
            jax.tree_map(lambda x: x[None], obs_dict),
            self.task,
            unnormalization_statistics=self.agent.dataset_statistics["bridge_dataset"][
                "action"
            ],
            rng=jax.random.PRNGKey(0),
        )
        # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
        actions = actions[0]  # note that actions here could be chucked

        return actions


class OpenVLAPolicy(BasePolicy):
    def create_agent(self):
        import peft
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # Load Processor & VLA
        processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        )
        base_vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # For fine-tuned VLA policies, need to add the Lora adapters and new dataset stats
        if "lora_adapter_dir" in self.config:
            assert (
                "dataset_stats_path" in self.config
            ), "fine-tuned VLA usually requires custom dataset stats"
            adapter_dir = self.config["lora_adapter_dir"]
            print(f"Loading LORA Adapter from: {adapter_dir}")
            vla = peft.PeftModel.from_pretrained(base_vla, adapter_dir)
            vla = (
                vla.merge_and_unload()
            )  # this merges the adapter into the model for faster inference
        else:
            vla = base_vla
        vla = vla.to(self.device)

        # Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if "dataset_stats_path" in self.config:
            dataset_stats_path = self.config["dataset_stats_path"]
            assert (
                "dataset_statistics.json" in dataset_stats_path
            ), "Please provide the correct dataset statistics file."
            path = os.path.expanduser(dataset_stats_path)
            print(f"Loading custom dataset statistics .json from: {path}")

            with open(path, "r") as f:
                import json

                vla.norm_stats = json.load(f)

                # assume only one key in the .json file and gets the key
                dataset_names = vla.norm_stats.keys()
                assert (
                    len(dataset_names) == 1
                ), "Only one dataset name should be in the .json file."
                unnorm_key = list(dataset_names)[0]
        else:
            unnorm_key = "bridge_orig"

        print(f"Un-normalization key: {unnorm_key}")

        # Construct a callable function to predict actions
        def _call_action_fn(obs_dict, language_instruction):
            image = obs_dict["image_primary"]
            image_cond = Image.fromarray(image)
            prompt = f"In: What action should the robot take to {language_instruction}?\nOut:"
            inputs = processor(prompt, image_cond).to(self.device, dtype=torch.bfloat16)

            # Predict Action (7-DoF; un-normalize for BridgeData V2)
            action = vla.predict_action(
                **inputs, unnorm_key=unnorm_key, do_sample=False
            )
            assert (
                len(action) == 7
            ), f"Action size should be in x, y, z, rx, ry, rz, gripper"

            return action

        self._call_action_fn = _call_action_fn

    def __call__(self, obs_dict, language_instruction):
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3
        return self._call_action_fn(obs_dict, language_instruction)


class OpenPiZero(BasePolicy):
    def __init__(self, config):
        super().__init__(config)

        # load bridge dataset statistics [hard coded]
        import json

        dataset_statistics_path = os.path.expanduser(
            "~/checkpoints/openvla/converted/dataset_statistics.json"
        )
        try:
            with open(dataset_statistics_path, "r") as f:
                self.dataset_stats = json.load(f)["bridge_orig"]
            self.dataset_stats["action"] = {
                k: np.array(v) for k, v in self.dataset_stats["action"].items()
            }
            self.dataset_stats["proprio"] = {
                k: np.array(v) for k, v in self.dataset_stats["proprio"].items()
            }
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(
                f"Failed to load dataset statistics from {dataset_statistics_path}: {e}"
            )

    def create_agent(self):
        # lazy imports
        import torch
        from omegaconf import OmegaConf
        from src.model.vla.pizero import PiZeroInference
        from src.model.vla.processing import VLAProcessor
        from transformers import AutoTokenizer

        # load in fixed configs
        checkpoint_path = os.path.expanduser(
            "~/checkpoints/open-pi-zero/bridge_beta_step19296_2024-12-26_22-30_42.pt"
        )
        cfg = OmegaConf.load(
            os.path.expanduser("~/repos/open-pi-zero/config/eval/bridge.yaml")
        )
        os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser(
            "~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224"
        )

        # determine flow matching schedule
        if "uniform" in checkpoint_path:
            cfg.flow_sampling = "uniform"
        if "beta" in checkpoint_path:
            cfg.flow_sampling = "beta"

        def load_checkpoint(model, path):
            """load to cpu first, then move to gpu"""
            data = torch.load(path, weights_only=True, map_location="cpu")
            # remove "_orig_mod." prefix if saved model was compiled
            data["model"] = {
                k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
            }
            model.load_state_dict(data["model"], strict=True)
            print(f"Loaded model from {path}")

        # model
        use_bf16 = True  # hard coded
        use_torch_compile = False  # hard coded
        model = PiZeroInference(cfg, use_ddp=False)
        load_checkpoint(model, checkpoint_path)
        model.freeze_all_weights()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        model.to(self.dtype)
        model.to(self.device)
        if use_torch_compile:
            # model being compiled in the first batch which takes some time
            model = torch.compile(
                model,
                mode="reduce-overhead",  # "reduce-overhead; max-autotune(-no-cudagraphs)
                # backend="inductor", # default: inductor; cudagraphs
            )
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        model.eval()

        self.model = model

        # tokenizer and processer --- assume paligemma for now
        tokenizer = AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224", padding_side="right"
        )
        num_image_tokens = 256  # hard coded
        max_seq_len = 276  # hard coded, 256 for image + max 20 for text
        tokenizer_padding = "max_length"
        self.processor = VLAProcessor(
            tokenizer,
            num_image_tokens=num_image_tokens,
            max_seq_len=max_seq_len,
            tokenizer_padding=tokenizer_padding,
        )

    def __call__(self, obs_dict, language_instruction):
        import numpy as np
        import torch  # lazy
        from src.utils.geometry import mat2euler, quat2mat

        # preprocess proprio from 8D to 7D
        proprio = obs_dict["proprio"]
        assert len(proprio) == 8, "original proprio should be size 8"
        rm_bridge = quat2mat(proprio[3:7])
        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        rpy_bridge_converted = mat2euler(rm_bridge @ default_rot.T)
        gripper = proprio[7]
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper],
            ]
        )

        # normalize proprios - gripper opening is normalized
        eps = 1e-8
        if self.config["action_normalization_type"] == "bounds":
            proprio = (
                2
                * (raw_proprio - self.dataset_stats["proprio"]["q01"])
                / (
                    self.dataset_stats["proprio"]["q99"]
                    - self.dataset_stats["proprio"]["q01"]
                    + eps
                )
                - 1
            )
            proprio = np.clip(proprio, -1, 1)
        elif self.config["action_normalization_type"] == "normal":
            proprio = (raw_proprio - self.dataset_stats["proprio"]["mean"]) / (
                self.dataset_stats["proprio"]["std"] + eps
            )

        # process image
        img = obs_dict["image_primary"]
        img = cv2.resize(
            img, (224, 224), interpolation=cv2.INTER_LANCZOS4
        )  # hard coded
        images = torch.as_tensor(img, dtype=torch.uint8).permute(2, 0, 1)[None]

        tokenized_inputs = self.processor(
            text=[language_instruction],
            images=images,
        )
        (
            causal_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
        ) = self.model.build_causal_mask_and_position_ids(
            tokenized_inputs["attention_mask"], dtype=self.dtype
        )
        image_text_proprio_mask, action_mask = self.model.split_full_mask_into_submasks(
            causal_mask
        )
        inputs = {
            "input_ids": tokenized_inputs["input_ids"],
            "pixel_values": tokenized_inputs["pixel_values"].to(self.dtype),
            "image_text_proprio_mask": image_text_proprio_mask,
            "action_mask": action_mask,
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": torch.as_tensor(proprio, dtype=self.dtype)[None, None],
        }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():  # speeds up
            actions = self.model(**inputs)
            actions = actions[0]  # remove batch dimension

        # normalize the actions
        normalized_actions = []
        for action in actions:
            # actions is an action chunk of shape [pred_horizon, action_dim]
            action = unnormalize_actions(
                action.float().cpu().numpy(),
                self.dataset_stats["action"],
                normalization_type=self.config["action_normalization_type"],
            )
            normalized_actions.append(action)
        normalized_actions = np.array(normalized_actions)

        return normalized_actions


class GCPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        self.action_statistics = {
            "mean": self.config["ACT_MEAN"],
            "std": self.config["ACT_STD"],
        }

    def create_agent(self):
        # lazy imports
        from flax.training import checkpoints
        from jaxrl_m.agents import agents
        from jaxrl_m.vision import encoders

        # encoder
        encoder_def = encoders[self.config["encoder"]](**self.config["encoder_kwargs"])

        # create agent
        example_batch = create_bridge_example_batch(
            batch_size=1, img_size=self.config["obs_image_size"]
        )
        self.rng = jax.random.PRNGKey(self.config["seed"])
        self.rng, construct_rng = jax.random.split(self.rng)
        agent = agents[self.config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **self.config["agent_kwargs"],
        )
        assert os.path.exists(self.config["checkpoint_path"]), "Checkpoint not found"
        agent = checkpoints.restore_checkpoint(self.config["checkpoint_path"], agent)
        return agent

    def __call__(
        self,
        obs_dict,
        language_instruction,
        deterministic=True,
    ):
        """the run loop code should pass in a `goal` field in the obs_dict
        Otherwise, the policy will look for a pre-specified goal in the `goal_images/` dir,
        with the filename being the language instruction
        """
        assert all(key in obs_dict.keys() for key in ["image_primary", "proprio"])
        obs_image = obs_dict["image_primary"]
        pose = obs_dict["proprio"]

        # get the goal image
        try:
            goal_image = obs_dict["goal"]
        except KeyError:
            goal_images_dir = self.config.get("goal_images_dir", "goal_images")
            print(
                f"Goal not provided in obs_dict, looking for pre-specified goal from [{goal_images_dir}]"
            )
            goal_file = os.path.join(goal_images_dir, language_instruction + ".png")
            assert os.path.exists(
                goal_file
            ), f"Goal file {goal_file} not found, and not provided in obs_dict"
            goal_image = cv2.imread(goal_file)

        assert obs_image.shape == (
            self.config["obs_image_size"],
            self.config["obs_image_size"],
            3,
        ), "Bad input obs image shape"
        assert goal_image.shape == (
            self.config["obs_image_size"],
            self.config["obs_image_size"],
            3,
        ), f"Bad input goal image shape, {goal_image.shape}"

        self.rng, action_rng = jax.random.split(self.rng)
        actions_result = self.agent.sample_actions(
            {"image": obs_image[np.newaxis, ...]},
            {"image": goal_image[np.newaxis, ...]},
            temperature=0.0,
            argmax=deterministic,
            seed=None if deterministic else action_rng,
        )
        # NOTE(YL), Due to different version of jaxrl_m
        # this checks if action result is a tuple of (action, action_mode), or action
        if isinstance(actions_result, tuple):
            actions, action_mode = actions_result
        else:
            actions = actions_result
        actions = np.array(actions)[0]  # unbatch
        actions = unnormalize_actions(
            actions,
            self.action_statistics,
            normalization_type=self.config["action_normalization_type"],
        )

        return actions


class SOARPolicy(GCPolicy):
    """from https://arxiv.org/abs/2407.20635
    Essentially, this is a decomposed language-conditioned policy.
    The language is used to generate a sub-goal image with SuSIE,
    and the low-level goal-conditioned policy tries to reach the goal
    """

    def __init__(self, config):
        super().__init__(config)
        # user able to define {susie_client: {host: "localhost", port: 8001}} in config
        _use_client = False
        self.goal_generator = self.create_goal_generator(_use_client)
        # don't generate susie goal every step, just generate one every goal_horizon
        self.goal_horizon = 20
        self.policy_step = 0
        self.current_goal = None

    def create_goal_generator(self, use_client):
        if use_client:
            if "susie_client" in self.config:
                # for user to specify the host and port in the config
                susie = GoalImageGeneratorClient(
                    host=self.config["susie_client"]["host"],
                    config=self.config["susie_kwargs"],
                    ssh_port=self.config["susie_client"].get("ssh_port", 2222),
                    port=self.config["susie_client"].get("port", 8001),
                )
            else:
                susie = GoalImageGeneratorClient(self.config["susie_kwargs"])
        else:
            susie = GoalImageGenerator(self.config["susie_kwargs"])
        return susie

    def __call__(self, obs_dict, language_instruction):
        # convert language to goal image
        if self.policy_step % self.goal_horizon == 0:
            print("Generating new SuSIE goal, step:", self.policy_step)
            start_time = time.time()
            self.current_goal = self.goal_generator(
                obs_dict["image_primary"], language_instruction
            )
            print(
                f"Done generating new SuSIE goal, took: {time.time() - start_time:.2f}s"
            )
        else:
            assert self.current_goal is not None

        goal_image = self.current_goal
        self.policy_step += 1

        # show goal image
        # TODO: visualize goals with local/webviewer
        if self.config.get("show_goal_image", False):
            cv2.imshow(
                f"SuSIE goal: {language_instruction}",
                cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR),  # cv2 reads in BGR
            )

        action = super().__call__(
            {
                "image_primary": obs_dict["image_primary"],
                "goal": goal_image,
                "proprio": obs_dict["proprio"],
            },
            language_instruction=language_instruction,
            deterministic=True,
        )
        return action

    def reset(self):
        self.policy_step = 0
        self.current_goal = None


class GoalImageGenerator:
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

    def __call__(self, image_obs: np.ndarray, prompt: str):
        assert image_obs.shape == (
            self.image_size,
            self.image_size,
            3,
        ), f"Bad input image shape {image_obs.shape}"
        return self.diffusion_sample_func(image_obs, prompt)


class RecordedPolicy(BasePolicy):
    """Replay some recorded policy with deterministic actions"""

    def create_agent(self):
        saved_path = self.config["policy_save_path"]
        with open(saved_path, "rb") as f:
            self.action_seq = pickle.load(f)  # list of transition dicts
        self.current_step = 0

    def __call__(self, obs_dict, language_instruction):
        if self.current_step >= len(self.action_seq):
            action = np.array([0.0] * 6 + [1.0])
        else:
            action = self.action_seq[self.current_step]
        self.current_step += 1
        return action

    def reset(self):
        self.current_step = 0


class SequenceRecordedPolicy(BasePolicy):
    """
    a sequence of recorded policies, executed deterministically one after another.
    Optionally do env.reset() between each policy.

    e.g. the open drawer scripted policy is the sequence of:
     - close drawer fully
     - env.reset()
     - open drawer fully
    """

    def create_agent(self):
        # load the policies
        policy_paths = self.config["policy_save_paths"]
        self.policies = [
            RecordedPolicy(config={"policy_save_path": path}) for path in policy_paths
        ]

        # keep track of the number of steps for each policy
        self.num_steps = [len(policy.action_seq) for policy in self.policies]
        self.seq_total_steps = np.cumsum(self.num_steps)
        self.current_step = 0

        # whether to reset the env (e.g. move to neutral position) between each policy
        self.env_reset_in_between = self.config.get("env_reset_in_between", True)

    def _currently_on_policy_i(self):
        for i, total_steps in enumerate(self.seq_total_steps):
            if self.current_step <= total_steps - 1:
                return i
        return -1

    def _in_between_policy(self):
        # whether the current step is in between policies
        return self.current_step in self.seq_total_steps - 1

    def __call__(self, obs_dict, language_instruction):
        # need to know which policy to call
        i = self._currently_on_policy_i()
        if i == -1:
            # the current step is out of range
            action = np.array([0.0] * 6 + [1.0])
            return action
        action = self.policies[i](obs_dict, language_instruction)

        # need to know whether to reset the env
        if self.env_reset_in_between and self._in_between_policy():
            self.env.reset()

        self.current_step += 1
        return action

    def reset(self):
        self.current_step = 0
        for policy in self.policies:
            policy.reset()


policies = {
    "jaxrl_gc_policy": GCPolicy,
    "octo": OctoPolicy,
    "openvla": OpenVLAPolicy,
    "soar": SOARPolicy,
    "scripted": RecordedPolicy,
    "scripted_sequence": SequenceRecordedPolicy,
    "pizero": OpenPiZero,
}
