#!/usr/bin/env python3
# NOTE: modified from: https://github.com/rail-berkeley/manipulator_gym/blob/main/teleop.py
# to also support recording transitions out as pickle file

import argparse
import copy
import os
import pickle

import cv2
import numpy as np
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from PIL import Image

from auto_eval.visualization import visualize_image


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def print_help(with_keyboard=True):
    print_yellow("  Teleop Controls:")

    if with_keyboard:
        print_yellow("    w, s : move forward/backward")
        print_yellow("    a, d : move left/right")
        print_yellow("    z, c : move up/down")
        print_yellow("    i, k:  rotate yaw")
        print_yellow("    j, l:  rotate pitch")
        print_yellow("    n  m:  rotate roll")
    else:
        print_yellow("    SpaceMouse control [x, y, z, rx, ry, rz]")

    print_yellow("    space: toggle gripper")
    print_yellow("    r: reset robot")
    print_yellow("    g: go to sleep")
    print_yellow("    t: turn on torque [experimental]")
    print_yellow("    /: reboot mulfuction motor [experimental]")
    print_yellow("    q: quit")


class PickleLogger:
    def __init__(self, filename, log_actions_only=False):
        self.filename = filename
        self.data = []
        self.log_actions_only = log_actions_only
        print("Logging pkl to: ", filename)

    def __call__(self, action, obs, reward, metadata=None, step_type=0):
        """Log a step, step_type=0 for transition, 1 for termination"""
        if self.log_actions_only:
            step = copy.deepcopy(action)
        else:
            step = copy.deepcopy(
                dict(
                    action=action,
                    observation=obs,
                    mask=0,
                    reward=reward,
                    metadata=metadata,
                    done=(step_type == 1),
                )
            )
        self.data.append(step)

    def close(self):
        print("Saving log to: ", self.filename)
        with open(self.filename, "wb") as f:
            pickle.dump(self.data, f)
        print("Done saving.")


###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Teleoperation to a manipulator server"
    )
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--visualization_method", type=str, default="display")
    parser.add_argument("--translation_diff", type=float, default=0.01)
    parser.add_argument("--rotation_diff", type=float, default=0.1)
    parser.add_argument("--use_spacemouse", action="store_true")
    parser.add_argument("--no_rotation", action="store_true")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--log_type", type=str, default="rlds")
    parser.add_argument("--log_actions_only", action="store_true", default=False)
    parser.add_argument("--log_lang_text", type=str, default="null task")
    parser.add_argument("--reset_pose", nargs="+", type=float, default=None)
    parser.add_argument("--track_workspace_limits", action="store_true", default=False)
    parser.add_argument("--pg", type=str, default=None, help="use paligemma detector")
    args = parser.parse_args()

    # if user specify where to reset the robot
    reset_kwargs = {}
    if args.reset_pose:
        # e.g. np.array([0.26, 0.0, 0.26, 0.0, math.pi/2, 0.0, 1.0]),
        assert len(args.reset_pose) == 7, "Reset pose must 7 values"
        reset_kwargs = {"target_state": args.reset_pose}

    # if user want to track workspace limits
    if args.track_workspace_limits:
        xyz_min, xyz_max = (
            np.array([1.0, 1.0, 1.0]) * np.inf,
            np.array([1.0, 1.0, 1.0]) * -np.inf,
        )

    interface = ActionClientInterface(host=args.ip, port=args.port)

    if args.pg:
        from auto_eval.success_detector.paligemma import PaligemmaDetector

        detector = PaligemmaDetector(
            processor_id="google/paligemma-3b-pt-224",
            model_id=args.pg,
            # device=device,
            quantize=True,
        )

    _dt = args.translation_diff
    _dr = args.rotation_diff

    if args.use_spacemouse:
        print("Using SpaceMouse for teleoperation.")
        from manipulator_gym.control.spacemouse import SpaceMouseControl

        spacemouse = SpaceMouseControl()

        def _get_spacemouse_action(with_rotation=True):
            sm_action, buttons = spacemouse.get_action()
            action = np.zeros(7)
            for i in range(3):
                action[i] = (
                    _dt if sm_action[i] > 0.5 else (-_dt if sm_action[i] < -0.5 else 0)
                )
            if with_rotation:
                for i in range(3, 6):
                    action[i] = (
                        _dr
                        if sm_action[i] > 0.5
                        else (-_dr if sm_action[i] < -0.5 else 0)
                    )
            return action

    else:
        keyboard_action_map = {
            ord("w"): np.array([_dt, 0, 0, 0, 0, 0, 0]),
            ord("s"): np.array([-_dt, 0, 0, 0, 0, 0, 0]),
            ord("a"): np.array([0, _dt, 0, 0, 0, 0, 0]),
            ord("d"): np.array([0, -_dt, 0, 0, 0, 0, 0]),
            ord("z"): np.array([0, 0, _dt, 0, 0, 0, 0]),
            ord("c"): np.array([0, 0, -_dt, 0, 0, 0, 0]),
            ord("i"): np.array([0, 0, 0, _dr, 0, 0, 0]),
            ord("k"): np.array([0, 0, 0, -_dr, 0, 0, 0]),
            ord("j"): np.array([0, 0, 0, 0, _dr, 0, 0]),
            ord("l"): np.array([0, 0, 0, 0, -_dr, 0, 0]),
            ord("n"): np.array([0, 0, 0, 0, 0, _dr, 0]),
            ord("m"): np.array([0, 0, 0, 0, 0, -_dr, 0]),
        }

    def _get_full_obs():
        obs = {
            "image_primary": interface.primary_img,
            "state": np.concatenate(
                [interface.eef_pose[:6], [0.0], [interface.gripper_state]],  # padding
                dtype=np.float32,
            ),
        }
        if interface.wrist_img is not None:
            obs["image_wrist"] = interface.wrist_img
        return obs

    if args.log_dir:
        if args.log_type == "rlds":
            import tensorflow_datasets as tfds
            from oxe_envlogger.data_type import get_gym_space
            from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType

            # Create RLDSLogger
            logger = RLDSLogger(
                observation_space=get_gym_space(_get_full_obs()),
                action_space=get_gym_space(np.zeros(7, dtype=np.float32)),
                dataset_name="expert_demos",
                directory=args.log_dir,
                max_episodes_per_file=1,
                step_metadata_info={"language_instruction": tfds.features.Text()},
            )
        elif args.log_type == "pkl":
            logger = PickleLogger(
                filename=args.log_dir, log_actions_only=args.log_actions_only
            )
        else:
            raise ValueError("Invalid log type: ", args.log_type)

        _mdata = {"language_instruction": args.log_lang_text}

    ############# Wrap execution of actions for logging #############
    def _execute_action(action, first_step=False):
        obs = _get_full_obs()
        interface.step_action(action)

        if args.track_workspace_limits:
            global xyz_min, xyz_max
            xyz_min = np.minimum(xyz_min, interface.eef_pose[:3])
            xyz_max = np.maximum(xyz_max, interface.eef_pose[:3])

        if args.log_dir:

            if args.log_type == "rlds":
                step_type = (
                    RLDSStepType.RESTART if first_step else RLDSStepType.TRANSITION
                )
            elif args.log_type == "pkl":
                step_type = 0

            logger(action, obs, 0.0, metadata=_mdata, step_type=step_type)

    ############# Wrap execution of reset for logging #############
    def _execute_reset():
        null_action = np.array([0, 0, 0, 0, 0, 0, 1])
        if args.log_dir:
            obs = _get_full_obs()
            step_type = 1 if args.log_type == "pkl" else RLDSStepType.TERMINATION
            logger(null_action, obs, 1.0, metadata=_mdata, step_type=step_type)

        interface.reset(**reset_kwargs)

        if args.log_dir:
            obs = _get_full_obs()
            step_type = 0 if args.log_type == "pkl" else RLDSStepType.RESTART
            logger(null_action, obs, 0.0, metadata=_mdata, step_type=step_type)

    ########################## Main loop ##########################
    print_help(not args.use_spacemouse)
    is_open = 1
    running = True
    image_counter = 0  # Add counter for image names when saving
    os.makedirs("teleop_saved_images", exist_ok=True)  # save image to this dir

    _execute_action(np.array([0, 0, 0, 0, 0, 0, is_open]), first_step=True)

    while running:
        # Check for key press
        key = cv2.waitKey(40) & 0xFF

        # escape key to quit
        if key == ord("q"):
            print("Quitting teleoperation.")
            running = False
            continue

        # space bar to change gripper state
        elif key == ord(" "):
            is_open = 1 - is_open
            print("Gripper is now: ", is_open)
            _execute_action(np.array([0, 0, 0, 0, 0, 0, is_open]))
        elif key == ord("r"):
            print("Resetting robot...")
            _execute_reset()
            is_open = 1  # interface.gripper_state > 0.5
            print_help()
        elif key == ord("g"):
            print("Going to sleep... make sure server has this method")
            kwargs = {"go_sleep": True}
            interface.reset(**kwargs)
            print_help()
        elif key == ord("t"):
            print("Turning on torque for the robot...")
            interface.custom_fn("enable_torque")
        elif key == ord("y"):
            status = interface.custom_fn("get_torque_status")
            print(status)
        elif key == ord("/"):
            print("[experimental feature] reboot mulfuction motor for widowx")
            widowx_joints = [
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
                "gripper",
            ]
            res = interface.custom_fn("motor_status")
            print("Motor status: ", res)

            # check motor failure and reset it
            for i, status in enumerate(res):
                joint_name = widowx_joints[i]
                print("Rebooting motor: ", joint_name)
                interface.custom_fn("reboot_motor", joint_name=joint_name)

            print_help()
        elif key == ord("b"):
            """save the image with resize to 256x256 with incrementing numbers"""
            image = cv2.resize(interface.primary_img, (256, 256))
            image = Image.fromarray(image)
            image_counter += 1
            image.save(f"teleop_saved_images/image{image_counter}.png")
            print(f"Saved image{image_counter}.png")
        elif key == ord("p") and args.pg:
            """To eval the paligemma detector"""
            # prompt = "is the drawer open? answer yes or no"  # hard coded
            # prompt = "is the eggplant in the sink or in the basket? answer sink or basket or invalid"  # hard coded
            prompt = "is the cloth folded or unfolded? answer yes or no"  # hard coded
            image = Image.fromarray(interface.primary_img).convert("RGB")
            res = detector(prompt, image)
            print("pali-gemma detector result: ", res)

        # command robot with spacemouse (continuous)
        if args.use_spacemouse:
            action = _get_spacemouse_action(not args.no_rotation)
            action[-1] = is_open

            # if action is more than 0.001 or less than -0.001 then move
            if np.any(action[:6] > 0.001) or np.any(action[:6] < -0.001):
                _execute_action(action)
            # keep command gripper if gripper state is different
            if (interface.gripper_state > 0.5) != is_open:
                _execute_action(action)

        # command robot with keyboard (event based)
        elif key in keyboard_action_map:
            action = keyboard_action_map[key]
            action[-1] = is_open
            _execute_action(action)

        visualize_image(
            method=args.visualization_method,
            img=interface.primary_img,
            language_instruction="teleop",
            wait_time=0,  # no wait, keep pressing keyboard to update
        )

    if args.track_workspace_limits:
        print("Workspace limits during teleop: ")
        print("x_min: ", xyz_min[0], " // x_max: ", xyz_max[0])
        print("y_min: ", xyz_min[1], " // y_max: ", xyz_max[1])
        print("z_min: ", xyz_min[2], " // z_max: ", xyz_max[2])

    if args.log_dir:
        logger.close()
        print("Done logging.")

    cv2.destroyAllWindows()
    print("Teleoperation ended.")
