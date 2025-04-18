"""
Script to test only the reset policy for a specified number of episodes.
"""
import argparse

from absl import flags
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.manipulator_env import ManipulatorEnv, StateEncoding
from manipulator_gym.utils.gym_wrappers import (
    CheckAndRebootJoints,
    ClipActionBoxBoundary,
    ConvertState2Proprio,
    LimitMotorMaxEffort,
    ResizeObsImageWrapper,
)
from ml_collections import config_flags

from auto_eval.robot.policy import policies
from auto_eval.robot.policy_clients import policy_clients
from auto_eval.utils.info import print_obvious, print_red
from auto_eval.visualization import visualize_image
from auto_eval.web_ui.launcher import RobotIDs, RobotIPs


def get_single_img(obs):
    img = obs["image_primary"]
    return img[-1] if img.ndim == 4 else img


def get_current_obs(obs):
    """In case that obs has history, only get the most current one"""
    current_obs = {}
    img = obs["image_primary"]
    proprio = obs["proprio"]

    if img.ndim == 4:  # has history
        current_obs["image_primary"] = img[-1]
    else:
        current_obs["image_primary"] = img

    if proprio.ndim == 2:  # has history
        current_obs["proprio"] = proprio[-1]
    else:
        current_obs["proprio"] = proprio

    return current_obs


def _create_env(manipulator_interface, workspace_bounds, reboot_with_sleep_pose=True):
    env = ManipulatorEnv(
        manipulator_interface=manipulator_interface,
        state_encoding=StateEncoding.POS_EULER,
        use_wrist_cam=False,
    )
    # work space boundary
    if workspace_bounds is not None:
        x_bounds = workspace_bounds["x"]
        y_bounds = workspace_bounds["y"]
        z_bounds = workspace_bounds["z"]
        env = ClipActionBoxBoundary(
            env, workspace_boundary=list(zip(*[x_bounds, y_bounds, z_bounds]))
        )

    env = ConvertState2Proprio(env)
    env = ResizeObsImageWrapper(
        env, resize_size={"image_primary": (256, 256), "image_wrist": (128, 128)}
    )
    env = CheckAndRebootJoints(
        env,
        force_reboot_per_episode=False,
        reboot_with_sleep_pose=reboot_with_sleep_pose,
    )

    return env


def reset_rollout(
    reset_env,
    reset_policy,
    reset_language_instruction,
    max_reset_steps,
    visualization_method,
    robot_id,
    i_episode,
):
    """Run the reset policy for one episode"""
    obs, info = reset_env.reset(moving_time=5)
    reset_policy.reset()  # some policies need to reset their state
    frames_recorder = []

    # run the reset policy
    for i in range(max_reset_steps):
        # visualize
        img = get_single_img(obs)
        visualize_image(
            visualization_method,
            img,
            reset_language_instruction,
            robot_id=robot_id,
            episode=i_episode,
            timestep=i,
        )

        actions = reset_policy(obs, reset_language_instruction)
        obs, reward, done, trunc, info = reset_env.step(actions)
        frames_recorder.append(get_single_img(obs))
        print(f"Reset step {i}")
        if done or trunc:
            break  # could be because motor failure or success

    # reset the environment after running the policy
    obs, info = reset_env.reset(moving_time=5)
    print_red(f"Reset episode {i_episode} completed")

    return frames_recorder


# Define flags
FLAGS = flags.FLAGS

# Define all command line flags
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the configuration.",
    lock_config=False,
)
flags.DEFINE_string("robot_ip", "localhost", "IP address of the robot action server.")
flags.DEFINE_integer("num_episodes", 5, "Number of episodes to run the reset policy.")
flags.DEFINE_integer(
    "max_reset_steps", 110, "Maximum number of steps per reset episode."
)
flags.DEFINE_string(
    "visualization_method",
    "display",
    "How to visualize the current image status. Limited to display, web_viewer, and none.",
)
flags.DEFINE_integer(
    "maximal_joint_effort",
    1500,
    "Maximum joint effort allowed",
)


def main(_):

    # Get robot ID for web viewer
    robot_name = {
        RobotIPs.WIDOWX_DRAWER: "widowx_drawer",
        RobotIPs.WIDOWX_SINK: "widowx_sink",
        RobotIPs.WIDOWX_CLOTH: "widowx_cloth",
    }[FLAGS.robot_ip]
    robot_id = RobotIDs.get_id(robot_name)

    # Determine if we need to go to sleep pose when rebooting motors
    if FLAGS.robot_ip in (RobotIPs.WIDOWX_DRAWER, RobotIPs.WIDOWX_CLOTH):
        reboot_with_sleep_pose = True
    elif FLAGS.robot_ip == RobotIPs.WIDOWX_SINK:
        reboot_with_sleep_pose = False
    else:
        raise ValueError("Unknown robot IP: ", FLAGS.robot_ip)

    # Create task specification
    reset_language_instruction = FLAGS.config.reset_language_cond
    print_red(f"Reset language instruction: {reset_language_instruction}")

    # Create interface and environment
    manipulator_interface = ActionClientInterface(host=FLAGS.robot_ip)

    # Create reset environment
    reset_env = _create_env(
        manipulator_interface,
        FLAGS.config.get("reset_workspace_bounds", FLAGS.config.workspace_bounds),
        reboot_with_sleep_pose,
    )
    reset_env = LimitMotorMaxEffort(
        reset_env,
        max_effort_limit=FLAGS.maximal_joint_effort,
    )

    # Initialize reset policy
    if "client" in FLAGS.config.reset_policy_type:
        reset_policy = policy_clients[FLAGS.config.reset_policy_type](
            **dict(FLAGS.config.reset_policy_kwargs)
        )
    else:
        reset_policy = policies[FLAGS.config.reset_policy_type](
            dict(FLAGS.config.reset_policy_kwargs)
        )
        if "sequence" in FLAGS.config.reset_policy_type:
            # sequence policies need to know the env to do env.reset in between
            reset_policy.env = reset_env

    # Run reset policy for specified number of episodes
    try:
        for i_episode in range(FLAGS.num_episodes):
            print_obvious(f"Running Reset Episode # {i_episode}")
            frames_recorder = reset_rollout(
                reset_env,
                reset_policy,
                reset_language_instruction,
                FLAGS.max_reset_steps,
                FLAGS.visualization_method,
                robot_id,
                i_episode,
            )
            input("Press Enter to continue...")

    finally:
        # Clean up resources
        try:
            reset_env.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            pass


if __name__ == "__main__":
    from absl import app

    app.run(main)
