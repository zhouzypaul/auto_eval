"""
utilities for checking the robot status
"""
import threading
import time

import numpy as np
import requests
from absl import flags

from auto_eval.robot.policy import policies
from auto_eval.utils.info import print_red
from auto_eval.visualization.web_viewer import WEB_VIEWER_IP, WEB_VIEWER_PORT

FLAGS = flags.FLAGS

"""
Exceptions for robot status checks
"""
# failures
class RobotFailure(Exception):
    pass


class RobotTorqueFailure(RobotFailure):
    pass


# stuck
class RobotStuck(Exception):
    pass


"""
Getting human input to continue the operation
"""
# Global variable to store user input
user_input_received = False


def get_user_input():
    global user_input_received
    input("Press Enter to continue or type 'y' on slack to confirm: ")
    user_input_received = True


def update_web_viewer_status(robot_id=0, waiting_for_human=False):
    """
    Update the web viewer status to indicate if the robot is waiting for human intervention
    """
    try:
        status_url = (
            f"http://{WEB_VIEWER_IP}:{WEB_VIEWER_PORT}/update_status/{robot_id}"
        )
        status_data = {
            "waiting_for_human": waiting_for_human,
        }
        requests.post(status_url, json=status_data, timeout=1)
    except requests.RequestException as e:
        print(f"Error updating web viewer status: {e}")


def continue_after_confirmation(slack_bot, robot_id=0):
    """
    Check for a response in the slack channel or wait for user input.
    Returns True if 'y' is found in the slack response or if user inputs anything.
    """
    global user_input_received
    # Start a thread to get user input
    input_thread = threading.Thread(target=get_user_input)
    input_thread.start()

    # Update the web viewer to show we're waiting for human intervention
    update_web_viewer_status(robot_id, waiting_for_human=True)

    while True:
        response = slack_bot.check_for_response()
        if response:
            break
        if user_input_received:
            break
        time.sleep(1)  # Sleep briefly to avoid busy waiting
    print_red("User input received, continuing...")

    # Update the web viewer to show we're no longer waiting for human intervention
    update_web_viewer_status(robot_id, waiting_for_human=False)

    user_input_received = False  # reset the global variable


def check_robot_torque(info):
    """
    Check whether the torque is enabled, and re-enable them if not.

    When the robot fails because of excessive force, it will auto turn off all torques,
    and no further action commands will be executed if the torque is not turned back on.
    """
    res = info["torque_status"]
    if res is None or sum(res) < len(res):
        # some torques are not enabled
        raise RobotTorqueFailure


def check_robot_proprio_after_reset(obs, tol=0.2):
    """
    Call this function after an environment reset, to check that the proprio is
    indeed the actual reset proprio. This function only checks the xyz positions,
    and not the rotation and gripper position.
    If the robot failed / stuck somewhere, this check will fail.
    """
    current_proprio = obs["proprio"]
    ground_truth_proprio = np.array(
        [
            2.4830225e-01,
            0,
            1.8059616e-01,
            -3.0892963e00,
            1.5416106e00,
            -3.0891564e00,
            0.0000000e00,
            3.9364347e-01,
        ]
    )

    delta = np.linalg.norm(current_proprio[:3] - ground_truth_proprio[:3])
    if delta > tol:
        print_red("The robot proprio is not the expected reset proprio.")
        raise RobotFailure
    else:
        pass


def check_robot_proprio_for_conditions(obs, failure_conditions, stuck_conditions):
    """
    Call this function at the end of an episode, to check that whether the proprio
    satiesfies the certain conditions. When the proprio satisfies the
    failure_conditions (e.g. falling on the table), this will raise a RobotFailure.
    When the proprio satisfies the stuck_conditions (e.g. stuck in drawer handle),
    this will raise a RobotStuck.

    Args:
        obs: the observation dictionary
        failure_conditions: a list of failure conditions, and any of them will
            trigger the failure. Each failure condition is {x: condition, y: condition, z: condition}.
            E.g. [{"x": lambda x: x > 0.5, "y": lambda y: y > 0.5, "z": lambda z: z > 0.5}, ...]
        stuck_conditions: a list of stuck conditions, similar to failure_conditions.
    """
    current_proprio = obs["proprio"]
    # check for failure
    for cond in failure_conditions:
        if all(
            [cond[axis](current_proprio[i]) for i, axis in enumerate(["x", "y", "z"])]
        ):
            print_red("The robot proprio satisfies the failure conditions.")
            print_red(f"Current proprio: {current_proprio}")
            raise RobotFailure
    # check for stuck
    for cond in stuck_conditions:
        if all(
            [cond[axis](current_proprio[i]) for i, axis in enumerate(["x", "y", "z"])]
        ):
            print_red("The robot proprio satisfies the stuck conditions.")
            print_red(f"Current proprio: {current_proprio}")
            raise RobotStuck
    pass


def run_recovery_policy(env):
    if "recovery_policy_type" not in FLAGS.config:
        print("Robot is stuck but no recovery policy provided")
        return

    assert (
        FLAGS.config.recovery_policy_type == "scripted"
    ), "only scripted policy is supported for now"
    reset_policy = policies[FLAGS.config.recovery_policy_type](
        dict(FLAGS.config.recovery_policy_kwargs)
    )
    reset_policy.reset()
    for _ in range(FLAGS.config.recovery_policy_kwargs["recovery_steps"]):
        action = reset_policy(
            None, None
        )  # no obs/language needed for replaying scripted policy
        env.step(action)


def handle_robot_torque_failure(obs, info, manipulator_interface, slack_bot):
    """
    NOTE(zhouzypaul): this is currently deprecated

    Handle robot torque failure by notifying user, re-enabling torque,
    and rebooting motors if necessary.
    Returns True if torque was successfully re-enabled.
    """
    slack_bot.send(
        f"""The robot has some torque disabled: {info['torque_status']}.
        Please check the robot and place in a safe position.
        Afterwards the robot torque will be re-enabled automatically.""",
        image=obs["image_primary"],
    )
    continue_after_confirmation(slack_bot)

    # re-enable the torque
    manipulator_interface.custom_fn("enable_torque")
    # check that the torque is re-enabled, if not, reboot motors
    new_torque_status = manipulator_interface.custom_fn("get_torque_status")
    try:
        assert sum(new_torque_status) == len(
            new_torque_status
        )  # 0 is disabled, 1 is enabled
    except AssertionError:
        slack_bot.send("Failed to re-enable the torque. Need to reboot motors")
        motor_status = manipulator_interface.custom_fn("motor_status")
        assert sum(motor_status) > 0, motor_status  # motor must have failed

        manipulator_interface.custom_fn("safe_reboot_all_motors")
        new_torque_status = None
        while new_torque_status is None:
            # need to wait for motors to reboot
            new_torque_status = manipulator_interface.custom_fn("get_torque_status")
        assert sum(new_torque_status) == len(
            new_torque_status
        )  # 0 is disabled, 1 is enabled
    return True


def run_pre_rollout_checks(obs, info, manipulator_interface, slack_bot):
    """
    all checks to run at the beginning of a rollout episode
    returns whether the checks have failed.
    """
    try:
        check_robot_proprio_after_reset(obs)
        # check_robot_torque(info)  # moved to gym wrapper
        return 0

    except RobotTorqueFailure as e:
        # NOTE(zhouzypaul): this is currently deprecated
        handle_robot_torque_failure(obs, info, manipulator_interface, slack_bot)
        check_robot_proprio_after_reset(obs)  # re-run other checks
        return 1

    except RobotFailure as e:
        slack_bot.send(
            "The robot has failed pre-rollout check. Please check the robot status and reset it.",
            image=obs["image_primary"],
        )
        continue_after_confirmation(slack_bot)
        return 1


def run_post_rollout_checks(
    env,
    obs,
    info,
    manipulator_interface,
    failure_conditions=[],
    stuck_conditions=[],
    slack_bot=None,
):
    """
    all checks to run at the end of a rollout episode
    """
    try:
        check_robot_proprio_for_conditions(obs, failure_conditions, stuck_conditions)
        # check_robot_torque(info)  # moved to gym wrapper
        return 0

    except RobotTorqueFailure:
        # NOTE(zhouzypaul): this is currently deprecated
        handle_robot_torque_failure(obs, info, manipulator_interface, slack_bot)
        # re-run other checks
        check_robot_proprio_for_conditions(obs, failure_conditions, stuck_conditions)
        return 1

    except RobotFailure as e:
        slack_bot.send(
            "The robot has failed post-rollout check. Please check the robot status and reset it.",
            image=obs["image_primary"],
        )
        continue_after_confirmation(slack_bot)
        return 1
    except RobotStuck as e:
        print_red(
            "The robot is likely stuck. Attempting to run recovery policy to unstuck."
        )
        run_recovery_policy(env)
        return 0  # no human intervention needed, count as not failed
