import math
import time

import numpy as np
from manipulator_gym.interfaces.interface_service import ActionClientInterface

from auto_eval.web_ui.launcher import RobotIPs


def move_eef_to_reset_position(manipulator_interface):
    """
    move the eef to the reset position, but don't change the gripper
    """
    default_reset_pos = np.array([0.258325, 0, 0.19065, 0, math.pi / 2, 0])
    manipulator_interface.move_eef(default_reset_pos)


def create_interface(robot_ip):
    return ActionClientInterface(host=robot_ip)


def sleep_and_torque_off(interface):
    """
    go to sleep pose and turn off torque
    """

    # go to sleep pose
    kwargs = {"go_sleep": True}
    interface.reset(**kwargs)
    time.sleep(1)

    # turn off torque
    interface.custom_fn("enable_torque", enable=False)


def torque_on(interface):
    """
    turn torque back on
    """

    # torque on
    interface.custom_fn("enable_torque")
    interface.reset()  # go to neutral
    time.sleep(5)  # wait


if __name__ == "__main__":
    # for testing this script
    IP = RobotIPs.WIDOWX_DRAWER
    interface = create_interface(IP)
    sleep_and_torque_off(interface)
    time.sleep(7)
    interface = create_interface(IP)
    torque_on(interface)
