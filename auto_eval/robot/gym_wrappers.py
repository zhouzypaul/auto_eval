import gym
import numpy as np

from auto_eval.utils.info import print_yellow


class ClipActionMagnitude(gym.Wrapper):
    """
    For operation safety, clip the action magnitudes that exceeds the limits.
    """

    def __init__(self, env, max_magnitude):
        super().__init__(env)
        self.max_magnitude = max_magnitude

    def step(self, action):
        if np.max(np.abs(action)) > self.max_magnitude:
            print_yellow(
                f"Action magnitude max {np.max(np.abs(action))} exceeds threshold, clipping to {self.max_magnitude}"
            )
        action = np.clip(action, -self.max_magnitude, self.max_magnitude)
        return super().step(action)
