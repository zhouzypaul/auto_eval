import os
import sys
import traceback

from absl import app, flags

"""
Meta info for launching AutoEval jobs
"""

# Map of robot names to their action server IPs
# These are the IP address of the machine that runs the robot environment server via manipulator_gym
class RobotIPs(str):
    WIDOWX_DRAWER = os.environ["WIDOWX_DRAWER_IP"]
    WIDOWX_SINK = os.environ["WIDOWX_SINK_IP"]
    WIDOWX_CLOTH = os.environ["WIDOWX_CLOTH_IP"]


# Map of robot names to their web viewer IDs
class RobotIDs:
    ROBOT_ID_MAP = {
        "widowx_drawer": 0,
        "widowx_sink": 1,
        "widowx_cloth": 2,
    }

    @classmethod
    def get_id(cls, robot_name: str) -> int:
        if robot_name not in cls.ROBOT_ID_MAP:
            raise ValueError(f"Unknown robot name: {robot_name}")
        return cls.ROBOT_ID_MAP[robot_name]


# Map of tasks to their eval config paths
class TaskConfigs:
    WIDOWX_DRAWER = {
        "open_drawer": "scripts/configs/eval_config.py:open_drawer",
        "close_drawer": "scripts/configs/eval_config.py:close_drawer",
    }
    WIDOWX_SINK = {
        "put_eggplant_blue_sink": "scripts/configs/eval_config.py:put_eggplant_in_sink",
        "put_eggplant_yellow_basket": "scripts/configs/eval_config.py:put_eggplant_in_basket",
    }
    # not putting WIDOWX_CLOTH because it's not public

    @classmethod
    def get_config(cls, robot_name):
        return getattr(cls, robot_name.upper(), None)


# Map of task to the reset max steps
class ResetMaxSteps:
    WIDOWX_DRAWER = 110
    WIDOWX_SINK = 80

    @classmethod
    def get(cls, robot_name):
        return getattr(cls, robot_name.upper(), 100)


# Map of task to maximal joint effort
class MaximalJointEffort:
    WIDOWX_DRAWER = 700
    WIDOWX_SINK = 700

    @classmethod
    def get(cls, robot_name):
        return getattr(cls, robot_name.upper(), 700)


# Map of task to maximal joint effort for reset
class MaximalJointEffortForReset:
    WIDOWX_DRAWER = 1500
    WIDOWX_SINK = 700

    @classmethod
    def get(cls, robot_name):
        return getattr(cls, robot_name.upper(), 700)


class Result(str):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


"""
Class for launching the AutoEval Python process
"""


class Launcher:
    def __init__(self, config):
        self.config = config

        # Get the robot IP based on the robot name
        robot_name = self.config["robot"]
        if robot_name == "widowx_drawer":
            self.config["robot_ip"] = RobotIPs.WIDOWX_DRAWER
        elif robot_name == "widowx_sink":
            self.config["robot_ip"] = RobotIPs.WIDOWX_SINK
        else:
            raise ValueError(
                f"Unknown robot name: {robot_name}. Must be one of ['widowx_drawer', 'widowx_sink']"
            )

        # Get the robot ID for web viewer
        self.config["robot_id"] = RobotIDs.get_id(robot_name)

        # Get the eval config path based on the robot and task
        task = self.config["task"]
        robot_configs = TaskConfigs.get_config(robot_name)
        if robot_configs is None or task not in robot_configs:
            raise ValueError(f"Invalid task '{task}' for robot '{robot_name}'")

        self.config["eval_config_path"] = robot_configs[task]

        # Get the reset max steps based on the robot and task
        self.config["max_reset_steps"] = ResetMaxSteps.get(robot_name)

        # Get the maximal joint effort based on the robot and task
        self.config["maximal_joint_effort"] = MaximalJointEffort.get(robot_name)
        self.config["maximal_joint_effort_for_reset"] = MaximalJointEffortForReset.get(
            robot_name
        )

    def run(self):
        print("Running job")

        # Set required environment variables
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        # Save original argv
        original_argv = sys.argv[:]

        try:
            # Set argv[0] to the run_eval.py script path and add our arguments
            run_eval_script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "run_eval.py"
            )
            print("Original argv:", sys.argv)
            sys.argv = [run_eval_script]  # Set the correct script path
            print("After setting script path:", sys.argv)
            sys.argv.extend(
                [
                    "--robot_ip",
                    self.config["robot_ip"],
                    "--policy_server_ip",
                    self.config["policy_server_ip"],
                    "--policy_server_port",
                    str(self.config["policy_server_port"]),
                    "--config",
                    self.config["eval_config_path"],
                    "--num_episodes",
                    str(self.config["num_episodes"]),
                    "--max_steps",
                    str(self.config["max_steps"]),
                    "--max_reset_steps",
                    str(self.config["max_reset_steps"]),
                    "--maximal_joint_effort",
                    str(self.config["maximal_joint_effort"]),
                    "--exp_name",
                    str(self.config["description"]),
                ]
            )

            # Add the always_execute_reset_policy option for widowx_drawer robot
            if self.config["robot"] == "widowx_drawer":
                sys.argv.extend(["--always_execute_reset_policy"])

            # Mark flags as parsed to avoid duplicate parsing
            from run_eval import FLAGS

            flags.FLAGS.mark_as_parsed()

            # Run the evaluation
            eval_result_store = {
                "result": None
            }  # need to store this to be accessible outside of the closure (app.run)

            def run_eval(_):
                from run_eval import run_with_results_returned

                eval_result = run_with_results_returned(_)
                eval_result_store["result"] = eval_result
                return eval_result

            try:
                app.run(run_eval)
            except SystemExit as e:
                # After app.run completes, we can access the result from our container
                eval_result = eval_result_store["result"]
                if eval_result is None:
                    raise Exception("Evaluation failed to return a result")

                print("Evaluation completed successfully")
                return {
                    "status": eval_result["status"],
                    "wandb_url": eval_result["wandb_url"],
                    "success_rate": eval_result["success_rate"],
                }

        except Exception as e:
            print(f"Evaluation failed with error:")
            traceback.print_exc()
            return {"status": Result.FAILED, "wandb_url": None, "success_rate": None}
        finally:
            # Restore original argv
            sys.argv = original_argv
