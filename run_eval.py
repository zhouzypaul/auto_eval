"""
Main script for running the autonomous evaluation procedure
"""
import os
import signal
import tempfile
import threading

import numpy as np
import wandb
from absl import app, flags
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.manipulator_env import ManipulatorEnv, StateEncoding
from manipulator_gym.utils.gym_wrappers import (
    CheckAndRebootJoints,
    ClipActionBoxBoundary,
    ConvertState2Proprio,
    InHouseImpedanceControl,
    LimitMotorMaxEffort,
    ResizeObsImageWrapper,
)
from ml_collections import config_flags
from robot_eval_logger import (
    EvalLogger,
    FrameVisualizer,
    HuggingFaceStorage,
    LocalStorage,
    WandBLogger,
)

from auto_eval.robot.gym_wrappers import ClipActionMagnitude
from auto_eval.robot.policy import policies
from auto_eval.robot.policy_clients import policy_clients
from auto_eval.robot.robot_commands import move_eef_to_reset_position
from auto_eval.robot.robot_status_check import (
    continue_after_confirmation,
    run_post_rollout_checks,
    run_pre_rollout_checks,
)
from auto_eval.success_detector import detectors
from auto_eval.utils.info import print_obvious, print_red, print_yellow
from auto_eval.utils.slack_bot import DummyBot, SlackMessenger
from auto_eval.utils.timer_util import Timer
from auto_eval.visualization import stream_images, visualize_image
from auto_eval.web_ui.launcher import RobotIDs, RobotIPs

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "localhost", "IP address of the robot action server.")
flags.DEFINE_string("policy_server_ip", "localhost", "IP address of the policy server.")
flags.DEFINE_integer("policy_server_port", 8000, "Port of the policy server.")
flags.DEFINE_string(
    "visualization_method",
    "web_viewer",
    "How to visualize the current image status of the robot. Limited to None, display, and web_viewer.",
)
flags.DEFINE_string("text_cond", None, "Language prompt for the task.")

flags.DEFINE_integer("num_episodes", 60, "Number of episodes to evaluate.")
flags.DEFINE_integer("max_steps", 70, "Maximum number of steps per episode.")
flags.DEFINE_integer("max_reset_steps", 110, "Maximum number of steps per episode.")
flags.DEFINE_integer("max_reset_attempts", 3, "Maximum number of reset attempts.")
flags.DEFINE_integer("log_every_n_frames", 10, "Log every n frames.")
flags.DEFINE_bool(
    "redo_motor_failure",
    True,
    "Whether to redo the trajectories where the robot motor failed.",
)
flags.DEFINE_integer(
    "maximal_joint_effort",
    1000,
    "Maximum joint effort allowed, anything above this will be cut to null action",
)
flags.DEFINE_integer(
    "maximal_joint_effort_for_reset",
    None,
    "Optionally specify a different maximal joint effort for the reset policy. If not specified, will use the same value as maximal_joint_effort.",
)

flags.DEFINE_bool("debug", False, "Whether to debug or not.")
flags.DEFINE_string("exp_name", "", "Name of the experiment for wandb logging.")
flags.DEFINE_bool(
    "human_eval",
    False,
    "Whether to run human evaluation or not. If so, turn off success detection & reset policy.",
)
flags.DEFINE_bool(
    "save_classifier_data", False, "Whether to save classifier image input and output."
)
flags.DEFINE_bool(
    "log_eval_steps_per_min",
    False,
    "Wheter to call log_step() to log eval steps per min",
)
flags.DEFINE_bool(
    "always_execute_reset_policy", False, "Whether to always execute reset policy"
)
flags.DEFINE_bool("no_slack_bot", False, "Whether to disable the slack bot.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

# Global slack bot instance
slack_bot = None
# Global wandb logger instance
wandb_logger = None


def get_single_img(obs):
    img = obs["image_primary"]
    return img[-1] if img.ndim == 4 else img


def get_current_obs(obs):
    """in case that obs has history, only get the most current one"""
    current_obs = {}
    img = obs["image_primary"]
    proprio = obs["proprio"]
    if img.ndim == 4:
        current_obs["image_primary"] = img[-1]
    else:
        current_obs["image_primary"] = img
    if proprio.ndim == 2:
        current_obs["proprio"] = proprio[-1]
    else:
        current_obs["proprio"] = proprio
    return current_obs


"""
handle signals
"""
sigint_caught = False


def signal_handler(sig, frame):
    global sigint_caught
    print_red("Caught SIGINT (Control-C). Entering debugger...")
    sigint_caught = True


def enter_pdb_on_signal():
    global sigint_caught
    if sigint_caught:
        breakpoint()
        sigint_caught = False


signal.signal(signal.SIGINT, signal_handler)


def main(_):
    """
    Parse the arguments
    """
    global slack_bot, wandb_logger
    # Initialize slack bot if not already initialized
    if slack_bot is None:
        slack_bot = DummyBot() if FLAGS.no_slack_bot else SlackMessenger()

    if FLAGS.human_eval:
        print_red(
            "Running in human evaluation mode. Success detection and reset policy will be disabled."
            "Workspace bounds will be ignored."
        )
        FLAGS.config.success_detector_type = "none"
        FLAGS.config.workspace_bounds = None
        assert not FLAGS.always_execute_reset_policy
    if FLAGS.debug:
        FLAGS.save_classifier_data = False
    if FLAGS.visualization_method:
        assert FLAGS.visualization_method in ["display", "web_viewer", "none"]
    # whether go to sleep pose when rebooting motors
    if FLAGS.robot_ip in (RobotIPs.WIDOWX_DRAWER, RobotIPs.WIDOWX_CLOTH):
        reboot_with_sleep_pose = True
    elif FLAGS.robot_ip == RobotIPs.WIDOWX_SINK:
        reboot_with_sleep_pose = False
    else:
        raise ValueError("Unknown robot IP: ", FLAGS.robot_ip)

    # Get robot ID for web viewer
    robot_name = {
        RobotIPs.WIDOWX_DRAWER: "widowx_drawer",
        RobotIPs.WIDOWX_SINK: "widowx_sink",
        RobotIPs.WIDOWX_CLOTH: "widowx_cloth",
    }[FLAGS.robot_ip]
    robot_id = RobotIDs.get_id(robot_name)

    """
    load the policy to be evaled
    """
    if "client" in FLAGS.config.eval_policy_type:
        eval_policy = policy_clients[FLAGS.config.eval_policy_type](
            host=FLAGS.policy_server_ip, port=FLAGS.policy_server_port
        )
    else:
        eval_policy = policies[FLAGS.config.eval_policy_type](
            dict(FLAGS.config.eval_policy_kwargs)
        )

    # Create task specification
    if FLAGS.text_cond is None:
        FLAGS.text_cond = FLAGS.config.text_cond
    language_instruction = FLAGS.text_cond
    reset_language_instruction = FLAGS.config.reset_language_cond

    """
    set up logging
    """
    # wandb logger
    wandb_config = WandBLogger.get_default_config()
    exp_descriptor = f"{FLAGS.exp_name}_{FLAGS.config.eval_policy_type}"
    exp_descriptor += f"_{language_instruction.replace(' ', '_')}"
    wandb_config.update(
        {
            "exp_descriptor": exp_descriptor,
            "project": "auto_eval_human" if FLAGS.human_eval else "auto_eval_public",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=FLAGS.debug,
    )

    # frames visualizer
    frames_visualizer = FrameVisualizer(
        episode_viz_frame_interval=10,
        success_viz_every_n=FLAGS.log_every_n_frames,
        periodic_log_initial_and_final_frames=True,
    )

    hf_repo = "zhouzypaul/auto_eval"
    data_saver = HuggingFaceStorage(
        storage_dir=tempfile.gettempdir(),
        repo_id=hf_repo,
    )

    # create the eval logger
    eval_logger = EvalLogger(
        wandb_logger=wandb_logger,
        frames_visualizer=frames_visualizer,
        data_saver=data_saver,
    )

    if FLAGS.save_classifier_data:
        save_dir = os.path.join(
            os.path.expanduser("~"),
            "auto_eval_log",
            f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
        )
    else:
        save_dir = None

    # save metadata
    eval_logger.save_metadata(
        location="berkeley",
        robot_name="widowx_drawer",
        robot_type="widowx",
        evaluator_name="bridge_autoeval",
        eval_name=FLAGS.exp_name,
    )

    """
    Performance timer
    """
    timer = Timer()

    """
    create environment
    """
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_0": ...
    #     "image_1": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_0": ...
    #       "image_1": ...
    #     }
    #   }
    ##################################################################################################################
    manipulator_interface = ActionClientInterface(host=FLAGS.robot_ip)

    """
    Start visualization thread
    """
    vis_streamer = stream_images(
        manipulator_interface, FLAGS.visualization_method, robot_id
    )
    vis_thread = threading.Thread(
        target=vis_streamer.start,
        daemon=True,  # Thread will be terminated when main program exits
    )
    vis_thread.start()

    """
    Create environment
    """

    def _create_env(workspace_bounds):
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

    # create eval env & reset env, which may need different wrappers
    env = _create_env(FLAGS.config.workspace_bounds)
    env = InHouseImpedanceControl(
        env,
        max_effort_limit=FLAGS.maximal_joint_effort,
    )
    env = ClipActionMagnitude(env, max_magnitude=2)

    reset_env = _create_env(
        FLAGS.config.get("reset_workspace_bounds", FLAGS.config.workspace_bounds)
    )
    reset_env = (
        LimitMotorMaxEffort(  # don't do impedance because don't want to reverse action
            reset_env,
            max_effort_limit=FLAGS.maximal_joint_effort_for_reset
            or FLAGS.maximal_joint_effort,
        )
    )

    # add wrappers for history and "receding horizon control", i.e. action chunking
    if FLAGS.config.eval_policy_type == "octo":
        from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper

        env = HistoryWrapper(env, horizon=2)
        env = TemporalEnsembleWrapper(env, 4)
    elif FLAGS.config.eval_policy_type in ("pizero", "pi_zero_client"):
        from octo.utils.gym_wrappers import TemporalEnsembleWrapper

        env = TemporalEnsembleWrapper(env, 4)

    if FLAGS.config.reset_policy_type in ("pizero", "pi_zero_client"):
        from octo.utils.gym_wrappers import TemporalEnsembleWrapper

        reset_env = TemporalEnsembleWrapper(reset_env, 4)

    """
    success detection
    """
    if not FLAGS.human_eval:
        success_detector = detectors[FLAGS.config.success_detector_type](
            save_data=FLAGS.save_classifier_data,
            save_dir=save_dir,
            **FLAGS.config.success_detector_kwargs.vlm_config,
        )
    else:
        success_detector = lambda *args, **kwargs: True

    """
    reset policy
    """
    if not FLAGS.human_eval:
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

    """
    test action server
    """
    # sometimes after a long idle time the action server does not respond
    # on the first try (agentlace issues), need to try a second time to establish connection
    n_action_server_retries = 3
    dummy_action = np.zeros(7)
    for _ in range(n_action_server_retries):
        try:
            env.step(dummy_action)
            break
        except Exception as e:
            print("Failed to connect to action server, retrying...")
            pass

    """
    eval rollouts fn
    """

    def eval_rollout(log_step=True):
        obs, info = env.reset(moving_time=5)
        pre_check_failed = run_pre_rollout_checks(
            get_current_obs(obs), info, manipulator_interface, slack_bot
        )  # check robot is healthy
        frames_recorder = []
        actions_recorder = []
        proprio_recorder = []
        infos = [info]
        eval_policy.reset()

        for i in range(FLAGS.max_steps):
            enter_pdb_on_signal()
            # visualize
            img = get_single_img(obs)
            visualize_image(
                FLAGS.visualization_method,
                img,
                language_instruction,
                robot_id=robot_id,
                episode=i_episode,
                timestep=i,
            )

            actions = eval_policy(obs, language_instruction)
            print(f"Step {i} with action size of {len(actions)}")
            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            frames_recorder.append(get_single_img(obs))
            actions_recorder.append(actions)
            proprio_recorder.append(obs["proprio"])
            infos.append(info)
            if log_step:
                eval_logger.log_step()

            if done or trunc:
                # trunc is because of robot failure
                break

        # check robot is healthy
        enter_pdb_on_signal()
        post_check_failed = run_post_rollout_checks(
            reset_env,  # doesn't deal with action chunking
            get_current_obs(obs),
            info,
            manipulator_interface,
            FLAGS.config.failure_conditions,
            FLAGS.config.stuck_conditions,
            slack_bot,
        )

        # return whether rollout is successful without robot failure
        execution_successful = not trunc
        eval_len = i

        # flatten the info
        infos = {k: [info[k] for info in infos] for k in infos[0].keys()}
        infos["eval_len"] = eval_len
        infos["frames"] = frames_recorder
        infos["actions"] = actions_recorder
        infos["proprio"] = proprio_recorder
        infos["pre_check_failed"] = pre_check_failed
        infos["post_check_failed"] = post_check_failed

        return obs, infos, execution_successful

    """
    reset rollout fn
    """

    def reset_rollout(initial_success_detection=False):
        # success detection should happen after a reset, in case
        # that the reset accidentally mess up the scene
        obs, info = reset_env.reset(moving_time=5)
        reset_policy.reset()  # some policies need to reset their state
        frames_recorder = []

        # robot status check
        run_pre_rollout_checks(
            get_current_obs(obs), info, manipulator_interface, slack_bot
        )  # check robot is healthy

        if initial_success_detection:
            reset_successful = success_detector(
                FLAGS.config.success_detector_kwargs["vlm_question"],
                get_single_img(obs),
                answer=FLAGS.config.success_detector_kwargs[
                    "ground_truth_answer_reset_task"
                ],
            )

        # if no need for reset, skip
        if initial_success_detection and reset_successful:
            return frames_recorder, True

        # run the reset policy
        for i in range(FLAGS.max_reset_steps):
            # handle signal
            enter_pdb_on_signal()
            # visualize
            img = get_single_img(obs)
            visualize_image(
                FLAGS.visualization_method,
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

        # again, success detection should happen after reset
        obs, info = reset_env.reset(moving_time=5)
        reset_successful = success_detector(
            FLAGS.config.success_detector_kwargs["vlm_question"],
            get_single_img(obs),
            answer=FLAGS.config.success_detector_kwargs[
                "ground_truth_answer_reset_task"
            ],
        )

        return frames_recorder, reset_successful

    """
    ensure scene is reset at beginning of eval
    """
    if not FLAGS.human_eval:
        print_obvious("Check for scene reset at the beginning of eval")
        i_episode = 0  # needed for reset_rollout()
        for i_reset in range(FLAGS.max_reset_attempts):
            frames_recorder, reset_successful = reset_rollout(
                initial_success_detection=i_reset == 0
            )
            if reset_successful:
                print_red("Initial scene is reset. Starting main evals...")
                break

        # Add confirmation check if reset failed after max attempts
        if not reset_successful:
            slack_bot.send(
                f"❌ Reset policy failed on robot {FLAGS.robot_ip} for task {reset_language_instruction} after maximum attempts.",
                image=frames_recorder[-1],
            )
            continue_after_confirmation(slack_bot, robot_id)
    else:
        # human reset always successful
        reset_successful = True

    """
    run loop
    """

    print_obvious("Running Eval Rollouts")
    for i_episode in range(FLAGS.num_episodes):

        enter_pdb_on_signal()
        print_obvious(f"Eval episode {i_episode}")
        timer.tick("eval_rollout")
        obs, eval_infos, eval_without_robot_error = eval_rollout(
            log_step=FLAGS.log_eval_steps_per_min
        )
        experienced_motor_failure = False
        n_eval_retries = 0

        # whether need to re-run the eval
        while not eval_without_robot_error and FLAGS.redo_motor_failure:
            experienced_motor_failure = True

            # if task already successful, no need to re-run
            success = success_detector(
                FLAGS.config.success_detector_kwargs["vlm_question"],
                get_single_img(obs),
                answer=FLAGS.config.success_detector_kwargs[
                    "ground_truth_answer_eval_task"
                ],
            )
            if success:
                print_red(
                    "Motor failed but task already successful. No need to re-run."
                )
                break

            # else, re-run
            print_red("Re-running eval rollout due to robot error.")
            obs, eval_infos, eval_without_robot_error = eval_rollout(log_step=False)
            n_eval_retries += 1
        timer.tock("eval_rollout")

        # to make success detection easier (e.g. without occlusion)
        # some robots need to take certain actions
        if robot_name == "widowx_sink":
            move_eef_to_reset_position(manipulator_interface)

        # end of episode
        # success detection
        success = success_detector(
            FLAGS.config.success_detector_kwargs["vlm_question"],
            get_single_img(obs),
            answer=FLAGS.config.success_detector_kwargs[
                "ground_truth_answer_eval_task"
            ],
        )
        print_red(f"Episode {i_episode} completed with success: {success}")

        # logging
        joint_efforts = {
            k: [info[k] for info in eval_infos["joint_efforts"]]
            for k in eval_infos["joint_efforts"][0].keys()
        }  # flatten
        max_joint_efforts = tuple(
            np.max(joint_efforts[k]) for k in joint_efforts.keys()
        )
        eval_logger.log_episode(
            i_episode=i_episode,
            logging_prefix=language_instruction,
            episode_success=success,
            frames_to_log=eval_infos["frames"],
            actions=eval_infos["actions"],
            proprio=eval_infos["proprio"],
            eval_rollout_steps=eval_infos["eval_len"],
            eval_rollout_time=timer.get_times("eval_rollout"),
            max_joint_efforts=wandb.Histogram(max_joint_efforts),
            exceeded_joint_efforts=int(
                np.max(max_joint_efforts) > FLAGS.maximal_joint_effort
            ),
            experienced_motor_failure=int(experienced_motor_failure),
            pre_rollout_checks_failed=int(eval_infos["pre_check_failed"]),
            post_rollout_checks_failed=int(eval_infos["post_check_failed"]),
            need_human_intervention=int(
                eval_infos["pre_check_failed"]
                or eval_infos["post_check_failed"]
                or not reset_successful  # needed human reset
            ),
            num_eval_retries=n_eval_retries,
        )

        """
        run several reset attempts
        """
        if FLAGS.human_eval:
            reset_env.reset(moving_time=5)
            print_red("Press Enter to continue to next episode.")
            input()
        else:
            for i_reset in range(FLAGS.max_reset_attempts):
                print_obvious(f"Running Reset Attempt # {i_reset}")
                frames_recorder, reset_successful = reset_rollout(
                    initial_success_detection=(i_reset == 0)
                    and not FLAGS.always_execute_reset_policy
                )
                if reset_successful:
                    print_red("Reset successful! Continuing to next episode.")
                    break

            # Add confirmation check if reset failed after max attempts
            if not reset_successful:
                slack_bot.send(
                    f"❌ Reset policy failed on robot {FLAGS.robot_ip} for task {reset_language_instruction} after maximum attempts",
                    image=frames_recorder[-1],
                )
                continue_after_confirmation(slack_bot, robot_id)

    """
    Clean up resources
    """
    try:
        if "vis_streamer" in locals():
            vis_streamer.stop()
        env.close()
        reset_env.close()
    except Exception as e:
        print(f"Error during cleanup: {e}")
        pass

    """
    print out where the data is saved
    """
    print_red(
        f"Eval data saved to {eval_logger.data_saver.run_dir} on HuggingFace repo {hf_repo}"
    )


def run_with_error_handling(_):
    """Wrapper function that handles errors and notifications"""
    global slack_bot, wandb_logger
    try:
        main(_)
        if not FLAGS.debug:
            # Update wandb summary to ensure statistics are available
            slack_bot.send(
                f"✅ Evaluation {wandb_logger.run.name} on robot {FLAGS.robot_ip} completed successfully! Ran {FLAGS.num_episodes} episodes.\n"
                f"📊 View results at: {wandb_logger.run.get_url()}.\n"
                f"📈 Success rate: {wandb_logger.run.summary[f'{FLAGS.text_cond}/overall_success_rate']:.2f}%. Succeeded {wandb_logger.run.summary[f'{FLAGS.text_cond}/cumulative_num_success']} out of {FLAGS.num_episodes}. \n",
                get_response=False,
            )
    except Exception as e:
        import traceback

        if not FLAGS.debug:
            slack_bot.send(
                f"❌ Evaluation {wandb_logger.run.name} on robot {FLAGS.robot_ip} failed with error:\n```\n{traceback.format_exc()}\n```"
                f"📊 View results at: {wandb_logger.run.get_url()}.\n",
                get_response=False,
            )
        raise  # Re-raise the exception after sending notification
    finally:
        # Clean up wandb
        if wandb_logger and wandb_logger.run:
            wandb_logger.run.finish()


def run_with_results_returned(_):
    """Wrapper function that returns evaluation results in a json format
    This is used by the web UI, and doesn't send slack notifications
    """
    global wandb_logger
    result = {
        "wandb_url": None,
        "success_rate": None,
        "status": "failed",
    }
    try:
        main(_)
        if wandb_logger and wandb_logger.run:
            result["wandb_url"] = wandb_logger.run.get_url()
            result["success_rate"] = wandb_logger.run.summary[
                f"{FLAGS.text_cond}/overall_success_rate"
            ]
            result["status"] = "completed"
    except Exception as e:
        if wandb_logger and wandb_logger.run:
            result["wandb_url"] = wandb_logger.run.get_url()
        raise  # Re-raise the exception
    finally:
        # Clean up wandb
        if wandb_logger and wandb_logger.run:
            wandb_logger.run.finish()
    return result


if __name__ == "__main__":
    app.run(run_with_error_handling)
