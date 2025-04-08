import os

from ml_collections import ConfigDict

# BridgeDatav2 metadata (gotten from the greats who computed by hand in the past, not by Octo dataloader)

ACT_MEAN = [
    1.9296819e-04,
    1.3667766e-04,
    -1.4583133e-04,
    -1.8390431e-04,
    -3.0808983e-04,
    2.7425270e-04,
    5.9716219e-01,
]

ACT_STD = [
    0.00912848,
    0.0127196,
    0.01229497,
    0.02606696,
    0.02875283,
    0.07807977,
    0.48710242,
]

ACT_MIN = [
    -0.0437546,
    -0.052831028,
    -0.035931006,
    -0.14489305,
    -0.15591072,
    -0.26039174,
    -0.780331,
]  # 0.1% quantile

ACT_MAX = [
    0.04158026,
    0.05223833,
    0.05382493,
    0.15559858,
    0.142592,
    0.25956747,
    0.79311615,
]  # 99.9% quantile


REPO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
)


"""
Base policies
"""

jaxrl_gc_policy_kwargs = dict(
    encoder="resnetv1-34",
    encoder_kwargs=dict(
        pooling_method="avg",
        add_spatial_coordinates=False,
        act="swish",
    ),
    obs_image_size=256,
    seed=42,
    policy_class="gc_bc",
    checkpoint_path=os.path.expanduser(
        "~/checkpoints/soar/gcbc-bridge/checkpoint_75000"
    ),
    agent_kwargs=dict(
        network_kwargs=dict(
            hidden_dims=(256, 256, 256),
            dropout_rate=0.1,
        ),
        policy_kwargs=dict(
            tanh_squash_distribution=False,
            std_parameterization="fixed",
            fixed_std=[1, 1, 1, 1, 1, 1, 0.1],
        ),
        early_goal_concat=True,
        shared_goal_encoder=True,
        use_proprio=False,
        learning_rate=3e-4,
        warmup_steps=2000,
        decay_steps=int(2e6),
    ),
    ACT_MEAN=ACT_MEAN,
    ACT_STD=ACT_STD,
    action_normalization_type="normal",  # Use normal for non-PiZero policies
)

soar_policy_kwargs = dict(
    **jaxrl_gc_policy_kwargs,
    susie_kwargs=dict(
        diffusion_checkpoint="kvablack/susie",
        diffusion_wandb="kvablack/dlimp-diffusion/9n9ped8m",
        diffusion_num_steps=50,
        prompt_w=7.5,
        context_w=4.0,
        diffusion_pretrained_path="lodestones/stable-diffusion-v1-5-flax",
        image_size=256,
    ),
    show_goal_image=False,
)

pizero_policy_kwargs = dict(
    action_normalization_type="bounds",
)

octo_policy_kwargs = dict()

open_web_policy_kwargs = dict()


"""
Finetuned policies
"""

openvla_policy_kwargs = dict()
finetuned_openvla_open_drawer_kwargs = dict(
    **openvla_policy_kwargs,
    lora_adapter_dir=os.path.expanduser(
        "~/checkpoints/auto-eval-openvla-drawer-combined/adapter_checkpoints/openvla-7b+expert_demos+b16+lr-0.0005+lora-r64+dropout-0.0"
    ),
    dataset_stats_path=os.path.expanduser(
        "~/checkpoints/auto-eval-openvla-drawer-combined/dataset_statistics.json"
    ),
)
finetuned_openvla_sink_to_basket_kwargs = dict(
    **openvla_policy_kwargs,
    lora_adapter_dir=os.path.expanduser(
        "~/checkpoints/auto-eval-openvla-blue-sink/adapter_checkpoints/openvla-7b+to_basket+b16+lr-0.0005+lora-r64+dropout-0.0-first-100"
    ),
    dataset_stats_path=os.path.expanduser(
        "~/checkpoints/auto-eval-openvla-blue-sink/100-dataset_statistics.json"
    ),
)
finetuned_openvla_basket_to_sink_kwargs = dict(
    **openvla_policy_kwargs,
    lora_adapter_dir=os.path.expanduser(
        "./checkpoints/auto-eval-openvla-blue-sink/adapter_checkpoints/to-sink-checkpoint"
    ),
    dataset_stats_path=os.path.expanduser(
        "./checkpoints/auto-eval-openvla-blue-sink/dataset_statistics/to_sink_dataset_statistics.json"
    ),
)
finetuned_minivla_to_basket_kwargs = dict(
    host="localhost",
    port=8004,
)
finetuned_minivla_to_sink_kwargs = dict(
    host="localhost",
    port=8003,
)

"""
Scripted Policies
"""

scripted_close_drawer_policy_kwargs = dict(
    policy_save_path=os.path.join(
        REPO_DIR, "scripted_policies", "close the drawer.pkl"
    ),
)

scripted_open_drawer_policy_kwargs = dict(
    policy_save_paths=[
        os.path.join(REPO_DIR, "scripted_policies", "close the drawer.pkl"),
        os.path.join(REPO_DIR, "scripted_policies", "open the drawer.pkl"),
    ],
    env_reset_in_between=True,
)

scripted_out_of_drawer_handle_policy_kwargs = dict(
    policy_save_path=os.path.join(
        REPO_DIR, "scripted_policies", "out of drawer handle.pkl"
    ),
    recovery_steps=30,
)

scripted_unfold_cloth_policy_kwargs = dict(
    policy_save_path=os.path.join(REPO_DIR, "scripted_policies", "unfold cloth.pkl"),
)


"""
Env settings
"""
drawer_common_kwargs = dict(
    eval_policy_type="open_web_client",
    eval_policy_kwargs=open_web_policy_kwargs,
    # eval_policy_type="octo",
    # eval_policy_kwargs=octo_policy_kwargs,
    # eval_policy_type="pi_zero_client",
    # eval_policy_kwargs=pizero_policy_kwargs,
    # eval_policy_type="openvla_client",
    # eval_policy_kwargs=openvla_policy_kwargs,
    # eval_policy_type="jaxrl_gc_policy",
    # eval_policy_kwargs=jaxrl_gc_policy_kwargs,
    # eval_policy_type="soar",
    # eval_policy_kwargs=soar_policy_kwargs,
    recovery_policy_type="scripted",
    recovery_policy_kwargs=scripted_out_of_drawer_handle_policy_kwargs,
    workspace_bounds=dict(
        x=[0.12602980432553548, 0.3635847125408836],
        y=[-0.16452473627781203, 0.07700634681364794],
        z=[0.02160514560120032, 0.163698249549381],
    ),
    # x is towards front wall, y is towards left wall, z is up
    failure_conditions=[
        {
            "x": lambda x: x >= 0.43,
            "y": lambda y: True,
            "z": lambda z: z <= 0.03,
        },  # robot pushing the micromove and falling
        {
            "x": lambda x: True,
            "y": lambda y: True,
            "z": lambda z: z <= 0,
        },  # robot falling on the table somewhere
        # {
        #     "x": lambda x: x >= 0.382,
        #     "y": lambda y: y >= 0.01,
        #     "z": lambda z: z <= 0.07,
        # },  # robot arm stuck behind drawer handle so it's hard to get back to reset
    ],
    stuck_conditions=[
        # {
        #     "x": lambda x: 0.27 <= x <= 0.3,
        #     "y": lambda y: -0.05 <= y <= 0.05,
        #     "z": lambda z: 0.02 <= z <= 0.063,
        # },  # handle in drawer handle
    ],
)


drawer_detector_common_kwargs = dict(
    vlm_type="paligemma",
    vlm_config={
        "model_id": os.path.expanduser(
            "~/checkpoints/auto-eval-paligemma/drawer-checkpoint-600"
        ),
        "device": "cuda:0",
        "quantize": True,
    },
    vlm_question="is the drawer open? answer yes or no",
)


sink_common_kwargs = dict(
    workspace_bounds=dict(
        x=[0.19978141088404, 0.36472741193260866],
        y=[-0.09850688045208801, 0.1949976390571117],
        z=[0.08706181213354207, 0.23396981684648352],
    ),
    failure_conditions=[],
    stuck_conditions=[],
)


sink_detector_common_kwargs = dict(
    vlm_type="paligemma",
    vlm_config={
        "model_id": os.path.expanduser(
            "~/checkpoints/auto-eval-paligemma/sink-checkpoint-880"
        ),
        "device": "cuda:0",
        "quantize": True,
    },
    vlm_question="is the eggplant in the sink or in the basket? answer sink or basket or invalid",
)


"""
Task settings
"""


def get_config(config_string):
    possible_structures = {
        "open_drawer": ConfigDict(
            dict(
                text_cond="open the drawer",
                reset_language_cond="close the drawer",
                success_detector_type="paligemma",
                success_detector_kwargs=dict(
                    **drawer_detector_common_kwargs,
                    ground_truth_answer_eval_task="yes",
                    ground_truth_answer_reset_task="no",
                ),
                reset_policy_type="scripted",
                reset_policy_kwargs=scripted_close_drawer_policy_kwargs,
                **drawer_common_kwargs,
            )
        ),
        "close_drawer": ConfigDict(
            dict(
                text_cond="close the drawer",
                reset_language_cond="open the drawer",
                success_detector_type="paligemma",
                success_detector_kwargs=dict(
                    **drawer_detector_common_kwargs,
                    ground_truth_answer_eval_task="no",
                    ground_truth_answer_reset_task="yes",
                ),
                reset_policy_type="scripted_sequence",
                reset_policy_kwargs=scripted_open_drawer_policy_kwargs,
                reset_workspace_bounds=dict(
                    x=[0.12, float("inf")],  # edge of table
                    y=[-float("inf"), 0.05],  # don't go over the drawer
                    z=[0.042, float("inf")],  # don't push down on handle
                ),
                **drawer_common_kwargs,
            )
        ),
        "put_eggplant_in_sink": ConfigDict(
            dict(
                text_cond="put the eggplant in the blue sink",
                reset_language_cond="put the eggplant in the yellow basket",
                success_detector_type="paligemma",
                success_detector_kwargs=dict(
                    **sink_detector_common_kwargs,
                    ground_truth_answer_eval_task="sink",
                    ground_truth_answer_reset_task="basket",
                ),
                eval_policy_type="open_web_client",
                eval_policy_kwargs=open_web_policy_kwargs,
                # eval_policy_type="pi_zero_client",
                # eval_policy_kwargs=pize   ro_policy_kwargs,
                # eval_policy_type="minivla_client",
                # eval_policy_kwargs=openvla_policy_kwargs,
                # eval_policy_type="octo",
                # eval_policy_kwargs=octo_policy_kwargs,
                # eval_policy_type="jaxrl_gc_policy",
                # eval_policy_kwargs=jaxrl_gc_policy_kwargs,
                # eval_policy_type="soar",
                # eval_policy_kwargs=soar_policy_kwargs,
                reset_policy_type="open_web_client",
                reset_policy_kwargs=dict(
                    **finetuned_minivla_to_basket_kwargs,
                ),
                **sink_common_kwargs,
            )
        ),
        "put_eggplant_in_basket": ConfigDict(
            dict(
                text_cond="put the eggplant in the yellow basket",
                reset_language_cond="put the eggplant in the blue sink",
                success_detector_type="paligemma",
                success_detector_kwargs=dict(
                    **sink_detector_common_kwargs,
                    ground_truth_answer_eval_task="basket",
                    ground_truth_answer_reset_task="sink",
                ),
                eval_policy_type="open_web_client",
                eval_policy_kwargs=open_web_policy_kwargs,
                # eval_policy_type="pi_zero_client",
                # eval_policy_kwargs=pizero_policy_kwargs,
                # eval_policy_type="openvla",
                # eval_policy_kwargs=openvla_policy_kwargs,
                # eval_policy_type="pizero",
                # eval_policy_kwargs=pizero_policy_kwargs,
                # eval_policy_type="octo",
                # eval_policy_kwargs=octo_policy_kwargs,
                # eval_policy_type="jaxrl_gc_policy",
                # eval_policy_kwargs=jaxrl_gc_policy_kwargs,
                # eval_policy_type="soar",
                # eval_policy_kwargs=soar_policy_kwargs,
                reset_policy_type="open_web_client",
                reset_policy_kwargs=dict(
                    **finetuned_minivla_to_sink_kwargs,
                ),
                **sink_common_kwargs,
            )
        ),
        "fold_cloth": ConfigDict(
            dict(
                text_cond="fold the cloth from top right to bottom left",
                reset_language_cond="unfold the cloth",
                success_detector_type="paligemma",
                success_detector_kwargs=dict(
                    vlm_type="paligemma",
                    vlm_config={
                        "model_id": os.path.expanduser(
                            "./checkpoints/paligemma_checkpoints/paligemma-cloth-checkpoint"
                        ),
                        "device": "cuda:0",
                        "quantize": True,
                    },
                    vlm_question="is the cloth folded or unfolded? answer yes or no",
                    ground_truth_answer_eval_task="yes",
                    ground_truth_answer_reset_task="no",
                ),
                eval_policy_type="open_web_client",
                eval_policy_kwargs=open_web_policy_kwargs,
                # eval_policy_type="octo",
                # eval_policy_kwargs=octo_policy_kwargs,
                # eval_policy_type="soar",
                # eval_policy_kwargs=soar_policy_kwargs,
                # eval_policy_type="jaxrl_gc_policy",
                # eval_policy_kwargs=jaxrl_gc_policy_kwargs,
                # eval_policy_type="pi_zero_client",
                # eval_policy_kwargs=pizero_policy_kwargs,
                # eval_policy_type="openvla_client",
                # eval_policy_kwargs=openvla_policy_kwargs,
                reset_policy_type="scripted",
                reset_policy_kwargs=scripted_unfold_cloth_policy_kwargs,
                workspace_bounds=dict(
                    x=[0.16197588264170293, 0.40134804201187707],
                    y=[-0.14773004431891934, 0.2673874628958015],
                    z=[0.020418038958272246, 0.1786604848567076],
                ),
                failure_conditions=[],
                stuck_conditions=[],
            ),
        ),
    }

    return possible_structures[config_string]
