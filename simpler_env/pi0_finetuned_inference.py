"""
Simple script for real‑to‑sim eval with ManiSkill2 pre‑packaged visual‑matching envs.
Now supports RT‑1, Octo‑{base,small}, **and Pi0**.

Example:
    cd {path_to_simpler_env_repo_root}
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py \
        --policy pi0 \
        --ckpt-path lerobot/pi0 \
        --task google_robot_pick_coke_can \
        --logging-root ./results_simple_eval/ \
        --n-trajs 10
        
    python simpler_env/pi0_finetuned_inference.py --policy pi0 --ckpt-path lerobot/pi0 --task widowx_carrot_on_plate --n-trajs 1
    python simpler_env/pi0_finetuned_inference.py --policy pi0 --ckpt-path /mnt/pentagon/rutu/lerobot/outputs/train/2025-04-29/11-48-55_pi0/checkpoints/last/pretrained_model --task widowx_carrot_on_plate --n-trajs 1
"""

import argparse
import os
import mediapy as media
import numpy as np

import simpler_env
from simpler_env import ENVIRONMENTS
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# ────────────────────────────────────────────────────────────────────────────────
# Arg‑parser
# ────────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="rt1",
                    choices=["rt1", "octo-base", "octo-small", "pi0"])
parser.add_argument("--ckpt-path", type=str,
                    default="./checkpoints/rt_1_x_tf_trained_for_002272480_step/")
parser.add_argument("--task", default="google_robot_pick_horizontal_coke_can",
                    choices=ENVIRONMENTS)
parser.add_argument("--logging-root", type=str,
                    default="./results_simple_random_eval")
parser.add_argument("--tf-memory-limit", type=int, default=3072)
parser.add_argument("--n-trajs", type=int, default=10)
args = parser.parse_args()

# Auto‑map ckpt‑path for Octo / Pi0 presets
if args.policy in ["octo-base", "octo-small", "pi0"]:
    if args.ckpt_path in [None, "None"] or "rt_1_x" in args.ckpt_path:
        args.ckpt_path = args.policy  # will be interpreted inside each wrapper
if args.ckpt_path.endswith("/"):
    args.ckpt_path = args.ckpt_path[:-1]

# Logging dir
logging_dir = os.path.join(args.logging_root, args.task,
                           args.policy, os.path.basename(args.ckpt_path))
os.makedirs(logging_dir, exist_ok=True)

# GPU safety guards 
os.environ["DISPLAY"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# For TensorFlow models
if args.policy in ["rt1"]:
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            tf.config.set_logical_device_configuration(
                tf.config.list_physical_devices("GPU")[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
            )
    except ImportError:
        print("TensorFlow not available - RT1 policy will not work")

# ────────────────────────────────────────────────────────────────────────────────
# Environment + policy setup
# ────────────────────────────────────────────────────────────────────────────────
env = simpler_env.make(args.task)
policy_setup = "google_robot" if "google_robot" in args.task else "widowx_bridge"

env._max_episode_steps = 400

if args.policy == "rt1":
    from simpler_env.policies.rt1.rt1_model import RT1Inference
    model = RT1Inference(saved_model_path=args.ckpt_path, policy_setup=policy_setup)
elif "octo" in args.policy:
    from simpler_env.policies.octo.octo_model import OctoInference
    model = OctoInference(model_type=args.ckpt_path, policy_setup=policy_setup, init_rng=0)
elif args.policy == "pi0":
    from simpler_env.policies.pi0.pi0_model import Pi0Inference
    model = Pi0Inference(model_name=args.ckpt_path, policy_setup=policy_setup)
else:
    raise NotImplementedError()

# ────────────────────────────────────────────────────────────────────────────────
# Inference loop
# ────────────────────────────────────────────────────────────────────────────────
success_arr = []
for ep_id in range(args.n_trajs):
    obs, _ = env.reset()
    instruction = env.get_language_instruction()
    is_final_subtask = env.is_final_subtask()

    model.reset(instruction)
    print("Instruction:", instruction)

    image = get_image_from_maniskill2_obs_dict(env, obs)
    images = [image]
    terminated, success, truncated = False, False, False
    timestep = 0

    while not (terminated or truncated):
        raw_action, action = model.step(image)
        terminated = bool(action["terminate_episode"][0] > 0)
        if terminated and not is_final_subtask:
            terminated = False
            env.advance_to_next_subtask()

        obs, _, success, truncated, info = env.step(
            np.concatenate([action["world_vector"],
                            action["rot_axangle"],
                            action["gripper"]]),
        )
        print(timestep, info)

        # Handle possible new sub‑instruction
        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction
            model.reset(instruction)
            print("New instruction:", instruction)
        is_final_subtask = env.is_final_subtask()

        # Update image
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
        timestep += 1

    success_arr.append(success)
    print(f"Episode {ep_id} success: {success}")
    # media.write_video(f"{logging_dir}/episode_{ep_id}_success_{success}.mp4", images, fps=5, codec="libx264", bps="2M")
    media.write_video(f"{logging_dir}/episode_{ep_id}_success_{success}.gif", images, fps=25, codec="gif")
print("**Overall Success**", np.mean(success_arr), f"({np.sum(success_arr)}/{len(success_arr)})")