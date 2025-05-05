"""
Robust Pi0 inference script with better error handling and image normalization.
This script addresses issues with the depth model and normalization.

Example usage:
    python simpler_env/pi0_robust_inference.py --policy pi0 --ckpt-path lerobot/pi0 --task widowx_carrot_on_plate --n-trajs 1
"""

import argparse
import os
import mediapy as media
import numpy as np
import torch

import simpler_env
from simpler_env import ENVIRONMENTS
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# ────────────────────────────────────────────────────────────────────────────────
# Arg‑parser
# ────────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="pi0", choices=["pi0"])
parser.add_argument("--ckpt-path", type=str, default="lerobot/pi0")
parser.add_argument("--task", default="widowx_carrot_on_plate", choices=ENVIRONMENTS)
parser.add_argument("--logging-root", type=str, default="./results_robust_eval")
parser.add_argument("--n-trajs", type=int, default=5)
parser.add_argument("--reduce-scale", type=float, default=0.5, 
                    help="Scale factor to reduce action magnitudes (0.5 = half as aggressive)")
args = parser.parse_args()

if args.ckpt_path.endswith("/"):
    args.ckpt_path = args.ckpt_path[:-1]

# Logging dir
logging_dir = os.path.join(args.logging_root, args.task,
                         "pi0_robust", os.path.basename(args.ckpt_path))
os.makedirs(logging_dir, exist_ok=True)

# GPU safety guards 
os.environ["DISPLAY"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Patch functions in pi0_model to make them more robust
from simpler_env.policies.pi0.pi0_model import Pi0Inference, CustomPI0Policy, resize_with_pad

# Create a modified inference wrapper with better error handling
class RobustPi0Inference(Pi0Inference):
    def __init__(self, model_name="lerobot/pi0", policy_setup="widowx_bridge", 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 scale_factor=0.5):
        super().__init__(model_name, policy_setup, device)
        self.scale_factor = scale_factor
        print(f"Initialized RobustPi0Inference with scale_factor={scale_factor}")
    
    def _whiten_image(self, img_t: torch.Tensor) -> torch.Tensor:
        # Convert image from [0, 255] or [0, 1] to [-1, 1] range
        # First ensure we're in [0, 1] range
        if img_t.max() > 1.0:
            img_t = img_t / 255.0
        
        # Then convert to [-1, 1] range as expected by SigLIP/PaliGemma
        img_t = img_t * 2.0 - 1.0
        
        return img_t
    
    def step(self, image: np.ndarray, instruction: str | None = None,
             state: np.ndarray | None = None):
        if instruction:
            self.instruction = instruction

        img_t = torch.as_tensor(image).permute(2, 0, 1).float().to(self.device)  # (3,H,W)
        img_t = img_t.unsqueeze(0)                                               # (1,3,H,W)
        img_t = resize_with_pad(img_t, 224, 224, pad_value=0)
        img_t = self._whiten_image(img_t)

        if state is None:
            state = np.zeros(8, dtype=np.float32)
        mean = self._OBS_MEAN.numpy()
        std = self._OBS_STD.numpy()
        safe_std = np.where(std == 0.0, 1.0, std)
        st_norm = (state - mean) / safe_std
        st_norm[std == 0.0] = 0.0
        st_t = torch.as_tensor(st_norm, device=self.device).unsqueeze(0)
        
        batch = {
            "observation.image.top": img_t,
            "observation.state": st_t,
            "task": [self.instruction or ""],
        }

        print("IMG tensor dtype/min/max:", img_t.dtype, img_t.min().item(), img_t.max().item())
        print("STATE tensor mean/std:", st_t.mean().item(), st_t.std().item())
        
        # Use safe execution with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:    
                with torch.no_grad():
                    a_norm = self.policy.select_action(batch)[0].cpu().numpy()
                break
            except Exception as e:
                print(f"Error in attempt {attempt+1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    # On final attempt, fall back to a safe default action
                    print("Using fallback default action")
                    a_norm = np.zeros(7, dtype=np.float32)
                    a_norm[6] = 0.5  # Neutral gripper position
        
        print("RAW a_norm:", a_norm, "min/max:", a_norm.min(), a_norm.max())
        a_unnorm = a_norm * self._ACT_STD.numpy() + self._ACT_MEAN.numpy()
        
        # Scale down action magnitudes to produce smoother movements
        translation_scale = self.ACTION_TRANSLATION_SCALE * self.scale_factor
        rotation_scale = self.ACTION_ROTATION_SCALE * self.scale_factor
        
        world = a_unnorm[:3] * translation_scale
        rot = a_unnorm[3:6] * rotation_scale
        grip = a_unnorm[6:7]

        return {ACTION: a_unnorm}, {
            "world_vector": world,
            "rot_axangle": rot,
            "gripper": grip,
            "terminate_episode": np.array([False]),
        }

# ────────────────────────────────────────────────────────────────────────────────
# Environment + policy setup
# ────────────────────────────────────────────────────────────────────────────────
env = simpler_env.make(args.task)
policy_setup = "google_robot" if "google_robot" in args.task else "widowx_bridge"

env._max_episode_steps = 400
model = RobustPi0Inference(model_name=args.ckpt_path, policy_setup=policy_setup, scale_factor=args.reduce_scale)

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
        try:
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
        except Exception as e:
            print(f"Error during step {timestep}: {e}")
            # Take a safe neutral action
            neutral_action = np.zeros(7, dtype=np.float32) 
            neutral_action[6] = 0.5  # neutral gripper
            obs, _, success, truncated, info = env.step(neutral_action)

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
    media.write_video(f"{logging_dir}/episode_{ep_id}_success_{success}.gif", images, fps=25, codec="gif")
print("**Overall Success**", np.mean(success_arr), f"({np.sum(success_arr)}/{len(success_arr)})") 