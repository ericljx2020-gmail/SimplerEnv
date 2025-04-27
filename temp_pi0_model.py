# simpler_env/policies/pi0/pi0_model.py

"""Lightweight wrapper that adapts the Hugging‑Face Pi0 pipeline to the
SimplerEnv evaluation API (reset → step)."""

import numpy as np
import torch

class Pi0Inference:
    def __init__(self,
                 model_name: str = "lerobot/pi0",
                 policy_setup: str = "google_robot"):
        """
        Load the Pi0 policy from Hugging Face.

        Args:
            model_name: HF model identifier (e.g., "lerobot/pi0").
            policy_setup: environment setup string ("google_robot" or "widowx_bridge").
        """
        print("Using random policy instead of Pi0 for testing")
        self.policy_setup = policy_setup
        self.instruction = None

    def reset(self, instruction: str) -> None:
        """Start a new episode with a (sub‑)task instruction."""
        self.instruction = instruction

    def step(self, image: np.ndarray, instruction: str = None):
        """
        Run Pi0 on the given image and instruction.

        Args:
            image: np.ndarray, shape (H, W, 3), dtype uint8
            instruction: str, task description (optional, will use stored instruction if None)

        Returns:
            raw_action: dict with model outputs
            action: dict with keys:
                - world_vector: np.ndarray (3,), translation
                - rot_axangle: np.ndarray (3,), axis-angle rotation
                - gripper: np.ndarray (1,), gripper command
                - terminate_episode: np.ndarray (1,), termination flag
        """
        if instruction is not None:
            self.instruction = instruction
            
        # Generate random action for now
        world_vector = np.random.uniform(-0.05, 0.05, 3)
        rot_axangle = np.random.uniform(-0.05, 0.05, 3)  
        gripper = np.array([0.0])  # Closed
        
        # Format the action for the simpler_env interface
        action = {
            "world_vector": world_vector,
            "rot_axangle": rot_axangle,
            "gripper": gripper,
            "terminate_episode": np.array([False])
        }
        
        # Create a dummy raw_action tensor for compatibility
        raw_action = torch.tensor([0.0])
        
        return raw_action, action
