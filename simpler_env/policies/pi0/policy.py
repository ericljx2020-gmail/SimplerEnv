"""Pi0 policy implementation for SimplerEnv.

This module implements the Pi0 policy, a vision-based foundation model
that can be used for robot manipulation tasks.
"""

import os
import numpy as np
import tensorflow as tf

from simpler_env.policies.policy import Policy


class Pi0Policy(Policy):
    """Pi0 policy implementation.
    
    This policy loads the Pi0 model from Hugging Face Hub and uses it to predict
    actions based on the current observation and instruction.
    """
    
    def __init__(self, model_name="lerobot/pi0", policy_setup="google_robot"):
        """Initialize the Pi0 policy.
        
        Args:
            model_name: The name or path to the Pi0 model.
            policy_setup: The robot setup to use (google_robot or widowx_bridge).
        """
        super().__init__()
        self.model_name = model_name
        self.policy_setup = policy_setup
        self.instruction = None
        
        try:
            # Import here to avoid dependency if not used
            from lerobot.policies import Pi0Policy as LeRobotPi0Policy
            self.pi0_policy = LeRobotPi0Policy.from_pretrained(model_name)
            print(f"Successfully loaded Pi0 model from {model_name}")
        except ImportError:
            raise ImportError(
                "Could not import lerobot.policies. "
                "Please install the lerobot package with: pip install lerobot"
            )
        except Exception as e:
            raise Exception(f"Error loading Pi0 model: {str(e)}")
        
    def reset(self):
        """Reset the policy state."""
        self.instruction = None
        
    def set_instruction(self, instruction):
        """Set the instruction for the policy.
        
        Args:
            instruction: A natural language instruction for the robot.
        """
        print(f"Setting instruction: {instruction}")
        self.instruction = instruction
        
    def step(self, obs):
        """Predict the next action based on the current observation.
        
        Args:
            obs: A dictionary containing the current observation, including
                'rgb_obs' for the RGB image observation.
                
        Returns:
            A dictionary containing the predicted action.
        """
        if self.instruction is None:
            raise ValueError("No instruction set. Call set_instruction first.")
            
        # Check if we have the required RGB observation
        if 'rgb_obs' not in obs:
            raise ValueError("RGB observation is required for Pi0 policy.")
            
        # Process the input
        rgb_image = obs['rgb_obs']
        
        # Create batch with image and instruction
        batch = {
            "image": rgb_image,
            "text": self.instruction
        }
        
        # Get action from policy
        action_dict = self.pi0_policy.select_action(batch)
        
        # Extract action components
        world_vector = action_dict.get("world_vector", np.zeros(3))
        rot_axangle = action_dict.get("rot_axangle", np.zeros(3))
        gripper = action_dict.get("gripper", 0.0)
        terminate_episode = action_dict.get("terminate_episode", False)
        
        # Construct the action dictionary for the environment
        action = {
            "world_vector": world_vector,
            "rot_axangle": rot_axangle,
            "gripper": gripper,
            "terminate_episode": terminate_episode
        }
        
        return action 