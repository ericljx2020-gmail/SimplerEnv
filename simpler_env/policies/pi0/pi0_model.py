# simpler_env/policies/pi0/pi0_model.py

"""Lightweight wrapper that adapts the Hugging‑Face Pi0 pipeline to the
SimplerEnv evaluation API (reset → step)."""

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.common.constants import OBS_ROBOT, ACTION
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy, resize_with_pad

class CustomPI0Policy(PI0Policy):
    """Custom version of PI0Policy that handles our image format"""
    
    def __init__(self, config):
        """Initialize with default action dimension of 7"""
        super().__init__(config)
        # Define default action dimension for robotics (pos, rot, gripper)
        self.default_action_dim = 7
    
    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        # Use 'image' key if image_features is empty
        if not self.config.image_features and 'image' in batch:
            present_img_keys = ['image']
        else:
            present_img_keys = [key for key in self.config.image_features if key in batch]
        
        missing_img_keys = []
        
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks
    
    @torch.no_grad
    def select_action(self, batch):
        """Override the select_action method to handle cases where config.action_feature is None"""
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])

        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=None
            )

            # Unpad actions - use default dimension of 7 if config.action_feature is None
            original_action_dim = self.default_action_dim
            actions = actions[:, :, :original_action_dim]

            # No need for unnormalize since we don't have any stats yet
            #actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

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

    # ────────────────────────────────────────────────────────────────────────────
    # Public API expected by evaluation harness
    # ────────────────────────────────────────────────────────────────────────────
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
            
        # Generate random action for now until we properly understand the model output
        # This will let us test the overall pipeline
        world_vector = np.random.uniform(-0.05, 0.05, 3)
        rot_axangle = np.random.uniform(-0.05, 0.05, 3)  
        gripper = np.array([0.0])  # Closed
        
        # Format the action for the simpler_env interface
        action = {
            "world_vector": world_vector,
            "rot_axangle": rot_axangle,
            "gripper": gripper,
            "terminate_episode": np.array([False])  # Pi0 doesn't predict termination
        }
        
        # Create a dummy raw_action tensor for compatibility
        raw_action = torch.tensor([0.0])
        
        return raw_action, action