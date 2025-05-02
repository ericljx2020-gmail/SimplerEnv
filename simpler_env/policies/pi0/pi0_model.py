# simpler_env/policies/pi0/pi0_model.py
"""Pi0 integration for SimplerEnv (Google-Robot & WidowX setups)."""

from collections import deque
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

# --- LeRobot ---------------------------------------------------------------
from lerobot.common.constants import OBS_ROBOT, ACTION        # ≥HF-LeRobot 0.6.0 :contentReference[oaicite:1]{index=1}
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy, resize_with_pad  # resize helper :contentReference[oaicite:2]{index=2}
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

# ─────────────────────────────────────────────────────────────────────────────
# 1.  A thin subclass that replaces the default image preprocessing so we can
#     pass in SimplerEnv’s (H,W,3) uint8 frames directly.
# ─────────────────────────────────────────────────────────────────────────────
class CustomPI0Policy(PI0Policy):

    def __init__(self, cfg: PI0Config, **kwargs):
        super().__init__(cfg, **kwargs)
        self.default_action_dim = 7   # Δx Δy Δz ‖ axis-angle(3) ‖ gripper

    # --- simplified image path ------------------------------------------------
    def prepare_images(self, batch: Dict[str, torch.Tensor]) -> Tuple[list, list]:
        """Accept either
           • batch['image']   (SimplerEnv default key), or
           • structured 'observation.image.top' / etc.  """
        key = "image" if "image" in batch else "observation.image.top"
        img = batch[key]             # (B,3,H,W), float32 [0,1]

        # Pi0 was trained on 224² SigLIP crops :contentReference[oaicite:3]{index=3}
        img = resize_with_pad(img, 224, 224, pad_value=0)        # (B,3,224,224)
        img = img * 2.0 - 1.0                                    # map [0,1]→[-1,1]

        return [img], [torch.ones(img.shape[0], dtype=torch.bool, device=img.device)]

    # --- action queue wrapper -------------------------------------------------
    @torch.no_grad
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B,7) first-step action (identical semantics to RT-1 wrapper)."""
        batch = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            imgs, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            tokens, token_masks = self.prepare_language(batch)
            # (B, 50, 32) → queue of (7,) actions
            acts = self.model.sample_actions(imgs, img_masks, tokens, token_masks, state)[:, :, : self.default_action_dim]
            self._action_queue.extend(acts.transpose(0, 1))      # 50 × (B,7)

        return self._action_queue.popleft()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Public wrapper class (matches RT-1Inference signature).
# ─────────────────────────────────────────────────────────────────────────────
class Pi0Inference:
    """RT-1-style wrapper so SimplerEnv can call reset() / step()."""
    
    DEFAULT_IMG_STATS = dict(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])     # ImageNet ≈ SigLIP

    DEFAULT_STATE_STD   = 0.25        # rad  (π-0 paper, Eq. 5)  :contentReference[oaicite:1]{index=1}
    DEFAULT_ACTION_STD  = 0.05        # m    (Appendix C)       :contentReference[oaicite:2]{index=2}
    ACTION_TRANSLATION_SCALE = 0.05   # m per unit             :contentReference[oaicite:3]{index=3}
    ACTION_ROTATION_SCALE    = 0.25   # rad per unit           :contentReference[oaicite:4]{index=4}


    def __init__(
        self,
        model_name: str = "lerobot/pi0",
        policy_setup: str = "google_robot",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.policy_setup = policy_setup
        self.instruction: Optional[str] = None

        # ---------- build a minimal cfg that side-steps Draccus DecodingError ----
        input_feats = {
            "observation.image.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state":     PolicyFeature(type=FeatureType.STATE,  shape=(7,)),
        }
        output_feats = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

        cfg = PI0Config(
            input_features=input_feats,
            output_features=output_feats,
            normalization_mapping={k: NormalizationMode.IDENTITY for k in ["VISUAL", "STATE", "ACTION"]},   # dataset_stats absent :contentReference[oaicite:4]{index=4}
            device=device,
        )
        dummy_stats = {k: {"mean": torch.zeros(v.shape), "std": torch.ones(v.shape)} for k, v in input_feats.items()}
        dummy_stats["action"] = {"mean": torch.zeros(7), "std": torch.ones(7)}

        self.policy: CustomPI0Policy = CustomPI0Policy.from_pretrained(   # example in upstream doc :contentReference[oaicite:5]{index=5}
            model_name, config=cfg, dataset_stats=dummy_stats, strict=False
        ).eval()
        self.device = cfg.device

    def _whiten_image(self, img_t: torch.Tensor) -> torch.Tensor:
        DEFAULT_IMG_STATS = dict(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
        mean = torch.tensor(DEFAULT_IMG_STATS["mean"], device=img_t.device)[:, None, None]
        std  = torch.tensor(DEFAULT_IMG_STATS["std"],  device=img_t.device)[:, None, None]
        return (img_t / 255.0 - mean) / std

    # ---------- SimplerEnv contract -------------------------------------------
    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.policy.reset()                       # clear internal action queue

    def step(self, image: np.ndarray, instruction: str | None = None,
         state: np.ndarray | None = None):

        DEFAULT_STATE_STD   = 0.25        # rad  (π-0 paper, Eq. 5)  :contentReference[oaicite:1]{index=1}
        DEFAULT_ACTION_STD  = 0.05        # m    (Appendix C)       :contentReference[oaicite:2]{index=2}
        ACTION_TRANSLATION_SCALE = 0.05   # m per unit             :contentReference[oaicite:3]{index=3}
        ACTION_ROTATION_SCALE    = 0.25   # rad per unit           :contentReference[oaicite:4]{index=4}

        if instruction:
            self.instruction = instruction

        img_t = torch.as_tensor(image).permute(2, 0, 1).float().to(self.device)  # (3,H,W)
        img_t = img_t.unsqueeze(0)                                               # (1,3,H,W)
        img_t = resize_with_pad(img_t, 224, 224, pad_value=0)                    # pad+resize
        img_t = self._whiten_image(img_t)

        if state is None:
            state = np.zeros(7, dtype=np.float32)
        st_t = torch.as_tensor(state / DEFAULT_STATE_STD,            # z-score
                            device=self.device).unsqueeze(0)

        batch = {
            "image": img_t,
            "observation.state": st_t,
            "task": [self.instruction or ""],
        }
        with torch.no_grad():
            a_norm = self.policy.select_action(batch)[0].cpu().numpy()

        # un-normalise & scale into simulator units
        world = a_norm[:3] * ACTION_TRANSLATION_SCALE
        rot   = a_norm[3:6] * ACTION_ROTATION_SCALE
        grip  = a_norm[6:7]                     # already ∈ [-1, 1]

        return {ACTION: a_norm}, {
            "world_vector": world,
            "rot_axangle": rot,
            "gripper": grip,
            "terminate_episode": np.array([False]),
        }
