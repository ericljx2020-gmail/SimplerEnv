"""Pi0 integration for SimplerEnv (Google‑Robot & WidowX setups)."""

from collections import deque
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

# --- LeRobot ----------------------------------------------------------------
from lerobot.common.constants import OBS_ROBOT, ACTION
# from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy, resize_with_pad
from lerobot.common.policies.pi0.modeling_pi0_finetuned import PI0Policy, resize_with_pad
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Thin subclass to override image preprocessing.
# ─────────────────────────────────────────────────────────────────────────────
class CustomPI0Policy(PI0Policy):
    def __init__(self, cfg: PI0Config, **kwargs):
        super().__init__(cfg, **kwargs)
        self.default_action_dim = 7   # Δx Δy Δz | axis‑angle(3) | gripper

    # --- simplified image path ----------------------------------------------
    # def prepare_images(self, batch: Dict[str, torch.Tensor]) -> Tuple[list, list]:
    #     key = "image" if "image" in batch else "observation.image.top"
    #     img = batch[key]                                  # (B,3,H,W) float32 [0,1]
    #     img = resize_with_pad(img, 224, 224, pad_value=0) # (B,3,224,224)
    #     img = img * 2.0 - 1.0                             # → [‑1,1]
    #     return [img], [torch.ones(img.shape[0], dtype=torch.bool, device=img.device)]

    # --- en‑queue 50‑step autoregressive actions -----------------------------
    @torch.no_grad
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            imgs, img_masks, dav2_images, dav2_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            tokens, token_masks = self.prepare_language(batch)
            acts = self.model.sample_actions(
                imgs, img_masks, tokens, token_masks, state, dav2_images, dav2_masks
            )[:, :, : self.default_action_dim]            # (B,50,7)
            self._action_queue.extend(acts.transpose(0, 1))  # 50 × (B,7)

        return self._action_queue.popleft()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Public RT‑1‑style wrapper.
# ─────────────────────────────────────────────────────────────────────────────
class Pi0Inference:
    """RT‑1‑style wrapper so SimplerEnv can call reset() / step()."""

    DEFAULT_IMG_STATS = dict(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])

    DEFAULT_STATE_STD   = 0.25
    DEFAULT_ACTION_STD  = 0.05
    ACTION_TRANSLATION_SCALE = 0.05   # m per unit
    ACTION_ROTATION_SCALE    = 0.25   # rad per unit

    # -------- ① 训练集统计量（来自 stats.json） ------------------------------
    _OBS_MEAN = torch.tensor([
        0.3094516694545746,  0.030725326389074326, 0.06443972140550613,
        0.006490672472864389, -0.07720066606998444, 0.10766016691923141,
        0.0, 0.7081289887428284
    ])
    _OBS_STD = torch.tensor([
        0.060603249818086624, 0.09195369482040405, 0.05159375071525574,
        0.13121765851974487, 0.16923990845680237, 0.5787228941917419,
        0.0, 0.3536507487297058
    ])
    _ACT_MEAN = torch.tensor([
        0.00022731945500709116,  0.00013112067244946957, -0.000126416256534867,
       -0.00014410920266527683, -0.0003903077158611268,  0.00024063372984528542,
        0.5766201019287109
    ])
    _ACT_STD = torch.tensor([
        0.009782361797988415, 0.013714045286178589, 0.012687387876212597,
        0.02848990075290203, 0.030552811920642853, 0.0775114968419075,
        0.4940855801105499
    ])

    def __init__(
        self,
        model_name: str = "lerobot/pi0",
        policy_setup: str = "google_robot",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.policy_setup = policy_setup
        self.instruction: Optional[str] = None

        # ---------- build minimal cfg ----------------------------------------
        STATE_DIM = 8  # 改成 8，与 _OBS_* 对齐
        input_feats = {
            "observation.image.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state":     PolicyFeature(type=FeatureType.STATE,  shape=(STATE_DIM,)),
        }
        output_feats = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

        # cfg = PI0Config(
        #     input_features=input_feats,
        #     output_features=output_feats,
        #     normalization_mapping={k: NormalizationMode.IDENTITY for k in ["VISUAL", "STATE", "ACTION"]},
        #     device=device,
        # )
        # ─ 在 Pi0Inference.__init__ 中，建 cfg 时改这一段 ─
        cfg = PI0Config(
            input_features=input_feats,
            output_features=output_feats,
            normalization_mapping={
                "VISUAL": NormalizationMode.IDENTITY,  # 图像只需 [0,1]→[-1,1]
                "STATE":  NormalizationMode.IDENTITY,    # 关键：用你传进来的 dataset_stats
                "ACTION": NormalizationMode.IDENTITY,    # 关键：动作也自动反归一
            },
            device=device,
        )


        # -------- ② 组装 dataset_stats --------------------------------------
        dataset_stats = {
            "observation.state": {
                "mean": self._OBS_MEAN,
                "std":  self._OBS_STD,
            },
            "action": {
                "mean": self._ACT_MEAN,
                "std":  self._ACT_STD,
            },
            # 视觉特征：恒等映射 → 均值 0、方差 1 即可
            "observation.image.top": {
                "mean": torch.zeros(3, 224, 224),
                "std":  torch.ones(3, 224, 224),
            },
        }

        self.policy: CustomPI0Policy = CustomPI0Policy.from_pretrained(
            model_name, config=cfg, dataset_stats=dataset_stats, strict=False
        ).eval()
        # ── DEBUG ──
        print(">>> PI0 input_features:", self.policy.config.input_features)
        print(">>> PI0 output_features:", self.policy.config.output_features)
        self.device = cfg.device

    # ------------------------------------------------------------------------
    def _whiten_image(self, img_t: torch.Tensor) -> torch.Tensor:
        # Convert image from [0, 255] or [0, 1] to [-1, 1] range
        # First ensure we're in [0, 1] range
        if img_t.max() > 1.0:
            img_t = img_t / 255.0
        
        # Then convert to [-1, 1] range as expected by SigLIP/PaliGemma
        img_t = img_t * 2.0 - 1.0
        
        return img_t

    # ----------------- SimplerEnv contract ----------------------------------
    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.policy.reset()

    def step(self, image: np.ndarray, instruction: str | None = None,
             state: np.ndarray | None = None):

        if instruction:
            self.instruction = instruction

        img_t = torch.as_tensor(image).permute(2, 0, 1).float().to(self.device)  # (3,H,W)
        img_t = img_t.unsqueeze(0)                                               # (1,3,H,W)
        img_t = resize_with_pad(img_t, 224, 224, pad_value=0)
        img_t = self._whiten_image(img_t)

        if state is None:
            state = np.zeros(8, dtype=np.float32)  # 真实部署时要传环境的 state
        mean  = self._OBS_MEAN.numpy()
        std   = self._OBS_STD.numpy()
        safe_std = np.where(std == 0.0, 1.0, std)        # 避免除 0
        st_norm = (state - mean) / safe_std
        st_norm[std == 0.0] = 0.0                        # 常数维 → 0
        st_t = torch.as_tensor(st_norm, device=self.device).unsqueeze(0)

        
        batch = {
            "observation.image.top": img_t,
            "observation.state": st_t,
            "task": [self.instruction or ""],
        }

        # ── DEBUG ──
        print("IMG tensor dtype/min/max:", img_t.dtype, img_t.min().item(), img_t.max().item())
        print("STATE tensor mean/std:", st_t.mean().item(), st_t.std().item())
        print("TASK string:", self.instruction)
        
        # Set to use safe execution with retry logic
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
        # Reduce these values if movements are too aggressive
        translation_scale = self.ACTION_TRANSLATION_SCALE * 0.5  # Reduced by half
        rotation_scale = self.ACTION_ROTATION_SCALE * 0.5       # Reduced by half
        
        # un‑normalize with smaller scales for smoother movements
        world = a_unnorm[:3] * translation_scale
        rot   = a_unnorm[3:6] * rotation_scale
        grip  = a_unnorm[6:7]

        return {ACTION: a_unnorm}, {
            "world_vector": world,
            "rot_axangle": rot,
            "gripper": grip,
            "terminate_episode": np.array([False]),
        }
