**Integrating Pi0 Model into SimplerEnv Evaluation Script**

## Overview

This document describes the steps required to add support for the Pi0 model (hosted at `lerobot/pi0` on Hugging Face) into the `simpler_env` real-to-sim evaluation harness. Once complete, you will be able to run:

```bash
python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy pi0 \
    --ckpt-path lerobot/pi0 \
    --task <your_task> --logging-root <results_dir> --n-trajs <N>
```

## Prerequisites

- `simpler_env` repo checked out locally.
- Python 3.x environment with `transformers`, `mediapy`, `tensorflow`, and `tf_agents` installed.
- Access to the `lerobot/pi0` model on Hugging Face.

## 1. Create a new policy directory

In your `simpler_env` repository, under `simpler_env/policies/`, create a folder named `pi0`:

```
cd simpler_env/policies
mkdir pi0
```

## 2. Add the Pi0 inference class

Inside `simpler_env/policies/pi0/`, create a file `pi0_model.py` with the following content:

```python
# simpler_env/policies/pi0/pi0_model.py

from transformers import pipeline
import numpy as np

class Pi0Inference:
    def __init__(self,
                 model_name: str = "lerobot/pi0",
                 policy_setup: str = "google_robot"):
        """
        Load the Pi0 pipeline from Hugging Face.

        Args:
            model_name: HF model identifier (e.g., "lerobot/pi0").
            policy_setup: environment setup string ("google_robot" or "widowx_bridge").
        """
        self.pipe = pipeline(
            task="image-to-action",
            model=model_name,
            device=0  # or -1 for CPU
        )
        self.policy_setup = policy_setup

    def reset(self, instruction: str) -> None:
        # Store the instruction for this episode
        self.instruction = instruction

    def step(self, image: np.ndarray, instruction: str):
        """
        Run Pi0 on the given image and instruction.

        Args:
            image: np.ndarray, shape (H, W, 3), dtype uint8
            instruction: str, task description

        Returns:
            raw_action: dict with model outputs
            action: dict with keys:
                - world_vector: np.ndarray (3,), translation
                - rot_axangle: np.ndarray (3,), axis-angle rotation
                - gripper: np.ndarray (1,), gripper command
                - terminate_episode: np.ndarray (1,), termination flag
        """
        # Pi0 expects PIL images or arrays; adapt if necessary
        result = self.pipe({"image": image, "prompt": instruction})

        # Extract and format outputs
        raw_action = result
        action = {
            "world_vector": np.array(result["world_vector"], dtype=np.float64),
            "rot_axangle": np.array(result["rot_axangle"],    dtype=np.float64),
            "gripper":   np.array(result["gripper"],        dtype=np.float64),
            "terminate_episode": np.array(result["terminate_episode"])  
        }
        return raw_action, action
```

## 3. Make the package importable

In `simpler_env/policies/pi0/`, add an empty `__init__.py`:

```bash
cd simpler_env/policies/pi0
touch __init__.py
```

## 4. Update the evaluation script

Open `simpler_env/simple_inference_visual_matching_prepackaged_envs.py` and:

1. **Expand the parser choices** to include `pi0`:

    ```diff
    parser.add_argument(
        "--policy", default="rt1",
-       choices=["rt1", "octo-base", "octo-small"]
+       choices=["rt1", "octo-base", "octo-small", "pi0"]
    )
    ```

2. **Add a branch** to load `Pi0Inference`:

    ```diff
    if args.policy == "rt1":
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        model = RT1Inference(saved_model_path=args.ckpt_path, policy_setup=policy_setup)
    elif "octo" in args.policy:
        from simpler_env.policies.octo.octo_model import OctoInference
        model = OctoInference(model_type=args.ckpt_path, policy_setup=policy_setup, init_rng=0)
    +elif args.policy == "pi0":
    +    from simpler_env.policies.pi0.pi0_model import Pi0Inference
    +    model = Pi0Inference(model_name=args.ckpt_path, policy_setup=policy_setup)
    else:
        raise NotImplementedError()
    ```

## 5. (Optional) Update Documentation

In your repository README or examples, add usage for the new policy:

```markdown
### Pi0 Evaluation

```bash
python simpler_env/simple_inference_visual_matching_prepackaged_envs.py \
    --policy pi0 \
    --ckpt-path lerobot/pi0 \
    --task google_robot_pick_horizontal_coke_can \
    --logging-root ./results_pi0_eval \
    --n-trajs 10
```
```

---

Once you complete these steps, running with `--policy pi0` will instantiate your `Pi0Inference` class and integrate seamlessly into the SimplerEnv eval pipeline.  

