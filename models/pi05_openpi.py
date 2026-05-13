"""
AutoKernel wrapper for profiling openpi's pi0.5 PyTorch inference path.

Setup:
    1. Run one-time setup from the autokernel directory:
       uv run setup_pi05_openpi.py --download --convert
    2. Export the converted checkpoint path:
       export OPENPI_PI05_PT_CHECKPOINT=~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch

Profile with AutoKernel:
    uv run profile.py \
        --model models/pi05_openpi.py \
        --class-name PI05AutoKernelModel \
        --input-shape 1,10 \
        --dtype bfloat16

`--input-shape` is interpreted as:
    - dim 0: batch size
    - dim 1: denoising steps (optional, defaults to OPENPI_PI05_NUM_STEPS or 10)
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import sys

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPENPI_SRC = _REPO_ROOT / "openpi" / "src"
_OPENPI_CLIENT_SRC = _REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"
for _path in (_OPENPI_SRC, _OPENPI_CLIENT_SRC):
    if _path.exists():
        path_str = str(_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from openpi.models import model as _model
from openpi.training import config as _config


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class PI05AutoKernelModel(nn.Module):
    """Thin nn.Module wrapper around openpi's pi0.5 PyTorch inference."""

    autokernel_force_generic_input = True

    def __init__(self) -> None:
        super().__init__()

        config_name = os.environ.get("OPENPI_PI05_CONFIG", "pi05_droid")
        checkpoint_dir = Path(
            os.environ.get(
                "OPENPI_PI05_PT_CHECKPOINT",
                "~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch",
            )
        ).expanduser()
        weight_path = checkpoint_dir / "model.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(
                "Converted pi0.5 PyTorch checkpoint not found.\n"
                f"Expected: {weight_path}\n"
                "Set OPENPI_PI05_PT_CHECKPOINT to the converted checkpoint directory."
            )

        train_config = _config.get_config(config_name)
        compile_enabled = _env_flag("OPENPI_PI05_TORCH_COMPILE", False)
        model_config = dataclasses.replace(
            train_config.model,
            pytorch_compile_mode=train_config.model.pytorch_compile_mode if compile_enabled else None,
        )
        train_config = dataclasses.replace(train_config, model=model_config)

        self.model = train_config.model.load_pytorch(train_config, str(weight_path))
        self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

        observation_spec, _ = train_config.model.inputs_spec(batch_size=1)
        first_image = next(iter(observation_spec.images.values()))
        self.image_height = first_image.shape[1]
        self.image_width = first_image.shape[2]
        self.image_keys = tuple(observation_spec.images.keys())
        self.state_dim = observation_spec.state.shape[1]
        self.max_token_len = observation_spec.tokenized_prompt.shape[1]

        self.default_num_steps = int(os.environ.get("OPENPI_PI05_NUM_STEPS", "10"))
        self.prompt_tokens = max(
            1,
            min(int(os.environ.get("OPENPI_PI05_PROMPT_TOKENS", "32")), self.max_token_len),
        )
        self._precision_mode = "bfloat16"

    def to(self, *args, **kwargs):  # type: ignore[override]
        device = kwargs.get("device")
        requested_dtype = kwargs.get("dtype")

        for arg in args:
            if isinstance(arg, (str, torch.device)):
                device = arg
            elif isinstance(arg, torch.dtype):
                requested_dtype = arg

        if device is not None:
            super().to(device=device)

        # Keep openpi's mixed-precision layout intact. Global dtype casts break
        # the action head because sample_actions internally creates float32
        # tensors even when the profiler asks for bf16 model casting.
        if requested_dtype == torch.float32:
            self.model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")
            self._precision_mode = "float32"
        else:
            self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
            self._precision_mode = "bfloat16"

        return self

    def _resolve_num_steps(self, x: torch.Tensor) -> int:
        if x.ndim >= 2:
            return max(1, int(x.shape[1]))
        return self.default_num_steps

    def _build_observation(self, batch_size: int, device: torch.device) -> _model.Observation[torch.Tensor]:
        images = {
            key: torch.zeros(
                (batch_size, 3, self.image_height, self.image_width),
                device=device,
                dtype=torch.float32,
            )
            for key in self.image_keys
        }
        image_masks = {
            key: torch.ones((batch_size,), device=device, dtype=torch.bool)
            for key in self.image_keys
        }
        tokenized_prompt = torch.zeros((batch_size, self.max_token_len), device=device, dtype=torch.long)
        tokenized_prompt[:, : self.prompt_tokens] = 1
        tokenized_prompt_mask = torch.zeros((batch_size, self.max_token_len), device=device, dtype=torch.bool)
        tokenized_prompt_mask[:, : self.prompt_tokens] = True

        return _model.Observation(
            images=images,
            image_masks=image_masks,
            state=torch.zeros((batch_size, self.state_dim), device=device, dtype=torch.float32),
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = int(x.shape[0]) if x.ndim >= 1 else 1
        num_steps = self._resolve_num_steps(x)
        observation = self._build_observation(batch_size, x.device)
        return self.model.sample_actions(x.device, observation, num_steps=num_steps)
