# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import yaml
from huggingface_hub import hf_hub_download
from terratorch.registry import TERRATORCH_FULL_MODEL_REGISTRY

from terracodec.imagecodecs import ELIC, FactorizedPrior
from terracodec.temporaltransformer import TemporalTransformer

logger = logging.getLogger("terracodec")

try:
    terracode_available = True
    import_error = None
except Exception as e:
    logger.debug(f"Could not import terracodec due to ImportError({e})")
    terracode_available = False
    import_error = e

# Model definitions
__all__ = [
    "terracodec_v1_fp_s2l2a",
    "terracodec_v1_elic_s2l2a",
    "terracodec_v1_tt_s2l2a",
    "terracodec_v1_tt_s2l1c",
    "flextec_v1_s2l2a",
]

pretrained_weights: Dict[str, Dict[str, Any]] = {
    "terracodec_v1_fp_s2l2a": {
        "hf_hub_id": "embed2scale/TerraCodec-1.0-FP-S2L2A",
        "hf_hub_filename": {
            "lambda-0.5": "TerraCodec_v1_FP_S2L2A_lambda-0.5.pt",
            "lambda-2": "TerraCodec_v1_FP_S2L2A_lambda-2.pt",
            "lambda-10": "TerraCodec_v1_FP_S2L2A_lambda-10.pt",
            "lambda-40": "TerraCodec_v1_FP_S2L2A_lambda-40.pt",
            "lambda-200": "TerraCodec_v1_FP_S2L2A_lambda-200.pt",
        },
    },
    "terracodec_v1_elic_s2l2a": {
        "hf_hub_id": "embed2scale/TerraCodec-1.0-ELIC-S2L2A",
        "hf_hub_filename": {
            "lambda-0.5": "TerraCodec_v1_ELIC_S2L2A_lambda-0.5.pt",
            "lambda-2": "TerraCodec_v1_ELIC_S2L2A_lambda-2.pt",
            "lambda-10": "TerraCodec_v1_ELIC_S2L2A_lambda-10.pt",
            "lambda-40": "TerraCodec_v1_ELIC_S2L2A_lambda-40.pt",
            "lambda-200": "TerraCodec_v1_ELIC_S2L2A_lambda-200.pt",
        },
    },
    "terracodec_v1_tt_s2l2a": {
        "hf_hub_id": "embed2scale/TerraCodec-1.0-TT-S2L2A",
        "hf_hub_filename": {
            "lambda-0.4": "TerraCodec_v1_TT_S2L2A_lambda-0.4.pt",
            "lambda-1": "TerraCodec_v1_TT_S2L2A_lambda-1.pt",
            "lambda-5": "TerraCodec_v1_TT_S2L2A_lambda-5.pt",
            "lambda-20": "TerraCodec_v1_TT_S2L2A_lambda-20.pt",
            "lambda-100": "TerraCodec_v1_TT_S2L2A_lambda-100.pt",
            "lambda-200": "TerraCodec_v1_TT_S2L2A_lambda-200.pt",
            "lambda-700": "TerraCodec_v1_TT_S2L2A_lambda-700.pt",
        },
    },
    "terracodec_v1_tt_s2l1c": {
        "hf_hub_id": "embed2scale/TerraCodec-1.0-TT-S2L1C",
        "hf_hub_filename": {
            "lambda-5": "TerraCodec_v1_TT_S2L1C_lambda-5.pt",
            "lambda-20": "TerraCodec_v1_TT_S2L1C_lambda-20.pt",
            "lambda-100": "TerraCodec_v1_TT_S2L1C_lambda-100.pt",
        },
    },
    "flextec_v1_s2l2a": {
        "hf_hub_id": "embed2scale/TerraCodec-1.0-FlexTEC-S2L2A",
        "hf_hub_filename": {
            "lambda-800": "FlexTEC_v1_S2L2A_lambda-800.pt",
            "lambda-800-lronly": "FlexTEC_LRonly_v1_S2L2A_lambda-800.pt",
        },
    },
}


def set_determinism() -> None:
    # ---- Determinism & numerics ----
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", ":16:8"
    )  # needed for full determinism
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # older torch


def load_config_from_hf(repo_id: str, filename: str) -> dict:
    """Download and parse YAML config from HuggingFace."""
    config_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Remove a prefix from state_dict keys if present."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def build_terracodec(
    model_type: str,
    ckpt_path: Optional[str] = None,
    pretrained: bool = False,
    variant: Optional[str] = None,
    compression: Union[str, float, int] = "lambda-1",
    hf_config: bool = False,
    in_channels: int = 12,  # default for S2 L2A
    N: Optional[int] = None,
    M: Optional[int] = None,
    lr_only: Optional[bool] = None,
    **kwargs,
) -> torch.nn.Module:
    """
    Build a TerraCodec model.

    Args:
        model_type: Type of model ('factorizedprior', 'elic', 'temporaltransformer', 'flextec').
        ckpt_path: Path to a local checkpoint file.
        pretrained: Whether to load pretrained weights.
        variant: Model variant key for loading pretrained models and hf configs.
        compression: Target lambda. Accepts either a numeric (int/float) lambda value or an exact
            string key of the form 'lambda-<value>' (optionally 'lambda-<value>-lronly' for FlexTEC).
            For numeric input, the nearest available lambda checkpoint will be used (a warning is
            raised if it's not an exact match). No other formats are allowed.
        hf_config: Whether to load config from HuggingFace.
        in_channels: Number of input channels.
        N: Depth of conv. layers N for image codecs.
        M: Bottleneck dimension M for image codecs.
        lr_only: For FlexTEC variants, choose LR-only checkpoint/behavior when True.
        **kwargs: Additional model arguments.

    Returns:
        torch.nn.Module: Initialized TerraCodec model.
    """
    set_determinism()

    # ---- Helpers for compression handling ----
    def _parse_available_keys(
        keys: Iterable[str], lr_only_flag: Optional[bool]
    ) -> Tuple[Dict[float, str], Dict[float, str]]:
        """Return mappings from numeric lambda -> key, split by standard vs lr-only.

        Args:
            keys: iterable of keys like 'lambda-5', 'lambda-800-lronly'.
            lr_only_flag: when provided, can be used later to select the subset.

        Returns:
            (standard_map, lr_only_map): numeric->key for each subset.
        """
        std_map: Dict[float, str] = {}
        lr_map: Dict[float, str] = {}
        pat = re.compile(r"^lambda-([0-9]*\.?[0-9]+)(-lronly)?$")
        for k in keys:
            m = pat.match(k)
            if not m:
                # Ignore non-conforming keys silently
                continue
            val = float(m.group(1))
            if m.group(2):
                lr_map[val] = k
            else:
                std_map[val] = k
        return std_map, lr_map

    def _select_lambda_key(
        compression_value: Union[str, float, int],
        keys: Iterable[str],
        variant_name: str,
        lr_only_flag: Optional[bool] = None,
    ) -> str:
        """Validate and resolve the requested compression to an available key.

        Rules:
        - If string 'lambda-<v>' (or 'lambda-<v>-lronly'), must match an available key exactly;
          otherwise raise ValueError.
        - If numeric, choose the nearest available lambda (in the appropriate subset for FlexTEC
          when lr_only_flag is provided). Warn when not exact.
        - Any other type raises ValueError.
        """
        available = list(keys)
        if isinstance(compression_value, str):
            if not compression_value.startswith("lambda-"):
                raise ValueError(
                    f"compression must be numeric or of the form 'lambda-<value>' (optionally '-lronly' for FlexTEC). Got: {compression_value}"
                )
            if compression_value not in available:
                raise ValueError(
                    f"compression='{compression_value}' not available for variant '{variant_name}'. "
                    f"Available: {available}"
                )
            return compression_value

        if isinstance(compression_value, (int, float)):
            std_map, lr_map = _parse_available_keys(available, lr_only_flag)

            # Choose subset for FlexTEC when lr_only_flag is set; otherwise prefer standard map and
            # fallback to lr_map if standard is empty (robustness in case only lr-only exists).
            if lr_only_flag is True and lr_map:
                target_map = lr_map
            elif lr_only_flag is False and std_map:
                target_map = std_map
            else:
                # pick the non-empty one
                target_map = std_map or lr_map

            if not target_map:
                raise ValueError(
                    f"No parsable lambda checkpoints found for variant '{variant_name}'. Keys: {available}"
                )

            req = float(compression_value)
            # Find nearest lambda
            lambdas = sorted(target_map.keys())
            nearest = min(lambdas, key=lambda x: abs(x - req))
            chosen_key = target_map[nearest]
            if abs(nearest - req) > 1e-9:
                warnings.warn(
                    f"Requested lambda={req} not available for '{variant_name}'. Using nearest available lambda={nearest} (key='{chosen_key}')."
                )
            return chosen_key

        raise ValueError(
            f"compression must be a number or 'lambda-<value>' string. Got type {type(compression_value)}"
        )

    # Prepare config
    config = {}
    if hf_config:
        if variant is None:
            raise ValueError("variant must be provided when hf_config=True.")
        repo_id = pretrained_weights[variant]["hf_hub_id"]

        # Resolve to a lambda key if possible (uses available filenames when present)
        available_keys = pretrained_weights[variant]["hf_hub_filename"].keys()
        resolved_key = _select_lambda_key(
            compression, available_keys, variant, lr_only
        )
        config_filename = f"config_{resolved_key}.yaml"

        try:
            config = load_config_from_hf(repo_id, config_filename)
        except Exception as e:
            warnings.warn(
                f"Could not load config from {config_filename} on repo {repo_id}: {e}"
            )
    else:
        if in_channels is not None:
            config["in_channels"] = in_channels
        if N is not None:
            config["N"] = N
        if M is not None:
            config["M"] = M

    # Model selection
    if model_type in ["factorizedprior", "elic"]:
        required_keys = {
            k: config[k] for k in ["in_channels", "N", "M"] if k in config
        }
        model_args = {**required_keys, **kwargs}
        if model_type == "factorizedprior":
            model = FactorizedPrior(**model_args)
        else:
            model = ELIC(**model_args)

    elif model_type == "temporaltransformer":
        model = TemporalTransformer(**kwargs, in_channels=in_channels)
    elif model_type == "flextec":
        model = TemporalTransformer(**kwargs, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    if ckpt_path:
        state_dict = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
        if variant in ["terracodec_v1_fp_s2l2a", "terracodec_v1_elic_s2l2a"]:
            state_dict = strip_prefix(state_dict, prefix="model.")
        loaded_keys = model.load_state_dict(state_dict, strict=False)
        if loaded_keys.missing_keys:
            warnings.warn(f"Missing keys: {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            warnings.warn(f"Unexpected keys: {loaded_keys.unexpected_keys}")

    elif pretrained:
        if variant is None:
            raise ValueError("variant must be provided when pretrained=True.")

        # Validate/resolve the requested compression to a proper filename key
        available = pretrained_weights[variant]["hf_hub_filename"].keys()
        resolved_key = _select_lambda_key(
            compression, available, variant, lr_only
        )

        # Load model from Hugging Face
        state_dict_file = hf_hub_download(
            repo_id=pretrained_weights[variant]["hf_hub_id"],
            filename=pretrained_weights[variant]["hf_hub_filename"][
                resolved_key
            ],
        )
        state_dict = torch.load(
            state_dict_file, map_location="cpu", weights_only=True
        )
        if variant in ["terracodec_v1_fp_s2l2a", "terracodec_v1_elic_s2l2a"]:
            state_dict = strip_prefix(state_dict, prefix="model.")
        model.load_state_dict(state_dict, strict=True)

        if variant in [
            "terracodec_v1_tt_s2l2a",
            "terracodec_v1_tt_s2l1c",
            "flextec_v1_s2l2a",
        ]:
            assert hasattr(model, "setup_model_for_compression")
            model.setup_model_for_compression()

    # Update bottleneck params
    if hasattr(model, "update"):
        try:
            model.update(force=True)
        except Exception:
            pass

    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terracodec_v1_fp_s2l2a(
    compression: Union[str, float, int] = "lambda-10",
    image_size: int = 256,
    mode="eval",
    **kwargs,
):
    """
    TerraCodec 1.0 FactorizedPrior model for Sentinel-2 L2A data.
    """
    model = build_terracodec(
        model_type="factorizedprior",
        variant="terracodec_v1_fp_s2l2a",
        compression=compression,
        hf_config=True,  # configs are provided on hf
        image_size=image_size,
        **kwargs,
    )

    if mode == "eval":
        model.eval()

    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terracodec_v1_elic_s2l2a(
    compression: Union[str, float, int] = "lambda-10",
    image_size: int = 256,
    mode="eval",
    **kwargs,
):
    """
    TerraCodec 1.0 ELIC model for Sentinel-2 L2A data.
    """
    model = build_terracodec(
        model_type="elic",
        variant="terracodec_v1_elic_s2l2a",
        compression=compression,
        hf_config=True,  # configs are provided on hf
        image_size=image_size,
        **kwargs,
    )

    if mode == "eval":
        model.eval()

    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terracodec_v1_tt_s2l2a(
    compression: Union[str, float, int] = "lambda-5",
    image_size: int = 256,
    mode="eval",
    **kwargs,
):
    """
    TerraCodec 1.0 Temporal Transformer model for Sentinel-2 L2A data.
    """
    model = build_terracodec(
        model_type="temporaltransformer",
        variant="terracodec_v1_tt_s2l2a",
        compression=compression,
        image_size=image_size,
        **kwargs,
    )

    if mode == "eval":
        model.eval()

    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terracodec_v1_tt_s2l1c(
    compression: Union[str, float, int] = "lambda-20",
    image_size: int = 256,
    mode="eval",
    **kwargs,
):
    """
    TerraCodec 1.0 Temporal Transformer model for Sentinel-2 L1C data.
    """
    model = build_terracodec(
        model_type="temporaltransformer",
        variant="terracodec_v1_tt_s2l1c",
        compression=compression,
        image_size=image_size,
        in_channels=13,  # S2 L1C has 13 channels
        **kwargs,
    )

    if mode == "eval":
        model.eval()

    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def flextec_v1_s2l2a(
    compression: Union[str, float, int] = "lambda-800",
    image_size: int = 256,
    mode="eval",
    lr_only: bool = False,
    **kwargs,
):
    """
    TerraCodec 1.0 Temporal Transformer model for Sentinel-2 L2A data.
    """
    model = build_terracodec(
        model_type="flextec",
        variant="flextec_v1_s2l2a",
        compression=compression,
        image_size=image_size,
        flextec=True,
        lr_only=lr_only,
        **kwargs,
    )

    if mode == "eval":
        model.eval()

    return model
