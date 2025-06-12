import argparse
import contextlib
import json
import logging
import os
from typing import List

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# --- Standard Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
MODEL_DIFFUSION_PREFIX = "model.diffusion_model."

# =================================================================================
# ===                  MODULAR MERGING HELPER FUNCTIONS                       ===
# =================================================================================


def get_canonical_key(original_key: str) -> str:
    """Strips a common prefix from key names to standardize them."""
    if original_key.startswith(MODEL_DIFFUSION_PREFIX):
        return original_key.removeprefix(MODEL_DIFFUSION_PREFIX)
    return original_key


def prune_none(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """A no-op pruning function that returns the original tensor."""
    return tensor


def prune_magnitude(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Prunes a tensor by keeping the top `density` fraction of values by magnitude."""
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)

    k = int(density * tensor.numel())
    logger.debug(f"    - Pruning tensor of shape {tensor.shape}. Keeping {k} of {tensor.numel()} elements.")
    if k == 0:
        return torch.zeros_like(tensor)

    threshold = torch.topk(tensor.abs().flatten(), k=k, sorted=True).values[-1]
    mask = tensor.abs() >= threshold
    return tensor * mask


def prune_random(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Prunes a tensor by randomly keeping `density` fraction of values and rescaling."""
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)

    mask = torch.bernoulli(torch.full_like(tensor, float(density)))
    pruned_tensor = tensor * mask
    logger.debug(f"    - Pruning tensor of shape {tensor.shape}. Randomly kept {int(mask.sum())} elements.")

    if density > 0:
        pruned_tensor /= density
    return pruned_tensor


PRUNE_METHODS = {
    "none": prune_none,
    "magnitude": prune_magnitude,
    "random": prune_random,
}


def calculate_majority_sign_mask(tensor_stack: torch.Tensor, method: str) -> torch.Tensor:
    """Calculates a mask where signs match the majority sign."""
    if method == "total":
        sign_magnitude = tensor_stack.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = tensor_stack.sign().sum(dim=0)
    else:
        raise ValueError(f"Unknown majority_sign_method: {method}")

    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return tensor_stack.sign() == majority_sign


def disjoint_merge_wrapper(weighted_deltas: torch.Tensor, majority_sign_method: str) -> torch.Tensor:
    """Wrapper for the disjoint merge operation."""
    sign_mask = calculate_majority_sign_mask(weighted_deltas, majority_sign_method)
    logger.debug(f"    - Disjoint merge: {int(sign_mask.sum())} of {sign_mask.numel()} elements had sign agreement.")
    masked_tensors = weighted_deltas * sign_mask
    summed_tensors = masked_tensors.sum(dim=0)
    num_params_preserved = sign_mask.sum(dim=0)
    return summed_tensors / torch.clamp(num_params_preserved, min=1.0)


def sum_merge(weighted_deltas: torch.Tensor, majority_sign_method: str) -> torch.Tensor:
    """Wrapper for the sum operation."""
    return weighted_deltas.sum(dim=0)


def mean_merge(weighted_deltas: torch.Tensor, majority_sign_method: str) -> torch.Tensor:
    """Wrapper for the mean operation."""
    return weighted_deltas.mean(dim=0)


MERGE_METHODS = {
    "sum": sum_merge,
    "mean": mean_merge,
    "disjoint_merge": disjoint_merge_wrapper,
}


def reshape_weights(tensors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reshapes weights to be broadcastable with tensors."""
    return weights.view(weights.shape + (1,) * (tensors.dim() - weights.dim()))


# =================================================================================
# ===                      CORE TASK ARITHMETIC FUNCTION                        ===
# =================================================================================


def compute_merged_delta(
    deltas: List[torch.Tensor],
    weights: torch.Tensor,
    prune_method: str,
    merge_method: str,
    density: float,
    majority_sign_method: str,
) -> torch.Tensor:
    """Performs task arithmetic using a modular combination of pruning and merging."""
    logger.debug("  - Entering compute_merged_delta ...")
    prune_function = PRUNE_METHODS[prune_method]
    pruned_deltas = [prune_function(d, density) for d in deltas]
    logger.debug(f"  - Step 1/3 (Prune): Completed '{prune_method}' pruning.")

    stacked_deltas = torch.stack(pruned_deltas, dim=0)
    reshaped_w = reshape_weights(stacked_deltas, weights)
    weighted_deltas = stacked_deltas * reshaped_w
    logger.debug(f"  - Step 2/3 (Weight): Deltas weighted successfully.")

    merge_function = MERGE_METHODS[merge_method]
    merged_delta = merge_function(weighted_deltas, majority_sign_method)
    logger.debug(f"  - Step 3/3 (Merge): Completed '{merge_method}' merge.")

    return merged_delta


# =================================================================================
# ===                      MAIN SCRIPT LOGIC                                    ===
# =================================================================================


def merge_models(args: argparse.Namespace) -> None:
    precision_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "float": torch.float32}
    dtype = precision_map[args.precision]
    save_dtype = precision_map[args.saving_precision]

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA device requested, but CUDA is not available.")
        return

    logger.info(f"Using device: {args.device} for tensor calculations.")
    logger.info(f"Prune Method: {args.prune_method} | Merge Method: {args.merge_method}")

    all_model_paths = [args.base_model] + args.models

    for model_path in all_model_paths:
        if not os.path.isfile(model_path):
            logger.error(f"Model file not found: {model_path}")
            return

    num_models = len(args.models)

    if args.task_weights and len(args.task_weights) != num_models:
        raise ValueError(
            f"Number of task weights ({len(args.task_weights)}) must match number of models to merge ({num_models})."
        )

    task_weights = torch.tensor(args.task_weights or [1.0] * num_models, dtype=dtype, device=args.device)

    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"Models to Merge: {args.models}")
    logger.info(f"Task Weights: {task_weights.tolist()}")

    logger.info("Scanning and standardizing model keys...")
    all_canonical_keys, model_key_maps = set(), []
    for model_path in all_model_paths:
        current_model_map = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for original_key in f.keys():
                canonical_key = get_canonical_key(original_key)
                all_canonical_keys.add(canonical_key)
                current_model_map[canonical_key] = original_key
        model_key_maps.append(current_model_map)

    merged_state_dict = {}

    logger.info("Starting merge process...")
    try:
        with contextlib.ExitStack() as stack:
            open_files = [
                stack.enter_context(safe_open(path, framework="pt", device="cpu")) for path in all_model_paths
            ]

            with torch.no_grad():
                for key in tqdm(sorted(list(all_canonical_keys)), desc="Merging Tensors", disable=args.verbose):
                    logger.debug(f"Processing key: {key}")
                    base_original_key = model_key_maps[0].get(key)
                    if not base_original_key:
                        logger.debug("  - Skipping key as it is not found in base model.")
                        continue

                    base_tensor = open_files[0].get_tensor(base_original_key).to(dtype=dtype)
                    base_tensor = base_tensor.to(args.device, non_blocking=True)
                    logger.debug(
                        f"  - Base tensor loaded. Shape: {base_tensor.shape}, Norm: {torch.linalg.norm(base_tensor.to(torch.float32)).item():.4f}"
                    )

                    deltas = []
                    for i in range(num_models):
                        model_index = i + 1
                        model_map = model_key_maps[model_index]

                        if key in model_map:
                            original_key = model_map[key]
                            tensor = open_files[model_index].get_tensor(original_key).to(args.device, dtype)
                            delta = tensor - base_tensor
                            deltas.append(delta)
                            logger.debug(
                                f"  - Delta {i} calculated. Shape: {delta.shape}, Norm: {torch.linalg.norm(delta.to(torch.float32)).item():.4f}"
                            )
                        else:
                            deltas.append(torch.zeros_like(base_tensor))
                            logger.debug(f"  - Delta {i} is zero as key was not found in model.")

                    merged_delta = compute_merged_delta(
                        deltas,
                        task_weights,
                        args.prune_method,
                        args.merge_method,
                        args.density,
                        args.majority_sign_method,
                    )
                    final_tensor = base_tensor + merged_delta
                    logger.debug(f"Merged delta norm: {torch.linalg.norm(merged_delta.to(torch.float32)).item():.4f}")
                    logger.debug(f"Final tensor norm: {torch.linalg.norm(final_tensor.to(torch.float32)).item():.4f}")

                    merged_state_dict[key] = final_tensor.to("cpu", save_dtype)
                    del base_tensor, deltas, merged_delta, final_tensor
                    torch.cuda.empty_cache()

        logger.info("Merge completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the tensor merging phase: {e}", exc_info=True)
        return

    output_file = args.output if args.output.endswith(".safetensors") else args.output + ".safetensors"
    logger.info(f"Saving merged model to {output_file}...")
    logger.info(f"Saving merged model to {output_file} with precision {args.saving_precision}...")

    metadata = None
    if args.save_metadata:
        logger.info("Applying merge parameters to file metadata.")
        metadata = {
            "base_model": args.base_model,
            "merged_models": json.dumps(args.models),
            "task_weights": json.dumps(task_weights.tolist()),
            "prune_method": args.prune_method,
            "merge_method": args.merge_method,
            "density": str(args.density),
            "majority_sign_method": args.majority_sign_method,
            "calculation_precision": args.precision,
            "saving_precision": args.saving_precision,
        }

    save_file(merged_state_dict, output_file, metadata=metadata)
    logger.info("Save completed successfully!")


if __name__ == "__main__":
    # Define the detailed help text to be displayed at the end
    epilog_text = """
How to Replicate Previous Named Methods:
-----------------------------------------
This script uses a modular approach ties and dare merging, as a base guide you can use the following flag combinations:

  TIES:
    --prune_method magnitude --merge_method disjoint_merge

  DARE_TIES:
    --prune_method random --merge_method disjoint_merge

  DARE_LINEAR:
    --prune_method random --merge_method sum

  Simple Average of Deltas:
    --prune_method none --merge_method mean
"""
    parser = argparse.ArgumentParser(
        prog="MergeFlux",
        description="A modular script for merging models using a combination of pruning and merging strategies.",
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter,  # This preserves the formatting of the epilog
    )

    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model.")
    parser.add_argument(
        "--models", type=str, nargs="+", required=True, help="Path(s) to the models to merge into the base."
    )
    parser.add_argument("--output", type=str, required=True, help="Path for the output merged model.")

    merge_group = parser.add_argument_group("Merging Strategy Arguments")
    merge_group.add_argument(
        "--prune_method",
        type=str,
        default="magnitude",
        choices=["none", "magnitude", "random"],
        help="The pruning method to apply to the deltas. (Default: magnitude)",
    )
    merge_group.add_argument(
        "--merge_method",
        type=str,
        default="disjoint_merge",
        choices=["sum", "mean", "disjoint_merge"],
        help="The method for combining the final deltas. (Default: disjoint_merge)",
    )
    merge_group.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="The fraction of parameters to keep for 'magnitude' or 'random' pruning.",
    )
    merge_group.add_argument(
        "--task_weights",
        type=float,
        nargs="+",
        help="Weights for each model in --models. Defaults to 1.0 each.",
    )
    merge_group.add_argument(
        "--majority_sign_method",
        type=str,
        default="total",
        choices=["total", "frequency"],
        help="Sign agreement method for 'disjoint_merge'. 'total' weights by magnitude, 'frequency' gives equal votes. (Default: total)",
    )

    tech_group = parser.add_argument_group("Technical Settings")
    tech_group.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    tech_group.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="Calculation precision during merge.",
    )
    tech_group.add_argument("--saving_precision", type=str, default="bf16", choices=["float", "fp16", "bf16"])
    tech_group.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging.")
    tech_group.add_argument(
        "--save-metadata", action="store_true", help="Save merge parameters to the output file metadata."
    )

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    merge_models(args)
