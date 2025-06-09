# V1 - Skunkworxdark 2025
import argparse
import contextlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# --- Standard Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def get_block_number(key: str) -> Optional[int]:
    """
    Parses a tensor key to find if it belongs to a single or double block and returns the block number.
    Returns None if the key is not part of a numbered block.
    """
    # Regex to find 'single_blocks.XX.' or 'double_blocks.XX.'
    match = re.search(r"(single_blocks|double_blocks)\.(\d+)\.", key)
    if match:
        return int(match.group(2))
    return None


def get_ratios_for_key(key: str, args: argparse.Namespace, block_info: Dict[str, int]) -> List[float]:
    """
    Determines the correct blend ratios for a given tensor key based on its block number.
    """
    block_num = get_block_number(key)
    if block_num is None:
        return args.base_ratios  # Use base for everything not in a block

    # Use start_ratios for any blocks before the transition starts
    if block_num < block_info["start_block"]:
        return args.start_ratios

    # Use end_ratios for any blocks after the transition ends
    if block_num > block_info["end_block"]:
        return args.end_ratios

    # Avoid division by zero if the range is a single block
    if block_info["start_block"] == block_info["end_block"]:
        return args.start_ratios

    # Calculate the progress (alpha) through the transition, from 0.0 to 1.0
    alpha = (block_num - block_info["start_block"]) / (block_info["end_block"] - block_info["start_block"])

    # Linearly interpolate between the start and end ratios
    start_ratios = torch.tensor(args.start_ratios, dtype=torch.float32)
    end_ratios = torch.tensor(args.end_ratios, dtype=torch.float32)
    interpolated_ratios = (1 - alpha) * start_ratios + alpha * end_ratios

    return interpolated_ratios.tolist()


def get_canonical_key(original_key: str) -> str:
    """
    Transforms a model's tensor key into a standardized, canonical format.
    """
    key = original_key
    if key.startswith("model.diffusion_model."):
        key = key.removeprefix("model.diffusion_model.")
    return key


def merge_models(args: argparse.Namespace) -> None:
    """
    Merges multiple safetensors models using a memory-efficient,
    tensor-by-tensor weighted average with block-level ratio control.
    """

    # --- Initial Validation ---
    precision_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "float": torch.float32}
    dtype = precision_map[args.precision]
    save_dtype = precision_map[args.saving_precision]

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA device requested, but CUDA is not available.")
        return
    logger.info(f"Using device: {args.device} for tensor calculations.")

    for model_path in args.models:
        if not os.path.isfile(model_path):
            logger.error(f"Model file not found: {model_path}")
            return

    # Ratio Validation
    def _validate_and_get_ratios(
        ratios: Optional[List[float]], name: str, num_models: int, default_ratios: List[float]
    ) -> List[float]:
        """Helper to validate and return a set of ratios."""
        if ratios is None:
            logger.info(f"No {name} specified, defaulting to: {default_ratios}")
            return default_ratios.copy()

        if len(ratios) != num_models:
            raise ValueError(f"Number of models ({num_models}) and provided {name} ({len(ratios)}) must match.")
        if not torch.isclose(torch.tensor(sum(ratios)), torch.tensor(1.0)):
            raise ValueError(f"Provided {name} {ratios} do not sum to 1.0 (sum: {sum(ratios)}).")

        logger.info(f"Using user-provided {name}: {ratios}")
        return ratios

    try:
        num_models = len(args.models)
        base_default = [1.0 / num_models] * num_models
        args.base_ratios = _validate_and_get_ratios(args.base_ratios, "base_ratios", num_models, base_default)
        args.start_ratios = _validate_and_get_ratios(args.start_ratios, "start_ratios", num_models, args.base_ratios)
        args.end_ratios = _validate_and_get_ratios(args.end_ratios, "end_ratios", num_models, args.base_ratios)
    except ValueError as e:
        logger.error(f"Ratio validation failed: {e}")
        return

    # --- Phase 1: setup
    logger.info("Scanning models, mapping keys, and detecting block range...")
    all_canonical_keys = set()
    model_key_maps: List[Dict[str, str]] = []
    all_block_nums = set()

    # Key Discovery
    for model_path in args.models:
        current_model_map: Dict[str, str] = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for original_key in f.keys():
                canonical_key = get_canonical_key(original_key)
                all_canonical_keys.add(canonical_key)
                current_model_map[canonical_key] = original_key
                block_num = get_block_number(canonical_key)
                if block_num is not None:
                    all_block_nums.add(block_num)
        model_key_maps.append(current_model_map)

    # Block Range Detection
    block_info: Dict[str, int] = {}
    if not all_block_nums:
        logger.warning("No blocks found. Block-specific ratios will not be applied.")
        block_info = {"start_block": -1, "end_block": -1}
    else:
        min_block, max_block = min(all_block_nums), max(all_block_nums)
        block_info["start_block"] = args.start_block if args.start_block is not None else min_block
        block_info["end_block"] = args.end_block if args.end_block is not None else max_block
        logger.info(f"Detected blocks from {min_block} to {max_block}.")
        logger.info(f"Transitioning ratios from block {block_info['start_block']} to {block_info['end_block']}.")

    # Build Merge Plan
    logger.info("Pre-calculating merge plan for all tensors...")
    merge_plan: Dict[str, Dict[str, Any]] = {}
    for key in tqdm(sorted(list(all_canonical_keys)), desc="Building Merge Plan"):
        tensors_to_process = []
        relevant_ratio_sum = 0.0
        ratios = get_ratios_for_key(key, args, block_info)

        for i, model_map in enumerate(model_key_maps):
            if key in model_map and ratios[i] > 1e-6:
                tensors_to_process.append({"model_index": i, "original_key": model_map[key], "ratio": ratios[i]})
                relevant_ratio_sum += ratios[i]

        if not tensors_to_process:
            logger.warning(f"Key '{key}' discovered but received a zero or negative ratio in all models. Skipping.")
            continue

        if len(tensors_to_process) == 1:
            merge_plan[key] = {"op_type": "copy", "source": tensors_to_process[0]}
        else:
            # normalise ratios
            for task in tensors_to_process:
                task["ratio"] /= relevant_ratio_sum
            merge_plan[key] = {"op_type": "merge", "sources": tensors_to_process}

        if key in merge_plan:
            logger.debug(f"Plan for key '{key}': {merge_plan[key]}")

    # --- Phase 2: Tensor-by-Tensor Merging ---
    logger.info(f"Starting merge of {len(merge_plan)} planned tensors...")
    merged_state_dict: Dict[str, torch.Tensor] = {}

    try:
        with contextlib.ExitStack() as stack:
            open_files = [
                stack.enter_context(safe_open(path, framework="pt", device=args.device)) for path in args.models
            ]

            with torch.no_grad():
                for key, plan in tqdm(merge_plan.items(), desc="Merging Tensors"):
                    logger.debug(f"Processing key '{key}' with op_type '{plan['op_type']}'")

                    if plan["op_type"] == "copy":
                        task = plan["source"]
                        model_handle = open_files[task["model_index"]]
                        tensor = model_handle.get_tensor(task["original_key"])
                        merged_state_dict[key] = tensor.to(save_dtype)
                    elif plan["op_type"] == "merge":
                        # First tensor is copied and scaled to initialize the accumulator.
                        first_task, other_tasks = plan["sources"][0], plan["sources"][1:]

                        model_handle = open_files[first_task["model_index"]]
                        accumulator = model_handle.get_tensor(first_task["original_key"]).to(dtype)
                        accumulator.mul_(first_task["ratio"])  # In-place multiplication

                        # Subsequent tensors are added in-place using a fused add-multiply operation.
                        for task in other_tasks:
                            model_handle = open_files[task["model_index"]]
                            tensor = model_handle.get_tensor(task["original_key"])
                            torch.add(accumulator, tensor.to(dtype), alpha=task["ratio"], out=accumulator)

                        merged_state_dict[key] = accumulator.to(save_dtype)

        logger.info("Merge completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the tensor merging phase: {e}", exc_info=True)
        return

    # --- Phase 3: Saving the Merged Model ---
    output_file = args.output if args.output.endswith(".safetensors") else args.output + ".safetensors"
    logger.info(f"Saving merged model to {output_file} with precision {args.saving_precision}...")

    metadata: Optional[Dict[str, str]] = None
    if args.save_metadata:
        logger.info("applying merge parameters to file metadata.")
        metadata = {
            "merged_from": json.dumps(args.models),
            "base_ratios": json.dumps(args.base_ratios),
            "start_ratios": json.dumps(args.start_ratios),
            "end_ratios": json.dumps(args.end_ratios),
            "start_block": str(block_info["start_block"]),
            "end_block": str(block_info["end_block"]),
            "calculation_precision": args.precision,
            "saving_precision": args.saving_precision,
        }

    try:
        save_file(merged_state_dict, output_file, metadata=metadata)
        logger.info("Save completed successfully!")
    except Exception as e:
        logger.error(f"Failed to save the final model: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple FLUX-style safetensors models with advanced block-level ratio control."
    )
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Paths to the models to merge.")
    parser.add_argument("--output", type=str, required=True, help="Path for the output merged model.")
    parser.add_argument(
        "--base_ratios",
        type=float,
        nargs="+",
        help="Base ratios for non-block layers. Sum must be 1. Defaults to equal ratios.",
    )
    parser.add_argument(
        "--start_ratios",
        type=float,
        nargs="+",
        help="Ratios for the start of the transition. Sum must be 1. Defaults to base_ratios.",
    )
    parser.add_argument(
        "--end_ratios",
        type=float,
        nargs="+",
        help="Ratios for the end of the transition. Sum must be 1. Defaults to base_ratios.",
    )
    parser.add_argument(
        "--start_block",
        type=int,
        help="Block number to start the ratio transition. Defaults to the first detected block.",
    )
    parser.add_argument(
        "--end_block", type=int, help="Block number to end the ratio transition. Defaults to the last detected block."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for tensor calculations."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["float", "fp16", "bf16"],
        help="Calculation precision during merge.",
    )
    parser.add_argument(
        "--saving_precision",
        type=str,
        default="fp16",
        choices=["float", "fp16", "bf16"],
        help="Precision for saving the final model.",
    )
    parser.add_argument(
        "--save-metadata", action="store_true", help="Save merge parameters to the output file metadata."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging for detailed script execution.",
    )

    args = parser.parse_args()

    # Set logging level based on the argument
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    merge_models(args)
