# mergeflux: Advanced FLUX Safetensors Model Merging

mergeflux is a powerful and memory-efficient Python script for merging multiple FLUX `safetensors` models. It goes beyond simple weighted averaging by providing granular control over the merge ratios at the block level, allowing for smooth, interpolated transitions between different models.

This tool is ideal for machine learning practitioners who want to combine the strengths of several FLUX models. For example, blending the initial layers of a base model with the final layers of a specialized, fine-tuned model.

## Key Features

- **Block-Level Interpolation**: Define different merge ratios for the start and end blocks of your model and the script will smoothly interpolate the weights for all blocks in between.
- **Memory Efficient**: Processes tensors one by one, using in-place operations to keep memory usage minimal, even when merging very large models.
- **Precision Control**: Perform calculations in `fp32`, `fp16`, or `bf16` and save the final model in your desired precision.
- **Flexible Ratio Control**: Specify a base ratio for non-transformer-block layers and custom ratios for the start and end of the block-wise merge.
- **Key-Aware**: Automatically handles models with slightly different key naming conventions (e.g., with or without the `model.diffusion_model.` prefix).
- **Reproducibility**: Optionally saves all merge parameters as metadata within the output `safetensors` file.
- Fast - depending on your IO speed.

## Requirements

The script requires InvokeAI or Python 3.8+ and the following libraries:

- `torch`
- `safetensors`
- `tqdm`

You can install them using pip:
```bash
pip install requirements.txt
```

Or place the script in a folder in the InvokeAI installation.  Open the InvokeAI launcher and start the dev console and run the script from the invoke environment. 

## Usage

The script is run from the command line. At its most basic, you provide two or more models and an output path.

```bash
python mergeflux.py --models model_A.safetensors model_B.safetensors --output merged_model.safetensors [OPTIONS]
```

### Command-Line Arguments

| Argument               | Type          | Description                                                                                             | Required |
|------------------------|---------------|---------------------------------------------------------------------------------------------------------|----------|
| `--models`             | `str` (list)  | Paths to the input `safetensors` models to merge.                                                       | **Yes** |
| `--output`             | `str`         | Path for the output merged model. `.safetensors` will be appended if not present.                       | **Yes** |
| `--base_ratios`        | `float` (list)| Ratios for non-block layers. Must sum to 1. Defaults to an equal share for each model.                  | No       |
| `--start_ratios`       | `float` (list)| Ratios for the start of the block transition. Must sum to 1. Defaults to `base_ratios`.                  | No       |
| `--end_ratios`         | `float` (list)| Ratios for the end of the block transition. Must sum to 1. Defaults to `base_ratios`.                    | No       |
| `--start_block`        | `int`         | The block number to start the ratio transition. Defaults to the first block found in the models.        | No       |
| `--end_block`          | `int`         | The block number to end the ratio transition. Defaults to the last block found in the models.           | No       |
| `--device`             | `str`         | Device for tensor calculations (`cpu` or `cuda`). Defaults to `cpu`.                                    | No       |
| `--precision`          | `str`         | Calculation precision (`float`, `fp16`, `bf16`). Defaults to `fp16`.                                     | No       |
| `--saving_precision`   | `str`         | Saving precision for the final model (`float`, `fp16`, `bf16`). Defaults to `fp16`.                      | No       |
| `--save-metadata`      | `flag`        | If set, saves the merge parameters to the output file's metadata.                                       | No       |
| `-v`, `--verbose`      | `flag`        | Enable verbose (DEBUG) logging for detailed script execution.                                           | No       |

## Examples

### Example 1: Simple 50/50 Merge

This command merges two models with equal weight for all tensors.

```bash
python mergeflux.py \
  --models model_A.safetensors model_B.safetensors \
  --output merged_50-50.safetensors
```
*(This is equivalent to setting `--base_ratios 0.5 0.5`)*


### Example 2: Three-Way Merge with Custom Base Ratios

Merge three models, giving 60% weight to Model A, 20% to B, and 20% to C.

```bash
python mergeflux.py \
  --models model_A.safetensors model_B.safetensors model_C.safetensors \
  --output merged_custom_base.safetensors \
  --base_ratios 0.6 0.2 0.2
```

### Example 3: Advanced Block-Based Merge (A/B Style)

This example creates a merge that starts as 100% Model A and ends as 100% Model B. This is useful for combining the early-stage generation of one model with the late-stage detail of another.

- The merge will start at block 4, using 100% of Model A (`--start_ratios 1.0 0.0`).
- It will end at block 12, using 100% of Model B (`--end_ratios 0.0 1.0`).
- Tensors in blocks before 4 will use the `start_ratios`.
- Tensors in blocks after 12 will use the `end_ratios`.
- Tensors for blocks 4 through 12 will have their ratios linearly interpolated.
- All non-block tensors (e.g., token embeddings) will be a 50/50 mix (`--base_ratios`).

```bash
python mergeflux.py \
  --models model_A.safetensors model_B.safetensors \
  --output merged_A-to-B.safetensors \
  --base_ratios 0.5 0.5 \
  --start_block 4 \
  --start_ratios 1.0 0.0 \
  --end_block 12 \
  --end_ratios 0.0 1.0 \
  --device cuda \
  --save-metadata
```

## How It Works

The script operates in three main phases:

1.  **Scanning & Planning**: It first scans all input models to discover every unique tensor key and the full range of transformer block numbers. It then creates a "merge plan" for every single tensor, pre-calculating the exact blend ratio based on the user's settings and the tensor's block location.

2.  **Tensor-by-Tensor Merge**: It iterates through the merge plan. For each tensor, it loads only the necessary data from the source files, performs the weighted addition on the specified device (`cpu` or `cuda`), and stores the result. This one-tensor-at-a-time approach ensures very low memory overhead. The ratio interpolation for a tensor in a block `b` is calculated as:
    $$ \text{ratio} = (1 - \alpha) \cdot \text{start\_ratios} + \alpha \cdot \text{end\_ratios} $$
    where $\alpha = (\text{b} - \text{start\_block}) / (\text{end\_block} - \text{start\_block})$.

3.  **Saving**: The final merged state dictionary is saved to a new `safetensors` file, optionally including the merge configuration in the file's metadata for future reference.

## License

This project is licensed under the MIT License.
