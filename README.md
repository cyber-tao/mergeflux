# MergeFlux: Advanced Safetensors Model Merging Utilities

MergeFlux is a collection of Python scripts designed for advanced merging of Flux `safetensors` models, offering fine-grained control and memory-efficient processing. These tools are ideal for machine learning practitioners looking to combine the strengths of multiple models in sophisticated ways.

## Scripts Included

1.  **`mergeflux.py`**: Merges multiple models using weighted averaging with block-level interpolation. Ideal for smoothly transitioning between models across different layers.
2.  **`mergeflux-delta.py`**: Implements advanced merging techniques like TIES and DARE, focusing on task arithmetic by merging deltas from a base model with modular pruning and merging strategies.

## Requirements

The script requires InvokeAI or Python 3.8+ and the following libraries:

- `torch`
- `safetensors`
- `tqdm`

You can install them using pip:
```bash
pip install .
```

Or place the scripts in a folder in the InvokeAI installation.  Open the InvokeAI launcher and start the dev console and run the script from the invoke environment. 

## Usage
Each script is run from the command line and has its own set of arguments.

### `mergeflux.py` Usage

This script is for general-purpose model merging with optional block-level ratio control.

```bash
python mergeflux.py --models model_A.safetensors model_B.safetensors --output merged_model.safetensors [OPTIONS]
```

#### `mergeflux.py` Arguments

| Argument | Type | Description | Required |
|---|---|---|---|
| `--models` | `str` (list) | Paths to the input `safetensors` models to merge. | **Yes** |
| `--output` | `str` | Path for the output merged model. | **Yes** |
| `--base_ratios` | `float` (list)| Ratios for non-block layers. Must sum to 1. Defaults to an equal share for each model. | No |
| `--start_ratios` | `float` (list)| Ratios for the start of the block transition. Must sum to 1. Defaults to `base_ratios`. | No |
| `--end_ratios` | `float` (list)| Ratios for the end of the block transition. Must sum to 1. Defaults to `base_ratios`. | No |
| `--start_block` | `int` | The block number to start the ratio transition. Defaults to the first block found. | No |
| `--end_block` | `int` | The block number to end the ratio transition. Defaults to the last block found. | No |
| `--device` | `str` | Device for tensor calculations (`cpu` or `cuda`). | No |
| `--precision` | `str` | Calculation precision (`float`, `fp16`, `bf16`). | No |
| `--saving_precision`| `str` | Saving precision for the final model (`float`, `fp16`, `bf16`). | No |
| `--save-metadata` | `flag` | If set, saves the merge parameters to the output file's metadata. | No |
| `-v`, `--verbose` | `flag` | Enable verbose (DEBUG) logging for detailed script execution. | No |

---

### `mergeflux-delta.py` Usage

This script is for advanced merging based on task arithmetic (model differences).

```bash
python mergeflux-delta.py --base_model base.safetensors --models model_A.safetensors --output merged_delta.safetensors [OPTIONS]
```

#### `mergeflux-delta.py` Arguments

| Argument | Type | Description | Required |
|---|---|---|---|
| `--base_model` | `str` | Path to the base model. | **Yes** |
| `--models` | `str` (list) | Path(s) to the models to merge into the base. | **Yes** |
| `--output` | `str` | Path for the output merged model. | **Yes** |
| `--prune_method` | `str` | Pruning method for deltas (`none`, `magnitude`, `random`). | No |
| `--merge_method` | `str` | Method for combining deltas (`sum`, `mean`, `disjoint_merge`).| No |
| `--density` | `float` | Fraction of parameters to keep for 'magnitude' or 'random' pruning. | No |
| `--task_weights` | `float` (list)| Weights for each model in `--models`. | No |
| `--majority_sign_method` | `str` | Sign agreement for 'disjoint_merge' (`total` or `frequency`). | No |
| `--device` | `str` | Device for calculations (`cpu` or `cuda`). | No |
| `--precision` | `str` | Calculation precision (`float`, `fp16`, `bf16`). | No |
| `--saving_precision`| `str` | Saving precision (`float`, `fp16`, `bf16`). | No |
| `--save-metadata` | `flag` | Save merge parameters to metadata. | No |
| `-v`, `--verbose` | `flag` | Enable verbose (DEBUG) logging. | No |

## Examples

### `mergeflux.py` Examples

#### Example 1: Simple 50/50 Merge

This command merges two models with equal weight for all tensors.

```bash
python mergeflux.py \
  --models model_A.safetensors model_B.safetensors \
  --output merged_50-50.safetensors
```

#### Example 2: Advanced Block-Based Merge (A/B Style)

This example creates a merge that starts as 100% Model A and ends as 100% Model B.

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

---

### `mergeflux-delta.py` Examples

This script can replicate several named merging methodologies through different flag combinations.

#### Example 1: TIES Merging

This method resolves conflicting changes between models by pruning less significant parameter changes and then merging the remaining ones that agree in sign.

```bash
# To replicate TIES use:
python mergeflux-delta.py \
  --base_model base_model.safetensors \
  --models fine-tuned_A.safetensors fine-tuned_B.safetensors \
  --output ties_merged.safetensors \
  --prune_method magnitude \
  --merge_method disjoint_merge \
  --density 0.5
```

#### Example 2: DARE (Linear) Merging

This method randomly prunes the model deltas and then simply adds them together, avoiding the sign agreement step of TIES.

```bash
# To replicate DARE_LINEAR use:
python mergeflux-delta.py \
  --base_model base_model.safetensors \
  --models fine-tuned_A.safetensors fine-tuned_B.safetensors \
  --output dare_linear_merged.safetensors \
  --prune_method random \
  --merge_method sum \
  --density 0.5
```

## How It Works

### `mergeflux.py`

The script operates in three main phases:

1.  **Scanning & Planning**: It first scans all input models to discover every unique tensor key and the full range of transformer block numbers. It then creates a "merge plan" for every single tensor, pre-calculating the exact blend ratio based on the user's settings and the tensor's block location.

2.  **Tensor-by-Tensor Merge**: It iterates through the merge plan. For each tensor, it loads only the necessary data from the source files, performs the weighted addition on the specified device (`cpu` or `cuda`), and stores the result. The ratio interpolation for a tensor in a block `b` is calculated as:
    $$\text{ratio} = (1 - \alpha) \cdot \text{start\_ratios} + \alpha \cdot \text{end\_ratios}$$
    where $\alpha = (\text{b} - \text{start\_block}) / (\text{end\_block} - \text{start\_block})$.

3.  **Saving**: The final merged state dictionary is saved to a new `safetensors` file, optionally including the merge configuration in the file's metadata.

### `mergeflux-delta.py`

This script uses task arithmetic to perform the merge:

1.  **Delta Calculation**: For each tensor, it calculates the "delta" or difference between the fine-tuned models (`--models`) and the `--base_model`. This delta represents the changes introduced during fine-tuning.

2.  **Modular Merging**: It processes these deltas using a combination of a pruning and a merging function:
    * **Pruning**: Each delta is pruned according to the `--prune_method`.
    * **Weighting**: The pruned deltas are weighted by the `--task_weights`.
    * **Merging**: The final weighted deltas are combined using the `--merge_method`.

3.  **Final Application**: The computed merged delta is added back to the base model's tensors to produce the final, merged model, which is then saved.

## License

This project is licensed under the MIT License.