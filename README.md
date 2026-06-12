## [***EMNLP 2025 Main Conference***] Mitigating Hallucinations in Vision-Language Models through Image-Guided Head Suppression

<div style='display:flex; gap: 0.25rem; '>
<a href='https://opensource.org/licenses/MIT'><img src=https://img.shields.io/badge/License-MIT-g.svg></a>
<a href='https://arxiv.org/abs/2505.16411'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>

This repository provides detailed instruction for reproducing our results reported in the paper. Other than our method SPIN, we also support PAI, DAMRO, OPERA, and VCD.

This fork additionally includes **CB-Spectral (Counteractive Balance)**, an extension that replaces SPIN's binary head pruning with a continuous, spectral-feature-driven head modulation. See [Section 5](#5-cb-spectral-experiment-counteractive-balance) below.

## Environment Setup

For experiments running with LLaVA-1.5 (7B or 13B), Minigpt4, and Shikra:
```
conda env create -f environment.yml
conda activate spin
```

For experiments running with Qwen-VL-Chat:
```
Please follow the official instruction of Qwen-VL to set up the environment.
```

## Prepare the Models Weights and the Scripts

Please refer to the corresponding official repositories of LLaVA-1.5, Minigpt4, Shikra, and Qwen-VL to download the model weights, and modify the necessary path (Minigpt4 and Shikra).

Go to our [ModelLoader](./model_loader.py#L171) class (under model_loader.py), and put your actual path.

Download the official MMHal Bench dataset, and modify the path in [load_image](./utils.py#L158) (under utils.py) to the folder contains the MMHal Bench images.

We directly changed the source code to implement OPERA and VCD. If you also want to run experiments with those two algorithms, copy the [txt file](./transformers_utils.txt) we provide, and paste to ```transformers/generation/utils.py``` under the transformers package in your environment.

## Quick Start

Followings are the commands to run evaluation on CHAIR, POPE, as well as the MMHal Bench. If you want to try the GPT-4o evaluation, make sure to apply for an API key.

### Arguments

When you are using Qwen-VL-Chat, provide `--model-path` instead of `--model`.

| Argument           | Example         | Description                                                          |
|--------------------|-----------------|----------------------------------------------------------------------|
| `--model`          | `llava-1.5`     | Currently we support: `minigpt4`, `llava-1.5`, `shikra`.             |
| `--model-path`     | `/path/to/Qwen-VL-Chat` | Path to `Qwen-VL-Chat` model                                         |
| `--data-path`      | `/path/to/COCO` | Path to `coco/val2014/`.                                             |
| `--llava-size`     | `7b`            | To use `LLaVA-1.5-7B` or `LLaVA-1.5-13B`.                            |
| `--pope-type`      | `random`        | Type of POPE Evaluation: `random`, `popular`, or `adversarial`.      |
| `--start-layer`    | `0`             | The starting layer of applying SPIN.                                 |
| `--end-layer`      | `32`            | The ending layer of applying SPIN.                                   |
| `--use-spin`       | `-`             | Activate SPIN.                                                       |
| `--routed-heads`   | `0.95`          | Ratio of active heads (1 - ratio of suppressed heads). Default: 0.95 |
| `--small-num-mask` | `0.05`          | The scaling factor for SPIN. Default: None.                          |
| `--repetition-penalty` | `1.1`           | Set to 1.1 when using Minigpt4 on CHAIR Evaluation. Default: 1.      |
| `--use-cb`         | `-`             | Activate CB-Spectral (overrides `--use-spin` when both are set).     |
| `--spectral-mode`  | `fft`           | Spectral feature mode: `fft` (default), `power`, `block`, `none`.    |
| `--suppression-coeff` | `0.3`        | Max suppression strength for hallucination-prone heads.              |
| `--reinforcement-coeff` | `0.15`    | Max reinforcement strength for faithful heads.                       |
| `--temperature`    | `0.1`           | Sigmoid temperature for the soft gate. Smaller = sharper transition. |
| `--n-calib-examples` | `64`         | Number of POPE-held-out examples used to fit per-layer thresholds.   |
| `--calib-pope-type` | `random`       | Which POPE split to draw calibration examples from.                  |
| `--output-path`    | `output.jsonl`  | Your output path.                                                    |

### 1. CHAIR Evaluation

For LLaVA-1.5, Minigpt4, and Shikra:

```bash
CUDA_VISIBLE_DEVICES=0 python chair_eval.py \
    --model llava-1.5 \
    --data-path /path/to/COCO \
    --llava-size 7b \
    --start-layer 0 \
    --end-layer 32 \
    --use-spin \
    --routed-heads 0.95 \
    --small-num-mask 0.08 \
    --repetition-penalty 1.0 \
    --output-path chair_output.jsonl
```

For Qwen-VL-Chat:

```bash
CUDA_VISIBLE_DEVICES=0 python chair_eval_qwen.py \
    --model-path /path/to/Qwen-VL-Chat \
    --data-path /path/to/COCO \
    --start-layer 0 \
    --end-layer 20 \
    --use-spin \
    --routed-heads 0.7 \
    --small-num-mask 0.08 \
    --output-path qwen_chair_output.jsonl
```

Compute the CHAIR scores with the generated jsonl file. Check `chair.py` for more detailed information. Before that, install `nltk=3.8.1`.

```bash
python chair.py --cap_file /path/to/jsonl
```

### 2. POPE Evaluation

For LLaVA-1.5, Minigpt4, and Shikra:

```bash
CUDA_VISIBLE_DEVICES=0 python pope_chat_eval.py  \
    --model llava-1.5 \
    --data-path /path/to/COCO \
    --pope-type random \
    --llava-size 7b \
    --start-layer 0 \
    --end-layer 32 \
    --use-spin \
    --routed-heads 0.8 \
    --small-num-mask 0.1 \
    --output-path pope_output_random.jsonl
```

Compute the POPE results with the generated jsonl file.

```bash
python pope_ans.py --ans-file pope_output_random.jsonl
```

### 3. MMHal Bench

For LLaVA-1.5, Minigpt4, and Shikra:

```bash
CUDA_VISIBLE_DEVICES=0 python mmhal_eval.py \
    --input /path/to/MMHal-Bench/response_template.json \
    --output mmhal_output.json \
    --model llava-1.5 \
    --llava-size 13b \
    --start-layer 0 \
    --end-layer 20 \
    --use-spin \
    --routed-heads 0.9
```

For Qwen-VL-Chat:

```bash
CUDA_VISIBLE_DEVICES=0 python mmhal_eval_qwen.py \
    --input /path/to/MMHal-Bench/response_template.json \
    --output qwen_mmhal_output.json \
    --model-path /path/to/Qwen-VL-Chat \
    --start-layer 0 \
    --end-layer 20 \
    --use-spin \
    --routed-heads 0.7 \
    --small-num-mask 0.08
```

Please follow the official instruction of MMHal-Bench to evaluate the generated responses.

### 4. GPT-4o Assisted Evaluation

Here, the caption files are the jsonl files generated by CHAIR evaluation. Please provide both of the vanilla model (first caption file) and SPIN (second caption file). You can swap the position.

```bash
python gpt_4o_eval.py \
    --cap-file-first /path/to/first/jsonl \
    --cap-file-second /path/to/second/jsonl \
    --data-path /path/to/COCO \
    --api-key your_api_key \
    --output gpt_4o_response.jsonl
```

### 5. CB-Spectral Experiment (Counteractive Balance)

**What it is.** A drop-in upgrade over SPIN that replaces the binary top-k head pruning with a continuous per-head modulation factor `α ∈ [ε, 1 + amp_max(l)]` derived from spectral features of the attention map. The shift is:

- **SPIN** decides which heads to *zero out* via `topk` over summed image-attention.
- **CB-Spectral** decides how strongly to *scale* each head via a soft sigmoid gate on a spectral feature (`λ_max` proxy). Heads with high-frequency / concentrated attention are reinforced; heads with diffuse, low-frequency attention are suppressed. All heads contribute, with varying strength.

**Three spectral modes** (selectable via `--spectral-mode`):

| Mode | Cost | Best for |
|---|---|---|
| `fft` (default) | O(n log n), 1D rFFT high-freq energy ratio | Spatial-token models (LLaVA, Qwen-VL) |
| `power` | O(k·n²), power iteration on circulant Laplacian | Higher-fidelity signal, ~3-5× more compute |
| `block` | O((n/G)² log(n/G)), coarsened 2D FFT | When patch boundaries are semantically meaningful |

**Backbone scope.** CB-Spectral is only wired for `llava-1.5` and `qwen-vl`. For other backbones (`minigpt4`, `shikra`), passing `--use-cb` will emit a warning and silently fall back to legacy SPIN (`spectral_mode="none"`).

**Calibration.** On every eval run, the script refits per-layer `τ_weak`, `τ_strong` thresholds on `--n-calib-examples` POPE examples (binary label = reference answer `no` vs `yes` acts as a hallucination-likeness proxy). The fit uses an L2 smoothing penalty (`l2_smoothing=1e-3`) to prevent overfitting. The fitted thresholds are passed into `llama_modify_cb` and live only for the duration of the run.

**Hardware.** CB-Spectral adds ~200-300 MB of GPU memory over baseline SPIN (for capture buffers + spectral feature tensors during the calibration pass). Recommended:

| Model | VRAM |
|---|---|
| LLaVA-1.5-7B | 24 GB (RTX 4090 / A6000 / A100-40) |
| LLaVA-1.5-13B | 40 GB (A100-40, tight) or 80 GB (A100-80) |

#### 5a. CHAIR Evaluation with CB-Spectral

For LLaVA-1.5:

```bash
CUDA_VISIBLE_DEVICES=0 python chair_eval.py \
    --model llava-1.5 \
    --data-path /path/to/COCO \
    --llava-size 7b \
    --start-layer 0 \
    --end-layer 32 \
    --use-cb \
    --spectral-mode fft \
    --n-calib-examples 64 \
    --output-path chair_cb_fft.jsonl
```

To compare against `power` and `block` modes, just swap `--spectral-mode`. For the original SPIN baseline, use `--use-spin` instead of `--use-cb`.

#### 5b. POPE Evaluation with CB-Spectral

```bash
CUDA_VISIBLE_DEVICES=0 python pope_chat_eval.py \
    --model llava-1.5 \
    --data-path /path/to/COCO \
    --pope-type random \
    --llava-size 7b \
    --start-layer 0 \
    --end-layer 32 \
    --use-cb \
    --spectral-mode fft \
    --n-calib-examples 64 \
    --output-path pope_cb_fft_random.jsonl
```

For Qwen-VL-Chat, use the corresponding `*_qwen.py` script with the same `--use-cb` / `--spectral-mode` flags.

#### 5c. MMHal Bench with CB-Spectral

```bash
CUDA_VISIBLE_DEVICES=0 python mmhal_eval.py \
    --input /path/to/MMHal-Bench/response_template.json \
    --output mmhal_cb_fft.json \
    --model llava-1.5 \
    --llava-size 13b \
    --start-layer 0 \
    --end-layer 20 \
    --use-cb \
    --spectral-mode fft
```

#### 5d. Full Ablation Grid (8 cells)

To reproduce the spectral_mode × backbone ablation (`{fft, power, block, none}` × `{llava-1.5, qwen-vl}`):

```bash
python run_ablation_grid.py \
    --data-path /path/to/COCO \
    --llava-size 7b \
    --results-dir runs/ablation \
    --all
```

Per-cell results land in `runs/ablation/chair_<mode>_<backbone>.jsonl`, with a `summary.csv` aggregating wall-clock and caption counts.

If you OOM, prepend `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or set `--max-tokens 256` to halve the KV cache.

## Acknowledgement
The partial code for this repo is taken from [PAI](https://github.com/LALBJ/PAI), [OPERA](https://github.com/shikiw/OPERA), and [VCD](https://github.com/DAMO-NLP-SG/VCD)

## Citation
If you find this repo useful for your research, please consider citing the following work:
```
@misc{sarkar2025spin,
      title={Mitigating Hallucinations in Vision-Language Models through Image-Guided Head Suppression}, 
      author={Sreetama Sarkar and Yue Che and Alex Gavin and Peter A. Beerel and Souvik Kundu},
      year={2025},
      eprint={2505.16411},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.16411}, 
}
