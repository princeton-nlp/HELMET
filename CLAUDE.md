# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HELMET (How to Evaluate Long-context Language Models Effectively and Thoroughly) is a benchmark for evaluating long-context LLMs across seven task categories: Recall, RAG, Re-ranking, Citation, LongQA, Summarization, and In-Context Learning. Published at ICLR 2025.

## Setup & Installation

```bash
pip install -r requirements.txt
# or with uv:
uv sync
```

Requires Python 3.12+. Download benchmark data (34GB):
```bash
bash scripts/download_data.sh
```

## Running Evaluations

```bash
# Basic evaluation with a config
python eval.py --config configs/{task}.yaml --model_name_or_path {model} --output_dir {dir}

# With vLLM backend
python eval.py --config configs/recall.yaml --model_name_or_path meta-llama/Llama-2-7b --use_vllm

# API models (OpenAI, Anthropic, Gemini)
python eval.py --config configs/longqa.yaml --model_name_or_path gpt-4o --output_dir output/gpt-4o

# SLURM-based batch evaluation
bash scripts/run_eval_slurm.sh
```

Key flags: `--use_vllm`, `--use_sglang`, `--use_vllm_serving`, `--use_tgi_serving`, `--endpoint_url`, `--overwrite`, `--debug`, `--count_tokens`, `--thinking N` (for reasoning models).

## GPT-4o Model-Based Evaluation

LongQA and Summarization tasks require GPT-4o judging after generation:
```bash
python scripts/eval_gpt4_longqa.py --input_file {results.json}
python scripts/eval_gpt4_summ.py --input_file {results.json}
```

## Collecting Results

```bash
python scripts/collect_results.py
```

## Architecture

### Core Pipeline (`eval.py`)

1. Parse args from YAML config + CLI overrides (`arguments.py`)
2. Load model once via `load_LLM()` (`model_utils.py`)
3. For each dataset: load data → prepare inputs → generate → post-process → evaluate → save

### Key Modules

- **`data.py`** — Dataset loaders. Each returns `{data, prompt_template, user_template, system_template, post_process}`. The `post_process` callable computes task-specific metrics. Loaders: `load_qa`, `load_ruler`, `load_icl`, `load_msmarco_rerank`, `load_multi_lexsum`, `load_narrativeqa`, etc.

- **`model_utils.py`** — LLM abstraction layer. Base `LLM` class with `prepare_inputs()` and `generate()` methods. Implementations: `HFModel`, `OpenAIModel`, `AnthropicModel`, `GeminiModel`, `TogetherModel`, `VLLMModel`, `TgiVllmModel`, `SGLangModel`. OpenAI/Anthropic use batch APIs for cost savings.

- **`utils.py`** — Evaluation metrics (EM, F1, ROUGE-L, NDCG, MAP, recall, precision, MRR) and output parsing utilities.

- **`eval_alce.py`** — Specialized evaluation for citation/ALCE tasks.

- **`arguments.py`** — `DatasetOptions` dataclass. Config fields like `datasets`, `input_max_length`, `generation_max_length` are comma-separated strings that get zipped together to define multiple evaluation runs.

### Configuration System

YAML configs in `configs/` specify dataset options (datasets, lengths, test files, demo files, shots). Subdirectories (`configs/icl/`, `configs/rag/`) hold per-dataset configs. The `_short` suffix variants use smaller context lengths.

Configs are generated programmatically via `scripts/generate_configs.py`.

### Input Truncation

Uses Llama-2 tokenizer as a reference tokenizer for consistent truncation across models. Context is truncated to fit `input_max_length` while preserving instructions and demos.

### Output Format

Results saved as JSON: `{dataset}_{tag}_{testname}_in{length}_size{samples}_shots{n}_...json` containing args, per-item predictions, metrics, throughput, and memory usage. A `.score` file contains summary metrics.
