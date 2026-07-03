# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

PPDPP (Proactive Planning for Dialogue Policy) is a reinforcement learning framework for training dialogue strategy policies. A BERT/RoBERTa policy network selects dialogue acts; three LLM agents (system, user, critic) simulate the conversation environment and compute rewards.

## Commands

### Supervised Fine-Tuning (SFT) — Phase 1
```bash
cd PPDPP
python sft.py --do_train --do_eval \
  --data_name esc \          # esc | cima | cb
  --model_name roberta \
  --model_name_or_path roberta-large \
  --data_dir ../data \
  --cache_dir /path/to/plm_cache \
  --gpu "0 1" \
  --per_gpu_train_batch_size 8 \
  --overwrite_output_dir
```
SFT checkpoints are saved to `sft/{data_name}/{model_name}/best_checkpoint/`.

### RL Training — Phase 2
```bash
python run.py --do_train --do_eval \
  --data_name esc \
  --system vicuna \          # vicuna | chatgpt | llama2
  --user vicuna \
  --critic vicuna \
  --model_path /path/to/vicuna_hf/7B \
  --model_name roberta \
  --model_name_or_path roberta-large \
  --cache_dir /path/to/plm_cache \
  --sft_dir sft \
  --max_steps 10 \
  --sample_times 100
```

### Evaluation Only
```bash
python run.py --do_eval \
  --data_name esc --system vicuna --user vicuna --critic vicuna \
  --load_rl_epoch 5          # load saved RL checkpoint at epoch 5
```

### OpenAI / ChatGPT
Set `YOUR_API_KEY` in `env.py` before using `--system chatgpt`, `--user chatgpt`, or `--critic chatgpt`.

## Architecture

### Two-Phase Training

**Phase 1 — SFT (`sft.py`):** Supervised pre-training of the policy on human-annotated dialogue acts. Trains `PPDPP` (BERT/RoBERTa + linear head) with cross-entropy loss. Best checkpoint by macro-F1 goes to `sft/{data_name}/{model_name}/best_checkpoint/`.

**Phase 2 — RL (`run.py`):** REINFORCE algorithm. Each episode: the policy selects an action (dialogue act), the `Env` simulates one full conversation turn via LLMs, the critic LLM scores the result as the reward, and the policy is updated via policy gradient. RL checkpoints saved to `tmp/{data_name}/RL-agent/`.

### Key Modules

- **`agent.py` — `PPDPP`**: The policy network. BERT/RoBERTa encodes truncated conversation history (most-recent turns first, up to `max_seq_length`), then a linear classifier scores each dialogue act. `select_action` samples during training (stochastic), argmax during eval. `optimize_model` runs one REINFORCE update.

- **`env.py` — `Env`**: The environment. `reset()` initializes a conversation from the dataset. `step(action)` calls three LLMs sequentially: (1) system agent generates a response using the selected act, (2) user agent responds, (3) critic agent evaluates and returns a scalar reward. Done condition is dataset-specific.

- **`prompt.py`**: Dialogue act dictionaries (`ESConvAct`, `CIMAAct`, `CBAct`) and prompt-formatting functions for each LLM backend (Vicuna, Llama2, ChatGPT) and each role (system, user, critic).

- **`data_reader.py`**: Tokenizes and caches features for SFT. Cache key includes data name, split, model name, and max sequence length. Cached as `.pkl` files in `--data_dir`.

- **`utils.py`**: `load_dataset` reads raw `.txt` files from `../data/` relative to `PPDPP/`; `TMP_DIR` maps dataset names to output directories.

### Datasets and Roles

| Dataset | System Role | User Role | Task |
|---------|-------------|-----------|------|
| `esc` | Therapist | Patient | Emotional support counselling |
| `cima` | Teacher | Student | Italian language tutoring |
| `cb` | Buyer | Seller | Price bargaining (CraigslistBargain) |

Raw data lives in `data/{esc,cima,cb}-{train,valid,test}.txt` (one JSON object per line).

### Output Layout

```
tmp/{data_name}/
  RL-agent/{filename}-epoch-{N}/   # RL policy checkpoints
  eval_result/                     # eval metrics and dialogue records
sft/{data_name}/{model_name}/
  best_checkpoint/                 # SFT checkpoint loaded by RL phase
```

## Important Details

- The `utils.load_dataset` path is hardcoded as `../data/` relative to the script working directory — run scripts from inside `PPDPP/`.
- Vicuna and Llama2 share the same model object (`vicuna_model`/`vicuna_tokenizer`); the `Env` loads it once during `train` mode and reuses it in `test` mode.
- The critic samples 10 responses (`num_return_sequences=10`) at temperature 1.1 to get a more stable reward signal via majority voting.
- For `cb` (bargaining), reward is the Sale-to-List Ratio extracted via regex from critic output; negative means no deal reached.
- prompt must be written externally in .yaml. 