# CPS572 Mini Project: Multi-Task LLM Fine-Tuning

This project fine-tunes a Llama base model to perform well on three tasks simultaneously:
- **Instruction Following** — evaluated on IFEval
- **Math Reasoning** — evaluated on GSM8K  
- **Code Generation** — evaluated on HumanEval

All training is done via LoRA fine-tuning on the [Tinker](https://tinkerhub.ai) platform.

---

## Results

| Task | Baseline | Our Best |
|---|---|---|
| IFEval | 45.0% | 72.0% |
| GSM8K | 50.0% | 67.2% |
| HumanEval | 30.0% | 48.2% |

Best checkpoint: `tinker://fd14953b-eff7-55e9-9c57-3154fc935ba9:train:0/sampler_weights/final_8b_v1_step3500`  
Base model: `meta-llama/Llama-3.1-8B`

---

## Repository Structure
```
.
├── evaluation/
│   ├── eval_all.py              # Run full evaluation on all three tasks
│   ├── eval_gsm8k.py            # Evaluate on GSM8K only
│   ├── eval_ifeval.py           # Evaluate on IFEval only
│   ├── eval_code.py             # Evaluate on HumanEval only
│   ├── train_and_publish.py     # Main training script
│   └── checkpoint_info.json     # Saved checkpoint paths and training metadata
├── data_prep.ipynb              # Data pipeline: loading, filtering, deduplication,
│                                #   difficulty sampling, contamination check, final mix
├── baseline_model.ipynb         # Main training notebook
├── pseudo RL.ipynb              # Extension using pseudo reinforcement learning
├── test_data_scan.ipynb         # Contamination check against test sets
├── training_data.jsonl          # Final preprocessed training data (30K examples, 1:1:1)
└── README.md
```

---

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess training data

Open and run `data_prep.ipynb` end-to-end. This notebook:
- Loads GSM8K train + MetaMathQA (math), Tulu-3 filtered by source (IF), and CodeAlpaca-20K + OpenCodeInstruct (code)
- Applies quality filtering, MinHash deduplication, and difficulty-stratified sampling
- Saves the final 30K training examples to `training_data_clean.jsonl`

Alternatively, use the pre-processed `training_data.jsonl` directly to skip this step.

### 3. Train

```bash
# Debug run on 3B (fast, cheap)
python evaluation/train_and_publish.py \
    --data_path "training_data_clean.jsonl" \
    --num_steps 300 \
    --batch_size 4 \
    --lr 2e-4 \
    --rank 16 \
    --warmup_steps 15 \
    --max_length 2048 \
    --checkpoint_name "debug_3b"

# Final run on 8B
python evaluation/train_and_publish.py \
    --data_path "training_data_clean.jsonl" \
    --num_steps 5000 \
    --batch_size 4 \
    --lr 1e-4 \
    --rank 32 \
    --warmup_steps 150 \
    --max_length 2048 \
    --save_every 500 \
    --checkpoint_name "final_8b"
```

Key arguments:

| Argument | Description |
|---|---|
| `--data_path` | Path to preprocessed training data |
| `--num_steps` | Total training steps |
| `--rank` | LoRA rank (16 for 3B debug, 32 for 8B final) |
| `--save_every` | Save intermediate checkpoint every N steps |
| `--resume_from` | Resume from a saved state path (`tinker://...`) |
| `--no_publish` | Skip publishing checkpoint after training |

### 4. Evaluate

```bash
# Smoke test (5 samples per task)
python evaluation/eval_all.py \
    --checkpoint_path "tinker://YOUR_CHECKPOINT" \
    --base_model meta-llama/Llama-3.1-8B \
    --limit 5

# Full evaluation
python evaluation/eval_all.py \
    --checkpoint_path "tinker://YOUR_CHECKPOINT" \
    --base_model meta-llama/Llama-3.1-8B
```

---

## Training Data

The final training set (`training_data_clean.jsonl`) contains 30,000 examples in equal 1:1:1 ratio:

| Task | Source | Size |
|---|---|---|
| Math | GSM8K train + MetaMathQA (GSM-origin) | 10,000 |
| Instruction Following | Tulu-3 `personahub_ifdata_manual_seed_v3_29980` | 10,000 |
| Code | CodeAlpaca-20K + OpenCodeInstruct (score=1.0) | 10,000 |

All examples are formatted to match the exact eval harness prompt templates.
Contamination-checked against all three test sets — one near-duplicate removed.

---

## Supplementary Documentation

Additional experiment artifacts and archived training materials are provided in [docs-artifacts](https://github.com/ciciliujx/multi-tasker-llm/tree/docs-artifacts) branch, including command history and an archived exploratory training script used during development.

## LLM Usage

Claude (Anthropic) was used for brainstorming data preprocessing strategies,
debugging training code, and drafting report sections.
All code was reviewed and validated by the team.
