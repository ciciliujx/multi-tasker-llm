# CPS572 Final Project: Multi-Task LLM Fine-Tuning

For setup instructions, evaluation commands, and getting started, see [`README.md`](README.md).

## Objective

Fine-tune a base large language model to perform well on three tasks simultaneously:

1. **Instruction Following** — evaluated on [IFEval](https://arxiv.org/abs/2311.07911)
2. **Math Reasoning** — evaluated on [GSM8K](https://arxiv.org/abs/2110.14168)
3. **Code Generation** — evaluated on [HumanEval](https://arxiv.org/abs/2107.03374)

All training can be done via LoRA fine-tuning on the [Tinker](https://github.com/thinking-machines-lab) platform, which provides API-based access to models and training infrastructure.

### Task Details

| Task | Benchmark | Samples | What It Measures | Metric |
|------|-----------|---------|------------------|--------|
| Instruction Following | IFEval | 541 | Whether the model follows verifiable constraints in prompts (e.g., "write exactly 3 paragraphs", "include the keyword X", "respond in all caps"). | Average of prompt-level and instruction-level accuracy (strict + loose) |
| Math Reasoning | GSM8K | 1,319 | Multi-step grade-school math word problems requiring arithmetic reasoning. The model must show its work and produce a final numeric answer. Evaluated zero-shot (no examples provided). | Exact-match on the final numeric answer |
| Code Generation | HumanEval | 164 | Python function completion given a docstring specification. The generated code is executed against hidden unit tests. | pass@1 (fraction of problems where the generated code passes all tests) |

---

## Model Constraints

- You **must** use a **Llama** model from the `meta-llama` family.
- Available models:
  - `meta-llama/Llama-3.2-1B` — smallest, fastest for development and debugging
  - `meta-llama/Llama-3.2-3B` — good balance for development
  - `meta-llama/Llama-3.1-8B` — **recommended for final submission**
- Start with a smaller model (3B) to iterate quickly on your approach, then train the 8B model for your final submission.

> **Note:** Each team member has a $250 Tinker budget. This should be more than sufficient to complete the project — plan your experiments accordingly and avoid unnecessary large-scale runs.

---

## Passing Baseline

The following per-task scores represent the baseline achieved by the TA using mixed supervised fine-tuning (SFT) on Llama-3.1-8B. Meeting these baselines earns you full baseline credit (4 out of 5 points per task):

| Task | Baseline Score |
|------|---------------|
| IFEval | 45.0% |
| GSM8K | 50.0% |
| HumanEval | 30.0% |

---

## Suggested Datasets

| Dataset | Task | Size |
|---------|------|------|
| [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (train split) | Math | 7,473 |
| [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | IF | ~939,000 |
| [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | Code | ~5M |

A mix of these three datasets with proper hyperparameter tuning should suffice to meet the baseline.

> **WARNING: Do NOT train on test data.** You must only use training splits for training. Do not train on the IFEval prompts, GSM8K test split, or HumanEval problems. We will review your code submissions — any team found to have trained on test data will receive a score of **0** for the entire project.

---

## What You Can Explore

There are many ways to improve beyond the baseline. Some ideas:

- **Data mixing strategies** — ratios, curriculum ordering, sampling weights
- **Hyperparameter tuning** — learning rate, LoRA rank, batch size, number of steps
- **Data selection and filtering** — quality filtering, deduplication, difficulty-based selection
- **Data augmentation** — generate additional training data or use alternative datasets
- **Reinforcement learning** — GRPO or other RL methods after SFT. Be careful: RL can improve one task at the cost of degrading others. Always evaluate all three tasks after any RL stage to ensure it does not harm overall performance.

> **Note:** For larger datasets, you do not need to use the full data — a well-chosen subset can be just as effective. You are also encouraged to try additional or alternative data sources to improve performance.

---

## Submission

Each team submits the following **three items** on Gradescope:

### 1. `submission.json` (Autograder)
Run the full evaluation (no `--limit` flag) on your best checkpoint:
```bash
python evaluation/eval_all.py --checkpoint_path "tinker://your-checkpoint" --base_model meta-llama/Llama-3.1-8B
```
Upload the resulting `submission.json` to the **Autograder** assignment on Gradescope. The leaderboard updates in real time — you may submit multiple times, but only your latest submission is graded. You can set a pseudonym for the leaderboard if you prefer to remain anonymous.

> **Note:** The `submission.json` includes your checkpoint path and generation settings (temperature, top_p). The teaching staff will **re-run the evaluation** using Tinker inference on the checkpoint and settings you provide to verify your results.

### 2. Final Report (PDF)

Upload your report as a single PDF. It must include:

1. **Methodology** — A clear description of exactly what you did: training strategy, datasets used, hyperparameters, number of experiments, etc.

2. **Extensions** — You must try at least one extension beyond simple fine-tuning and describe it in detail. Examples include (but are not limited to): reinforcement learning, data augmentation, curriculum learning, custom data filtering, novel mixing strategies, etc. Your report will be graded in part on the interestingness and thoughtfulness of the extensions you explored.

3. **Results and Analysis** — Detailed analysis of your results. Include:
   - What worked and why you think it worked
   - **What didn't work** — negative results are valued. Describe experiments that failed, degraded performance, or didn't help, and your analysis of why.
   - Comparisons across different approaches you tried
   - Learning curves or intermediate checkpoint results if applicable

4. **LLM Usage** — A dedicated section explicitly stating:
   - Which LLMs you used (e.g., ChatGPT, Gemini, Claude, etc.)
   - What you used them for (code writing, debugging, brainstorming, data generation, etc.)
   - Which specific parts of your code or approach were LLM-assisted

5. **Tinker Feedback** — Please answer the following questions about your experience with the Tinker platform:
   - What was the hardest part of using Tinker? Where did you get stuck?
   - Did you use the [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)? If yes, what was helpful and what was missing? If no, why not?
   - What is the one thing that would make Tinker significantly better for your use case?
   - Would you use Tinker again? Why or why not?

**Page Limit:** The main content of your report is limited to **4 pages**. Please note that the *LLM Usage* and *Tinker Feedback* sections do **not** count towards this limit.


### 3. Code (ZIP)
Upload a ZIP file containing all of your training code. This should include everything needed to understand and reproduce your approach.

---

## Grading (20% of Course Grade)

| Component | Weight | How It's Scored |
|-----------|--------|-----------------|
| **Final Report** | **5%** | Graded on clarity, detailed analysis, extensions, and inclusion of required sections (methodology, negative results, LLM usage) |
| **IFEval** | **5%** | 4%: meeting baseline (45.0%). 1%: relative grading across teams |
| **GSM8K** | **5%** | 4%: meeting baseline (50.0%). 1%: relative grading across teams |
| **HumanEval** | **5%** | 4%: meeting baseline (30.0%). 1%: relative grading across teams |

**Task Performance Scoring:**
- **Meeting Baseline (4/5 per task):** You receive 4 out of 5 points for each task where your model meets or exceeds the baseline score.
- **Improvements (1/5 per task):** The remaining point is awarded based on your improvement over the baseline. The actual thresholds will be determined based on the top and average scores achieved across the class. You have total freedom to try whatever methods you want.

---

## LLM Usage Policy

You are **allowed** to use LLMs (ChatGPT, Gemini, Claude, etc.) during this project. However, you **must** explicitly document your usage in the final report:

- Which LLMs you used
- For which specific steps (code writing, debugging, data generation, brainstorming, etc.)
- Which parts of your submission are LLM-assisted

Failure to disclose LLM usage is an academic integrity violation.

---

## Tips

- **Start small.** Use Llama-3.2-1B or 3B for rapid experimentation. Training is much faster on smaller models, and trends often transfer to the 8B model.
- **Evaluate intermediate checkpoints.** The final checkpoint is often not the best. Save checkpoints every N steps and evaluate each one. Overtraining is the main risk.
- **Watch for catastrophic forgetting.** Training on one task can hurt performance on others. Multi-task training helps prevent this.
