"""
Train a model with real data from GSM8K, Tulu-3, and OpenCodeInstruct.
Multi-task fine-tuning for IFEval, GSM8K, and HumanEval benchmarks.
"""

import argparse
import json
import os
import math

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from datasets import load_dataset

# MODEL = "meta-llama/Llama-3.2-3B"
MODEL = "meta-llama/Llama-3.1-8B"

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# 学习率 Warmup 工具
# ──────────────────────────────────────────────

def get_lr_with_warmup(step, base_lr, warmup_steps, total_steps, min_lr_ratio=0.1):
    """
    线性 warmup + cosine decay 学习率调度。
    - 前 warmup_steps 步：从 0 线性升到 base_lr
    - 之后：cosine 衰减到 base_lr * min_lr_ratio
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_factor)


# ──────────────────────────────────────────────
# 数据遍历工具
# ──────────────────────────────────────────────

class ShuffledDataLoader:
    """
    每个 epoch 结束后自动 reshuffle，避免边界重复采样。
    同时按 task_id 记录每个样本的任务类型，方便分任务统计 loss。
    """
    def __init__(self, data_with_task_ids, batch_size, seed=42):
        self.data = data_with_task_ids   # list of (datum, task_id)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.data))
        self.rng.shuffle(self.indices)
        self.pos = 0
        self.epoch = 0

    def next_batch(self):
        if self.pos + self.batch_size > len(self.indices):
            # 新 epoch：重新打乱
            self.rng.shuffle(self.indices)
            self.pos = 0
            self.epoch += 1
        batch_idx = self.indices[self.pos: self.pos + self.batch_size]
        self.pos += self.batch_size
        return [self.data[i] for i in batch_idx]


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=3000,
                        help="训练步数。8B 建议 2000-3000，3B 调试用 200-500")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="8B 模型建议 1e-4；3B 调试可用 2e-4")
    parser.add_argument("--rank", type=int, default=32,
                        help="LoRA rank。8B 建议 32-64；3B 调试用 16")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="学习率 warmup 步数，建议为总步数的 3-5%")
    parser.add_argument("--checkpoint_name", type=str, default="multitask_v1")
    parser.add_argument("--no_publish", action="store_true")
    # 数据量控制
    # parser.add_argument("--gsm8k_samples", type=int, default=7473,
    #                     help="GSM8K 样本数，默认全量")
    # parser.add_argument("--gsm8k_upsample", type=int, default=2,
    #                     help="GSM8K 上采样倍数，补偿数量少的问题")
    # parser.add_argument("--code_samples", type=int, default=5000)
    # parser.add_argument("--ifeval_samples", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=1024,
                        help="最大序列长度。512 会截断推理链，建议至少 1024")
    parser.add_argument("--data_path", type=str, default="training_data.jsonl",
                    help="Path to pre-processed training data jsonl")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=500,
                    help="Save intermediate checkpoint every N steps")
    parser.add_argument("--intermediate_ttl", type=int, default=86400,
                    help="TTL in seconds for intermediate checkpoints (default: 24h)")
    parser.add_argument("--resume_from", type=str, default=None,
                    help="Resume from saved state path (tinker://...)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print("=" * 60)
    print(f"Model:         {MODEL}")
    print(f"Steps:         {args.num_steps}")
    print(f"Batch size:    {args.batch_size}")
    print(f"LR:            {args.lr}  (warmup={args.warmup_steps} steps)")
    print(f"LoRA rank:     {args.rank}")
    print(f"Max length:    {args.max_length}")
    print(f"Checkpoint:    {args.checkpoint_name}")
    print("=" * 60)

    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}\n")

    # # ── 数据加载 ──────────────────────────────
    # gsm8k_convos  = load_gsm8k_data(args.gsm8k_samples, args.gsm8k_upsample)
    # ifeval_convos = load_ifeval_data(args.ifeval_samples)
    # code_convos   = load_code_data(args.code_samples)

    # task_id: 0=gsm8k, 1=ifeval, 2=code
    # labeled = (
    #     [(c, 0) for c in gsm8k_convos] +
    #     [(c, 1) for c in ifeval_convos] +
    #     [(c, 2) for c in code_convos]
    # )
    from collections import Counter

    print("Loading pre-processed training data...")
    with open(args.data_path) as f:
        training_data = [json.loads(line) for line in f]
    print(f"  Loaded {len(training_data)} examples")

    task_names = {0: "GSM8K", 1: "IFEval", 2: "Code"}
    dist = Counter(ex["task_id"] for ex in training_data)
    for tid, name in task_names.items():
        print(f"  {name}: {dist.get(tid, 0)}")

    labeled = [(ex["messages"], ex["task_id"]) for ex in training_data]

    # ── 转换成训练格式 ─────────────────────────
    print("\nPreparing training data...")
    data_with_ids = []
    skip_counts = {"too_long": 0, "error": 0}

    for convo, task_id in labeled:
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=args.max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
            data_with_ids.append((datum, task_id))
        except Exception as e:
            err_msg = str(e).lower()
            if "length" in err_msg or "token" in err_msg:
                skip_counts["too_long"] += 1
            else:
                skip_counts["error"] += 1

    # task_names = {0: "GSM8K", 1: "IFEval", 2: "Code"}
    task_counts = {name: 0 for name in task_names.values()}
    for _, tid in data_with_ids:
        task_counts[task_names[tid]] += 1

    print(f"  Ready: {len(data_with_ids)} examples")
    print(f"  Skipped (too long): {skip_counts['too_long']} | Skipped (error): {skip_counts['error']}")
    print(f"  Per-task breakdown: {task_counts}")

    if len(data_with_ids) == 0:
        raise RuntimeError("No training data after filtering! Check data loading.")

    # ── 训练 ──────────────────────────────────
    print(f"\nCreating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    # tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        tc = sc.create_training_client_from_state_with_optimizer(args.resume_from)
    else:
        tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready\n")

    loader = ShuffledDataLoader(data_with_ids, args.batch_size, seed=args.seed)

    # 分任务 loss 追踪
    task_loss_accum  = {0: 0.0, 1: 0.0, 2: 0.0}
    task_loss_counts = {0: 0,   1: 0,   2: 0}

    print(f"Training for {args.num_steps} steps...")
    intermediate_checkpoints = []

    for step in range(args.num_steps):

        # 动态学习率
        lr = get_lr_with_warmup(step, args.lr, args.warmup_steps, args.num_steps)
        adam_params = types.AdamParams(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8
        )

        batch_with_ids = loader.next_batch()
        batch   = [item[0] for item in batch_with_ids]
        task_ids = [item[1] for item in batch_with_ids]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future   = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # 计算整体 loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights  = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)

        # 按任务累积 loss（简化：整个 batch 用同一个 loss）
        # 如果 batch 里只有单一任务（理想情况），这是准确的
        for tid in task_ids:
            task_loss_accum[tid]  += loss / len(task_ids)
            task_loss_counts[tid] += 1

        if (step + 1) % args.log_interval == 0 or step == 0:
            # 计算各任务平均 loss
            task_avg = {}
            for tid in range(3):
                if task_loss_counts[tid] > 0:
                    task_avg[task_names[tid]] = task_loss_accum[tid] / task_loss_counts[tid]
                    task_loss_accum[tid]  = 0.0
                    task_loss_counts[tid] = 0

            loss_str = " | ".join([f"{k}: {v:.3f}" for k, v in task_avg.items()])
            print(
                f"  Step {step+1:4d}/{args.num_steps}"
                f" | LR: {lr:.2e}"
                f" | Loss: {loss:.4f}"
                f" | [{loss_str}]"
                f" | Epoch: {loader.epoch}"
            )

        if (step + 1) % args.save_every == 0 and (step + 1) < args.num_steps:
            ckpt_name = f"{args.checkpoint_name}_step{step+1}"
            print(f"\n  Saving intermediate checkpoint: {ckpt_name}")
            ckpt_mid = tc.save_weights_for_sampler(
                ckpt_name, ttl_seconds=args.intermediate_ttl
            ).result()
            print(f"  Saved: {ckpt_mid.path}")
            intermediate_checkpoints.append({
                "step": step + 1,
                "path": ckpt_mid.path,
                "name": ckpt_name
            })

    # ── 保存 & 发布 ───────────────────────────
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    print("Saving training state for potential resume...")
    state = tc.save_state(f"{args.checkpoint_name}_state").result()
    state_path = state.path
    print(f"  State saved: {state_path}")

    if not args.no_publish:
        print("Publishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published!")

    # ── 保存实验信息 ──────────────────────────
    # info = {
    #     "checkpoint_path": checkpoint_path,
    #     "base_model": MODEL,
    #     "training": {
    #         "num_steps": args.num_steps,
    #         "batch_size": args.batch_size,
    #         "learning_rate": args.lr,
    #         "warmup_steps": args.warmup_steps,
    #         "lora_rank": args.rank,
    #         "max_length": args.max_length,
    #         "gsm8k_samples": args.gsm8k_samples,
    #         "gsm8k_upsample": args.gsm8k_upsample,
    #         "code_samples": args.code_samples,
    #         "ifeval_samples": args.ifeval_samples,
    #         "total_training_examples": len(data_with_ids),
    #         "per_task_examples": task_counts,
    #     },
    # }
    # info = {
    #     "checkpoint_path": checkpoint_path,
    #     "base_model": MODEL,
    #     "training": {
    #         "num_steps":      args.num_steps,
    #         "batch_size":     args.batch_size,
    #         "learning_rate":  args.lr,
    #         "warmup_steps":   args.warmup_steps,
    #         "lora_rank":      args.rank,
    #         "max_length":     args.max_length,
    #         "data_path":      args.data_path,
    #         "total_examples": len(data_with_ids),
    #         "per_task":       task_counts,
    #     },
    # }
    info = {
        "checkpoint_path": checkpoint_path,
        "state_path": state_path,
        "intermediate_checkpoints": intermediate_checkpoints,
        "base_model": MODEL,
        "training": {
            "num_steps":      args.num_steps,
            "batch_size":     args.batch_size,
            "learning_rate":  args.lr,
            "warmup_steps":   args.warmup_steps,
            "lora_rank":      args.rank,
            "max_length":     args.max_length,
            "data_path":      args.data_path,
            "total_examples": len(data_with_ids),
            "per_task":       task_counts,
            "resumed_from":   args.resume_from,
        },
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Experiment info saved to {info_path}")

    print(f"\nDone! Evaluate with:")
    print(f'  python evaluation/eval_all.py --checkpoint_path "{checkpoint_path}" --base_model {MODEL} --limit 20')


if __name__ == "__main__":
    main()
