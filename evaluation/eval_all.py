"""
Evaluate base models or trained checkpoints on core tasks (IFEval, GSM8K, HumanEval).

Usage:
    # Evaluate one or more base models:
    python evaluation/eval_all.py --base_models meta-llama/Llama-3.2-3B

    # Evaluate a trained checkpoint:
    python evaluation/eval_all.py --checkpoint_path "tinker://..." --temperature 0.3 --top_p 0.9

    # Quick smoke test (5 samples per task):
    python evaluation/eval_all.py --base_models meta-llama/Llama-3.2-3B --limit 5
"""

import argparse
import asyncio
import json
import logging
import os

from tinker_cookbook.model_info import get_recommended_renderer_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

async def run_core(base_model, checkpoint_path, renderer_name, temperature,
                   top_p, limit, log_dir, verbose):
    """Run core 3 tasks for a single model/checkpoint. Returns (all_metrics, task_results)."""
    if not renderer_name:
        renderer_name = get_recommended_renderer_name(base_model)
    label = checkpoint_path or f"base:{base_model}"
    logger.info(f"Evaluating: {label}")
    logger.info(f"Renderer: {renderer_name} | temperature={temperature} top_p={top_p}")

    all_metrics = {}
    task_results = {}

    task_args = dict(
        checkpoint_path=checkpoint_path, base_model=base_model,
        renderer_name=renderer_name, temperature=temperature, top_p=top_p,
        max_tokens=1024, limit=limit, log_dir=log_dir, verbose=verbose,
    )

    # 1. IFEval
    logger.info("=" * 60)
    logger.info("TASK 1/3: IFEval (Instruction Following)")
    logger.info("=" * 60)
    try:
        from evaluation.eval_ifeval import run as run_ifeval
        result = await run_ifeval(argparse.Namespace(**task_args))
        all_metrics.update(result["metrics"])
        task_results["ifeval"] = result
    except Exception as e:
        logger.error(f"IFEval failed: {e}")
        all_metrics["ifeval/error"] = str(e)
        task_results["ifeval"] = {"error": str(e)}

    # 2. GSM8K
    logger.info("=" * 60)
    logger.info("TASK 2/3: GSM8K (Math Reasoning)")
    logger.info("=" * 60)
    try:
        from evaluation.eval_gsm8k import run as run_gsm8k
        result = await run_gsm8k(argparse.Namespace(**task_args))
        all_metrics.update(result["metrics"])
        task_results["gsm8k"] = result
    except Exception as e:
        logger.error(f"GSM8K failed: {e}")
        all_metrics["gsm8k/error"] = str(e)
        task_results["gsm8k"] = {"error": str(e)}

    # 3. HumanEval
    logger.info("=" * 60)
    logger.info("TASK 3/3: HumanEval (Code Generation)")
    logger.info("=" * 60)
    try:
        from evaluation.eval_code import run as run_code
        result = await run_code(argparse.Namespace(**task_args))
        all_metrics.update(result["metrics"])
        task_results["humaneval"] = result
    except Exception as e:
        logger.error(f"HumanEval failed: {e}")
        all_metrics["humaneval/error"] = str(e)
        task_results["humaneval"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY (core): {label}")
    print("=" * 60)
    print(json.dumps(all_metrics, indent=2))
    print("=" * 60)

    return all_metrics, task_results


def print_comparison(all_data, title):
    """Print a comparison table for a dict of {model: metrics}."""
    if len(all_data) < 2:
        return
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    all_keys = set()
    for m in all_data.values():
        all_keys.update(k for k in m if "error" not in k and "stderr" not in k)
    models_list = list(all_data.keys())
    header = f"{'Metric':<45}" + "".join(f"{m.split('/')[-1]:>15}" for m in models_list)
    print(header)
    print("-" * len(header))
    for key in sorted(all_keys):
        row = f"  {key:<43}"
        for model in models_list:
            val = all_data[model].get(key, "N/A")
            if isinstance(val, (int, float)):
                row += f"{val:>15.4f}"
            else:
                row += f"{str(val):>15}"
        print(row)
    print("=" * 80)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Evaluate on core + optional tasks")
    p.add_argument("--base_models", type=str, nargs="+", default=None,
                    help="One or more base models to evaluate (e.g. meta-llama/Llama-3.2-3B)")
    p.add_argument("--checkpoint_path", type=str, default=None,
                    help="Tinker checkpoint path to evaluate")
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B",
                    help="Base model for the checkpoint (used for tokenizer/renderer)")
    p.add_argument("--renderer_name", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (default: 0.0 = greedy)")
    p.add_argument("--top_p", type=float, default=1.0,
                    help="Top-p (nucleus) sampling (default: 1.0 = disabled)")
    p.add_argument("--limit", type=int, default=None, help="Max samples per task")
    p.add_argument("--output_path", type=str, default=None,
                    help="Path for submission JSON (default: evaluation/submission.json)")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not args.base_models and not args.checkpoint_path:
        p.error("Provide either --base_models or --checkpoint_path")

    if args.checkpoint_path:
        # Checkpoint mode: evaluate one checkpoint, save submission.json
        metrics, task_results = asyncio.run(run_core(
            base_model=args.base_model,
            checkpoint_path=args.checkpoint_path,
            renderer_name=args.renderer_name,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
            log_dir=args.log_dir,
            verbose=args.verbose,
        ))
        submission = {
            "checkpoint_path": args.checkpoint_path,
            "base_model": args.base_model,
            "settings": {
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            **task_results,
        }
        out_path = args.output_path or os.path.join(EVAL_DIR, "submission.json")
        save_json(out_path, submission)
        logger.info(f"Submission saved to {out_path}")

    else:
        # Baseline mode: evaluate one or more base models
        core_path = os.path.join(EVAL_DIR, "baseline_results.json")

        core_baselines = load_json(core_path).get("models", {})

        for model in args.base_models:
            # Core tasks
            if model in core_baselines and args.limit is None:
                logger.info(f"SKIP core: {model} already in baseline_results.json")
            else:
                print("\n" + "#" * 60)
                print(f"# BASELINE (core): {model}")
                print("#" * 60)
                metrics, _ = asyncio.run(run_core(
                    base_model=model,
                    checkpoint_path=None,
                    renderer_name=None,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    limit=args.limit,
                    log_dir=args.log_dir,
                    verbose=args.verbose,
                ))
                if args.limit is None:
                    core_baselines[model] = metrics
                    save_json(core_path, {"type": "baseline", "models": core_baselines})
                    logger.info(f"Core baselines updated: {core_path}")
                else:
                    logger.info(f"--limit={args.limit} set, results NOT saved")

        # Print comparison table
        print_comparison(core_baselines, "BASELINE COMPARISON")


if __name__ == "__main__":
    main()
