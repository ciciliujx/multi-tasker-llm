"""
Evaluate a checkpoint on IFEval (Instruction Following Evaluation).

Uses Inspect AI's built-in IFEval task via the tinker-cookbook adapter.

Usage:
    python evaluation/eval_ifeval.py
    python evaluation/eval_ifeval.py --checkpoint_path "tinker://..."
    python evaluation/eval_ifeval.py --limit 20
"""

import argparse
import asyncio
import json
import logging
import os

import tinker
from inspect_ai import eval_async
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, Model

from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from tinker_cookbook.model_info import get_recommended_renderer_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TASK = "inspect_evals/ifeval"


async def run(args):
    renderer_name = args.renderer_name or get_recommended_renderer_name(args.base_model)
    logger.info(f"Model: {args.base_model} | Renderer: {renderer_name}")

    sc = tinker.ServiceClient()
    if args.checkpoint_path:
        sampling_client = sc.create_sampling_client(model_path=args.checkpoint_path)
    else:
        sampling_client = sc.create_sampling_client(base_model=args.base_model)

    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=args.base_model,
        sampling_client=sampling_client,
        verbose=args.verbose,
    )
    model = Model(
        api=api,
        config=GenerateConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ),
    )

    log_dir = args.log_dir or os.path.join(EVAL_DIR, "inspect-logs")
    results = await eval_async(
        tasks=[TASK],
        model=[model],
        limit=args.limit,
        debug_errors=True,
        retry_on_error=5,
        fail_on_error=False,
        log_dir=log_dir,
        max_connections=512,
    )

    # Extract aggregate metrics
    metrics = {}
    for r in results:
        if r.results and r.results.scores:
            for name, score in r.results.scores[0].metrics.items():
                ds = r.eval.dataset.name if r.eval.dataset else "ifeval"
                metrics[f"{ds}/{name}"] = score.value

    # Extract per-sample results from the log file
    samples = []
    for r in results:
        if r.location:
            log = read_eval_log(r.location)
            if log.samples:
                for s in log.samples:
                    score_val = {}
                    if s.scores:
                        for scorer_name, score in s.scores.items():
                            if isinstance(score.value, dict):
                                score_val = score.value
                    samples.append({
                        "id": s.id,
                        "prompt_strict": bool(score_val.get("prompt_level_strict", False)),
                        "prompt_loose": bool(score_val.get("prompt_level_loose", False)),
                        "num_instructions": int(score_val.get("num_instructions", 0)),
                        "inst_strict": int(score_val.get("inst_level_strict", 0)),
                        "inst_loose": int(score_val.get("inst_level_loose", 0)),
                    })

    logger.info(f"IFEval results: {json.dumps(metrics, indent=2)}")
    return {"metrics": metrics, "samples": samples}


def main():
    p = argparse.ArgumentParser(description="Evaluate on IFEval")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--renderer_name", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None, help="Max samples (None=all 541)")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    result = asyncio.run(run(p.parse_args()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
