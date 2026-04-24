"""
Train a model (minimal SFT), save checkpoint, and publish it.

NOTE: This is a TOY EXAMPLE that trains for a few steps on dummy data
to verify the full workflow end-to-end. You should replace the training
data and training logic with your own implementation.

TODO:
  - Replace DEMO_CONVERSATIONS with your task-specific training data
  - Tune hyperparameters (learning rate, batch size, number of steps, LoRA rank)
  - Add validation / early stopping as needed

Usage:
    python evaluation/train_and_publish.py
    python evaluation/train_and_publish.py --num_steps 20
    python evaluation/train_and_publish.py --no_publish   # skip publishing
"""

import argparse
import math
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import tinker
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "meta-llama/Llama-3.2-3B"
# MODEL = "meta-llama/Llama-3.2-1B"    # Smaller, faster for development
# MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_ROLES = {"system", "user", "assistant"}
CACHE_ROOT = os.path.join(EVAL_DIR, "cached_datasets")
TASK_NAMES = ("gsm8k", "tulu", "code")
TULU_SOURCE_NAMES = {
    "all",
    "personahub_if",
    "personahub_math",
    "personahub_code",
}
IFEVAL_KEYWORDS = (
    "paragraph",
    "bullet",
    "keyword",
    "capital",
    "word count",
    "do not use",
    "must include",
    "exactly",
    "without using",
    "json format",
    "numbered list",
    "all caps",
    "lowercase",
    "end your",
    "start with",
    "first word",
    "last word",
    "contain",
    "avoid",
    "forbidden",
    "constraint",
)


def _to_text(value):
    """Best-effort conversion of dataset fields into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _normalize_message(message):
    role = message.get("role")
    content = _to_text(message.get("content")).strip()
    if role not in VALID_ROLES or not content:
        return None
    return {"role": role, "content": content}


def _clean_code_response(text):
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _conversation_signature(convo):
    return json.dumps(convo, sort_keys=True, ensure_ascii=True)


def _looks_like_code(text):
    code_markers = ("def ", "class ", "return ", "import ", "for ", "while ", "if ", "print(")
    return any(marker in text for marker in code_markers)


def _parse_source_list(spec):
    if not spec:
        return set()
    sources = {part.strip().lower() for part in spec.split(",") if part.strip()}
    unknown = sources.difference(TASK_NAMES)
    if unknown:
        raise ValueError(f"Unknown sources: {', '.join(sorted(unknown))}. Valid sources: {', '.join(TASK_NAMES)}")
    return sources


def _parse_tulu_source_names(spec):
    if not spec:
        return {"all"}
    names = {part.strip().lower() for part in spec.split(",") if part.strip()}
    unknown = names.difference(TULU_SOURCE_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown Tulu source names: {', '.join(sorted(unknown))}. "
            f"Valid names: {', '.join(sorted(TULU_SOURCE_NAMES))}"
        )
    return names


def _matches_tulu_source(source_value, selected_names):
    if "all" in selected_names:
        return True
    mapping = {
        "personahub_if": "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
        "personahub_math": "ai2-adapt-dev/personahub_math_v5_regen_149960",
        "personahub_code": "ai2-adapt-dev/personahub_code_v2_34999",
    }
    return any(source_value == mapping[name] for name in selected_names if name in mapping)


def _passes_quality_filter(convo, source):
    if len(convo) < 2:
        return False

    user_text = convo[-2]["content"] if len(convo) >= 2 else ""
    assistant_text = convo[-1]["content"]

    if source == "gsm8k":
        if "####" not in assistant_text:
            return False
        if len(user_text.split()) < 6 or len(assistant_text.split()) < 16:
            return False
        return any(ch.isdigit() for ch in assistant_text)

    if source == "code":
        if len(user_text.split()) < 6 or len(assistant_text.split()) < 6:
            return False
        if "```" in assistant_text:
            return False
        if not _looks_like_code(assistant_text):
            return False
        return "\n" in assistant_text

    if source == "tulu":
        turn_count = len([msg for msg in convo if msg["role"] != "system"])
        if turn_count < 2 or turn_count > 6:
            return False
        if convo[-2]["role"] != "user" or convo[-1]["role"] != "assistant":
            return False
        if len(user_text.split()) < 4 or len(assistant_text.split()) < 8:
            return False
        return True

    return True


def _is_ifeval_style_tulu(messages):
    if not messages:
        return False
    user_msg = _to_text(messages[0].get("content", "")).lower()
    return any(keyword in user_msg for keyword in IFEVAL_KEYWORDS)


def _difficulty_score(convo, source):
    user_text = convo[-2]["content"] if len(convo) >= 2 else ""
    assistant_text = convo[-1]["content"] if convo else ""

    if source == "gsm8k":
        question_numbers = sum(ch.isdigit() for ch in user_text)
        reasoning_steps = assistant_text.count("\n") + assistant_text.count("####")
        op_count = sum(assistant_text.count(op) for op in ("+", "-", "*", "/", "="))
        return float(len(user_text.split()) + len(assistant_text.split()) + question_numbers + 2 * op_count + 5 * reasoning_steps)

    if source == "code":
        line_count = assistant_text.count("\n") + 1
        indent_count = assistant_text.count("    ")
        branch_count = sum(assistant_text.count(tok) for tok in ("if ", "for ", "while ", "try:", "class ", "def "))
        prompt_len = len(user_text.split())
        return float(2 * line_count + indent_count + 3 * branch_count + prompt_len)

    if source == "tulu":
        turn_count = len([msg for msg in convo if msg["role"] != "system"])
        return float(turn_count + len(user_text.split()) + len(assistant_text.split()))

    return float(len(user_text.split()) + len(assistant_text.split()))


def _apply_difficulty_selection(conversations, sources, keep_ratio):
    if not conversations or not sources:
        return conversations
    keep_ratio = min(max(keep_ratio, 0.0), 1.0)
    if keep_ratio >= 1.0:
        return conversations

    grouped = {task: [] for task in TASK_NAMES}
    for convo, source in conversations:
        grouped[source].append((convo, source))

    kept = []
    for source in TASK_NAMES:
        items = grouped[source]
        if source not in sources:
            kept.extend(items)
            continue
        if not items:
            continue
        scored = sorted(items, key=lambda item: _difficulty_score(item[0], source), reverse=True)
        keep_count = max(1, math.ceil(len(scored) * keep_ratio))
        kept.extend(scored[:keep_count])
    return kept


def _get_lr_with_warmup(step, base_lr, warmup_steps, total_steps, min_lr_ratio):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_factor)


def _is_strict_tulu_conversation(convo, max_turns):
    if not convo:
        return False
    idx = 0
    if convo[0]["role"] == "system":
        idx = 1
    remaining = convo[idx:]
    if len(remaining) < 2 or len(remaining) > max_turns:
        return False
    if remaining[0]["role"] != "user" or remaining[-1]["role"] != "assistant":
        return False
    for pos, msg in enumerate(remaining):
        expected = "user" if pos % 2 == 0 else "assistant"
        if msg["role"] != expected:
            return False
    return True


def _clean_conversation(convo, source, strict_tulu=False, tulu_max_turns=4):
    cleaned = []
    for msg in convo:
        normalized = _normalize_message(msg)
        if normalized is None:
            continue
        if source == "code" and normalized["role"] == "assistant":
            normalized["content"] = _clean_code_response(normalized["content"])
            if not normalized["content"]:
                return None
        cleaned.append(normalized)

    if len(cleaned) < 2 or cleaned[-1]["role"] != "assistant":
        return None

    if source != "tulu" and cleaned[0]["role"] == "assistant":
        return None

    if source == "tulu" and strict_tulu and not _is_strict_tulu_conversation(cleaned, tulu_max_turns):
        return None

    return cleaned


def _build_cache_path(
    gsm8k_size,
    metamath_size,
    tulu_size,
    code_size,
    alpaca_size,
    mbpp_size,
    seed,
    clean_data,
    strict_tulu,
    tulu_max_turns,
    dedup,
    quality_filter,
    quality_filter_sources,
    difficulty_filter,
    difficulty_filter_sources,
    difficulty_keep_ratio,
    tulu_source_names,
    tulu_keyword_bias,
    tulu_keyword_fraction,
):
    cache_key = {
        "gsm8k_size": gsm8k_size,
        "metamath_size": metamath_size,
        "tulu_size": tulu_size,
        "code_size": code_size,
        "alpaca_size": alpaca_size,
        "mbpp_size": mbpp_size,
        "seed": seed,
        "clean_data": clean_data,
        "strict_tulu": strict_tulu,
        "tulu_max_turns": tulu_max_turns,
        "dedup": dedup,
        "quality_filter": quality_filter,
        "quality_filter_sources": sorted(quality_filter_sources),
        "difficulty_filter": difficulty_filter,
        "difficulty_filter_sources": sorted(difficulty_filter_sources),
        "difficulty_keep_ratio": difficulty_keep_ratio,
        "tulu_source_names": sorted(tulu_source_names),
        "tulu_keyword_bias": tulu_keyword_bias,
        "tulu_keyword_fraction": tulu_keyword_fraction,
    }
    digest = hashlib.sha1(json.dumps(cache_key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    label = (
        f"g{gsm8k_size}_mm{metamath_size}_t{tulu_size}_c{code_size}_a{alpaca_size}_m{mbpp_size}_s{seed}"
        f"_clean{int(clean_data)}_strict{int(strict_tulu)}_turns{tulu_max_turns}"
        f"_dedup{int(dedup)}_qf{int(quality_filter)}"
        f"_src{'-'.join(sorted(quality_filter_sources)) or 'none'}_{digest}"
        f"_df{int(difficulty_filter)}"
        f"_dfsrc{'-'.join(sorted(difficulty_filter_sources)) or 'none'}"
        f"_keep{difficulty_keep_ratio}"
        f"_tulu{'-'.join(sorted(tulu_source_names)) or 'all'}"
        f"_tbias{int(tulu_keyword_bias)}_tfrac{tulu_keyword_fraction}"
    )
    return os.path.join(CACHE_ROOT, label)


def _save_cached_conversations(cache_path, conversations, task_counts):
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    rows = [{"conversation": convo, "source": source} for convo, source in conversations]
    Dataset.from_list(rows).save_to_disk(cache_path)
    with open(os.path.join(cache_path, "metadata.json"), "w") as f:
        json.dump({"task_counts": task_counts, "num_examples": len(conversations)}, f, indent=2)


def _load_cached_conversations(cache_path, reshuffle_seed=None):
    dataset = load_from_disk(cache_path)
    rows = list(dataset)
    if reshuffle_seed is not None:
        rng = np.random.default_rng(reshuffle_seed)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        rows = [rows[i] for i in indices]
    conversations = [(row["conversation"], row["source"]) for row in rows]
    with open(os.path.join(cache_path, "metadata.json")) as f:
        metadata = json.load(f)
    return conversations, metadata["task_counts"]


def load_mixed_conversations(
    gsm8k_size,
    metamath_size,
    tulu_size,
    code_size,
    alpaca_size,
    mbpp_size,
    seed,
    clean_data=True,
    use_cache=True,
    strict_tulu=False,
    tulu_max_turns=4,
    dedup=False,
    quality_filter=False,
    quality_filter_sources=None,
    difficulty_filter=False,
    difficulty_filter_sources=None,
    difficulty_keep_ratio=1.0,
    tulu_source_names=None,
    tulu_keyword_bias=False,
    tulu_keyword_fraction=0.6,
):
    """Load and mix training data from the three suggested project datasets."""
    cache_path = _build_cache_path(
        gsm8k_size,
        metamath_size,
        tulu_size,
        code_size,
        alpaca_size,
        mbpp_size,
        seed,
        clean_data,
        strict_tulu,
        tulu_max_turns,
        dedup,
        quality_filter,
        quality_filter_sources or set(),
        difficulty_filter,
        difficulty_filter_sources or set(),
        difficulty_keep_ratio,
        tulu_source_names or {"all"},
        tulu_keyword_bias,
        tulu_keyword_fraction,
    )
    if use_cache and os.path.exists(cache_path):
        print(f"  Loading cached mixed dataset from {cache_path}")
        return _load_cached_conversations(cache_path, reshuffle_seed=seed)

    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    metamath = None
    tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    code = load_dataset("nvidia/OpenCodeInstruct", split="train")
    alpaca = None
    mbpp = None

    gsm8k = gsm8k.shuffle(seed=seed).select(range(min(gsm8k_size, len(gsm8k))))
    code = code.shuffle(seed=seed).select(range(min(code_size, len(code))))
    if metamath_size > 0:
        metamath = load_dataset("meta-math/MetaMathQA", split="train")
        metamath = metamath.shuffle(seed=seed).select(range(min(metamath_size, len(metamath))))
    tulu = tulu.shuffle(seed=seed)
    if tulu_keyword_bias:
        pool_size = min(len(tulu), max(tulu_size * 4, tulu_size + 5000))
        tulu_pool = tulu.select(range(pool_size))
        tulu_rows = [row for row in tulu_pool if _matches_tulu_source(_to_text(row.get("source")).strip(), tulu_source_names or {"all"})]
        keyword_rows = [row for row in tulu_rows if _is_ifeval_style_tulu(row.get("messages", []))]
        other_rows = [row for row in tulu_rows if not _is_ifeval_style_tulu(row.get("messages", []))]
        target = min(tulu_size, len(tulu_rows))
        target_keyword = min(len(keyword_rows), int(target * tulu_keyword_fraction))
        target_other = min(len(other_rows), target - target_keyword)
        selected_rows = keyword_rows[:target_keyword] + other_rows[:target_other]
        if len(selected_rows) < target:
            remaining = keyword_rows[target_keyword:] + other_rows[target_other:]
            selected_rows.extend(remaining[: target - len(selected_rows)])
        tulu = Dataset.from_list(selected_rows)
    else:
        tulu = tulu.select(range(min(tulu_size, len(tulu))))
    if alpaca_size > 0:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca = alpaca.shuffle(seed=seed).select(range(min(alpaca_size, len(alpaca))))
    if mbpp_size > 0:
        mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
        mbpp = mbpp.shuffle(seed=seed).select(range(min(mbpp_size, len(mbpp))))

    def format_gsm8k(example):
        return {
            "conversation": [
                {"role": "user", "content": _to_text(example["question"]).strip()},
                {"role": "assistant", "content": _to_text(example["answer"]).strip()},
            ],
            "source": "gsm8k",
        }

    def format_metamath(example):
        query = _to_text(example.get("query", "")).strip()
        response = _to_text(example.get("response", "")).strip()
        match = re.search(r"The answer is:\s*(-?[\d,]+\.?\d*)", response)
        if not query or not response or not match:
            return {"conversation": [], "source": "gsm8k"}
        answer_num = match.group(1).replace(",", "")
        response = re.sub(r"The answer is:.*$", f"#### {answer_num}", response, flags=re.DOTALL)
        return {
            "conversation": [
                {
                    "role": "system",
                    "content": (
                        "You are a math problem solver. "
                        "Think step by step and show your reasoning. "
                        "At the end, write your final answer in the format: #### <number>"
                    ),
                },
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ],
            "source": "gsm8k",
        }

    def format_tulu(example):
        if not _matches_tulu_source(_to_text(example.get("source")).strip(), tulu_source_names or {"all"}):
            return {"conversation": [], "source": "tulu"}
        messages = []
        for msg in example.get("messages", []):
            role = msg.get("role")
            content = _to_text(msg.get("content")).strip()
            if role in VALID_ROLES and content:
                messages.append({"role": role, "content": content})
        return {"conversation": messages, "source": "tulu"}

    def format_code(example):
        return {
            "conversation": [
                {"role": "user", "content": _to_text(example["input"]).strip()},
                {"role": "assistant", "content": _to_text(example["output"]).rstrip()},
            ],
            "source": "code",
        }

    def format_alpaca(example):
        user_text = _to_text(example["instruction"]).strip()
        extra_input = _to_text(example.get("input", "")).strip()
        if extra_input:
            user_text = f"{user_text}\n\n{extra_input}"
        return {
            "conversation": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": _to_text(example["output"]).strip()},
            ],
            "source": "tulu",
        }

    def format_mbpp(example):
        prompt = (
            f"Write a Python function to solve the following problem:\n{_to_text(example['text']).strip()}\n\n"
            "Your code should satisfy these tests:\n"
            + "\n".join(_to_text(test).strip() for test in example["test_list"][:3])
            + "\n\nOutput only raw Python code."
        )
        return {
            "conversation": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _to_text(example["code"]).rstrip()},
            ],
            "source": "code",
        }

    gsm8k = gsm8k.map(format_gsm8k, remove_columns=gsm8k.column_names)
    if metamath is not None:
        metamath = metamath.map(format_metamath, remove_columns=metamath.column_names)
    tulu = tulu.map(format_tulu, remove_columns=tulu.column_names)
    code = code.map(format_code, remove_columns=code.column_names)
    datasets_to_mix = [tulu, gsm8k, code]
    if metamath is not None:
        datasets_to_mix.append(metamath)
    if alpaca is not None:
        alpaca = alpaca.map(format_alpaca, remove_columns=alpaca.column_names)
        datasets_to_mix.append(alpaca)
    if mbpp is not None:
        mbpp = mbpp.map(format_mbpp, remove_columns=mbpp.column_names)
        datasets_to_mix.append(mbpp)

    mixed = concatenate_datasets(datasets_to_mix).shuffle(seed=seed)

    conversations = []
    seen = {task: set() for task in TASK_NAMES}
    for row in mixed:
        convo = row["conversation"]
        source = row["source"]
        if clean_data:
            convo = _clean_conversation(
                convo,
                source,
                strict_tulu=strict_tulu,
                tulu_max_turns=tulu_max_turns,
            )
        if convo is None:
            continue
        if quality_filter and source in quality_filter_sources and not _passes_quality_filter(convo, source):
            continue
        if dedup:
            signature = _conversation_signature(convo)
            if signature in seen[source]:
                continue
            seen[source].add(signature)
        conversations.append((convo, source))

    if difficulty_filter:
        conversations = _apply_difficulty_selection(
            conversations,
            difficulty_filter_sources or set(),
            difficulty_keep_ratio,
        )

    task_counts = {"tulu": 0, "gsm8k": 0, "code": 0}
    for _convo, source in conversations:
        task_counts[source] += 1

    if use_cache:
        print(f"  Saving mixed dataset cache to {cache_path}")
        _save_cached_conversations(cache_path, conversations, task_counts)

    return conversations, task_counts


class TaskAwareSampler:
    def __init__(self, data_by_task, batch_size, weights, seed):
        self.data_by_task = data_by_task
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.task_names = [task for task in TASK_NAMES if data_by_task[task]]
        if not self.task_names:
            raise RuntimeError("No training data available for task-aware sampling")
        raw_weights = np.array([max(weights.get(task, 0.0), 0.0) for task in self.task_names], dtype=float)
        if raw_weights.sum() <= 0:
            raw_weights = np.ones(len(self.task_names), dtype=float)
        self.probs = raw_weights / raw_weights.sum()
        self.indices = {task: np.arange(len(data_by_task[task])) for task in self.task_names}
        self.positions = {task: 0 for task in self.task_names}
        for task in self.task_names:
            self.rng.shuffle(self.indices[task])

    def _next_example(self, task):
        idxs = self.indices[task]
        pos = self.positions[task]
        if pos >= len(idxs):
            self.rng.shuffle(idxs)
            pos = 0
        example = self.data_by_task[task][idxs[pos]]
        self.positions[task] = pos + 1
        return example

    def next_batch(self):
        chosen_tasks = self.rng.choice(self.task_names, size=self.batch_size, p=self.probs)
        return [self._next_example(task) for task in chosen_tasks], chosen_tasks.tolist()


def _parse_task_weights(spec):
    weights = {task: 1.0 for task in TASK_NAMES}
    if not spec:
        return weights
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid task weight '{part}'. Expected format task=value")
        task, value = part.split("=", 1)
        task = task.strip().lower()
        if task not in weights:
            raise ValueError(f"Unknown task '{task}'. Valid tasks: {', '.join(TASK_NAMES)}")
        weights[task] = float(value)
    return weights


def _summarize_batch_tasks(task_names):
    counts = {task: 0 for task in TASK_NAMES}
    for task in task_names:
        counts[task] += 1
    return ", ".join(f"{task}:{counts[task]}" for task in TASK_NAMES if counts[task])


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Base model to fine-tune, e.g. meta-llama/Llama-3.2-3B or meta-llama/Llama-3.1-8B",
    )
    parser.add_argument("--num_steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps before cosine decay")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Final LR ratio for cosine decay")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--checkpoint_name", type=str, default="demo", help="Checkpoint name")
    parser.add_argument("--max_length", type=int, default=1024, help="Max tokens per conversation")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for dataset mixing")
    parser.add_argument("--gsm8k_size", type=int, default=2000, help="Number of GSM8K train examples")
    parser.add_argument("--metamath_size", type=int, default=0, help="Number of MetaMathQA examples to add as math augmentation")
    parser.add_argument("--tulu_size", type=int, default=10000, help="Number of Tulu train examples")
    parser.add_argument("--code_size", type=int, default=10000, help="Number of OpenCodeInstruct train examples")
    parser.add_argument(
        "--tulu_source_names",
        type=str,
        default="all",
        help="Comma-separated Tulu source groups: all, personahub_if, personahub_math, personahub_code",
    )
    parser.add_argument(
        "--tulu_keyword_bias",
        action="store_true",
        help="Bias Tulu sampling toward prompts that look like IFEval-style constraint following",
    )
    parser.add_argument(
        "--tulu_keyword_fraction",
        type=float,
        default=0.6,
        help="Target fraction of keyword-matched Tulu examples when keyword bias is enabled",
    )
    parser.add_argument("--alpaca_size", type=int, default=0, help="Number of Alpaca examples to add as instruction augmentation")
    parser.add_argument("--mbpp_size", type=int, default=0, help="Number of MBPP examples to add as code augmentation")
    parser.add_argument("--clean_data", action="store_true", help="Apply light cleaning/filtering before training")
    parser.add_argument("--dedup", action="store_true", help="Remove exact duplicate conversations within each source")
    parser.add_argument(
        "--quality_filter",
        action="store_true",
        help="Apply lightweight source-specific quality filters before training",
    )
    parser.add_argument(
        "--quality_filter_sources",
        type=str,
        default="gsm8k,code,tulu",
        help="Comma-separated sources to quality filter: gsm8k, tulu, code",
    )
    parser.add_argument(
        "--difficulty_filter",
        action="store_true",
        help="Keep only harder examples for selected sources using simple heuristics",
    )
    parser.add_argument(
        "--difficulty_filter_sources",
        type=str,
        default="gsm8k,code",
        help="Comma-separated sources to apply difficulty selection to: gsm8k, tulu, code",
    )
    parser.add_argument(
        "--difficulty_keep_ratio",
        type=float,
        default=0.6,
        help="Fraction of hardest examples to keep for selected sources",
    )
    parser.add_argument(
        "--strict_tulu",
        action="store_true",
        help="Keep only cleaner Tulu conversations: optional system + alternating user/assistant turns",
    )
    parser.add_argument(
        "--tulu_max_turns",
        type=int,
        default=4,
        help="Maximum number of non-system turns to keep for strict Tulu filtering",
    )
    parser.add_argument("--no_cache", action="store_true", help="Rebuild mixed dataset instead of reusing cached data")
    parser.add_argument(
        "--task_weights",
        type=str,
        default="gsm8k=1,tulu=1,code=1",
        help="Weighted task sampling, e.g. 'gsm8k=1,tulu=1.5,code=1'",
    )
    parser.add_argument(
        "--stage2_steps",
        type=int,
        default=0,
        help="Extra fine-tuning steps after stage 1 using stage2 task weights",
    )
    parser.add_argument(
        "--stage2_task_weights",
        type=str,
        default="gsm8k=3,tulu=0.5,code=0.5",
        help="Weighted task sampling for stage 2, e.g. 'gsm8k=3,tulu=0.5,code=0.5'",
    )
    parser.add_argument(
        "--stage2_tulu_source_names",
        type=str,
        default="",
        help="Optional Tulu source groups for stage 2 only. Empty means reuse stage 1 Tulu recipe.",
    )
    parser.add_argument(
        "--stage2_tulu_size",
        type=int,
        default=0,
        help="Optional Tulu size for stage 2 only. 0 means reuse stage 1 Tulu size.",
    )
    parser.add_argument(
        "--stage2_tulu_keyword_bias",
        action="store_true",
        help="Use IFEval-style keyword-biased Tulu sampling for stage 2 only.",
    )
    parser.add_argument(
        "--stage2_tulu_keyword_fraction",
        type=float,
        default=0.8,
        help="Target fraction of keyword-matched Tulu examples when stage-2 keyword bias is enabled.",
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="If > 0, save additional checkpoints every N global steps during training.",
    )
    parser.add_argument("--log_interval", type=int, default=10, help="How often to print training progress")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    args = parser.parse_args()
    quality_filter_sources = _parse_source_list(args.quality_filter_sources)
    difficulty_filter_sources = _parse_source_list(args.difficulty_filter_sources)
    tulu_source_names = _parse_tulu_source_names(args.tulu_source_names)
    stage2_tulu_source_names = (
        _parse_tulu_source_names(args.stage2_tulu_source_names)
        if args.stage2_tulu_source_names
        else tulu_source_names
    )

    # Setup
    print(f"Model: {args.model}")
    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Prepare training data
    print("Preparing training data...")
    conversations, task_counts = load_mixed_conversations(
        gsm8k_size=args.gsm8k_size,
        metamath_size=args.metamath_size,
        tulu_size=args.tulu_size,
        code_size=args.code_size,
        alpaca_size=args.alpaca_size,
        mbpp_size=args.mbpp_size,
        seed=args.seed,
        clean_data=args.clean_data,
        use_cache=not args.no_cache,
        strict_tulu=args.strict_tulu,
        tulu_max_turns=args.tulu_max_turns,
        dedup=args.dedup,
        quality_filter=args.quality_filter,
        quality_filter_sources=quality_filter_sources,
        difficulty_filter=args.difficulty_filter,
        difficulty_filter_sources=difficulty_filter_sources,
        difficulty_keep_ratio=args.difficulty_keep_ratio,
        tulu_source_names=tulu_source_names,
        tulu_keyword_bias=args.tulu_keyword_bias,
        tulu_keyword_fraction=args.tulu_keyword_fraction,
    )
    print(
        "  Loaded mixed conversations:"
        f" tulu={task_counts['tulu']}, gsm8k={task_counts['gsm8k']}, code={task_counts['code']}"
    )
    data_by_task = {task: [] for task in TASK_NAMES}
    skipped = 0
    for convo, source in conversations:
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=args.max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            data_by_task[source].append(datum)
        except Exception:
            skipped += 1
    total_examples = sum(len(v) for v in data_by_task.values())
    print(f"  {total_examples} training examples prepared")
    print(
        "  Prepared per task:"
        f" gsm8k={len(data_by_task['gsm8k'])}, tulu={len(data_by_task['tulu'])}, code={len(data_by_task['code'])}"
    )
    if skipped:
        print(f"  Skipped {skipped} examples during datum conversion")

    stage2_data_by_task = data_by_task
    stage2_recipe_changed = (
        args.stage2_steps > 0
        and (
            args.stage2_tulu_source_names
            or args.stage2_tulu_size > 0
            or args.stage2_tulu_keyword_bias
        )
    )
    if stage2_recipe_changed:
        print("Preparing stage-2 Tulu finisher data...")
        stage2_conversations, stage2_task_counts = load_mixed_conversations(
            gsm8k_size=0,
            metamath_size=0,
            tulu_size=args.stage2_tulu_size or args.tulu_size,
            code_size=0,
            alpaca_size=0,
            mbpp_size=0,
            seed=args.seed + 1000,
            clean_data=args.clean_data,
            use_cache=not args.no_cache,
            strict_tulu=args.strict_tulu,
            tulu_max_turns=args.tulu_max_turns,
            dedup=args.dedup,
            quality_filter=args.quality_filter,
            quality_filter_sources=quality_filter_sources,
            difficulty_filter=False,
            difficulty_filter_sources=difficulty_filter_sources,
            difficulty_keep_ratio=1.0,
            tulu_source_names=stage2_tulu_source_names,
            tulu_keyword_bias=args.stage2_tulu_keyword_bias,
            tulu_keyword_fraction=args.stage2_tulu_keyword_fraction,
        )
        stage2_tulu_data = []
        stage2_skipped = 0
        for convo, source in stage2_conversations:
            if source != "tulu":
                continue
            try:
                datum = conversation_to_datum(
                    convo,
                    renderer,
                    max_length=args.max_length,
                    train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                )
                stage2_tulu_data.append(datum)
            except Exception:
                stage2_skipped += 1
        stage2_data_by_task = {task: list(values) for task, values in data_by_task.items()}
        stage2_data_by_task["tulu"] = stage2_tulu_data
        print(
            "  Stage-2 Tulu prepared:"
            f" tulu={len(stage2_tulu_data)}"
            f" (raw_tulu={stage2_task_counts['tulu']}, skipped={stage2_skipped})"
        )

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.model, rank=args.rank)
    print("  Training client ready")
    saved_checkpoints = []

    # Train
    stage1_weights = _parse_task_weights(args.task_weights)
    stage2_weights = _parse_task_weights(args.stage2_task_weights)
    total_steps = args.num_steps + args.stage2_steps
    if total_examples == 0:
        raise RuntimeError("No training data available after preprocessing")

    def run_stage(stage_name, num_steps, weights, seed_offset, step_offset, stage_data_by_task):
        if num_steps <= 0:
            return
        sampler = TaskAwareSampler(
            data_by_task=stage_data_by_task,
            batch_size=args.batch_size,
            weights=weights,
            seed=args.seed + seed_offset,
        )
        print(
            f"\n{stage_name}: {num_steps} steps"
            f" | weights={weights}"
            f" | batch_size={args.batch_size}"
            f" | lr={args.lr}"
        )
        for local_step in range(num_steps):
            batch, batch_tasks = sampler.next_batch()
            global_step_idx = step_offset + local_step
            step_lr = _get_lr_with_warmup(
                global_step_idx,
                args.lr,
                args.warmup_steps,
                total_steps,
                args.min_lr_ratio,
            )
            adam_params = types.AdamParams(
                learning_rate=step_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )
            fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = tc.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
            weights_arr = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
            loss = -np.dot(logprobs, weights_arr) / max(weights_arr.sum(), 1)

            display_step = global_step_idx + 1
            if local_step == 0 or display_step % args.log_interval == 0 or display_step == total_steps:
                print(
                    f"  Step {display_step}/{total_steps}"
                    f" | stage={stage_name}"
                    f" | loss={loss:.4f}"
                    f" | lr={step_lr:.2e}"
                    f" | batch_tasks=[{_summarize_batch_tasks(batch_tasks)}]"
                )
            if args.save_every_steps > 0 and display_step % args.save_every_steps == 0 and display_step != total_steps:
                save_name = f"{args.checkpoint_name}-step{display_step}"
                print(f"  Saving intermediate checkpoint '{save_name}'...")
                ckpt = tc.save_weights_for_sampler(name=save_name).result()
                saved_checkpoints.append({"step": display_step, "name": save_name, "path": ckpt.path})
                print(f"    Saved: {ckpt.path}")

    run_stage("stage1_mixed", args.num_steps, stage1_weights, seed_offset=0, step_offset=0, stage_data_by_task=data_by_task)
    run_stage(
        "stage2_focus",
        args.stage2_steps,
        stage2_weights,
        seed_offset=1,
        step_offset=args.num_steps,
        stage_data_by_task=stage2_data_by_task,
    )

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # Save checkpoint info
    info = {
        "checkpoint_path": checkpoint_path,
        "base_model": args.model,
        "renderer_name": renderer_name,
        "training": {
            "num_steps": args.num_steps,
            "stage2_steps": args.stage2_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "lora_rank": args.rank,
            "task_weights": stage1_weights,
            "stage2_task_weights": stage2_weights,
            "stage2_tulu_source_names": sorted(stage2_tulu_source_names),
            "stage2_tulu_size": args.stage2_tulu_size,
            "stage2_tulu_keyword_bias": args.stage2_tulu_keyword_bias,
            "stage2_tulu_keyword_fraction": args.stage2_tulu_keyword_fraction,
            "max_length": args.max_length,
            "clean_data": args.clean_data,
            "dedup": args.dedup,
            "quality_filter": args.quality_filter,
            "quality_filter_sources": sorted(quality_filter_sources),
            "difficulty_filter": args.difficulty_filter,
            "difficulty_filter_sources": sorted(difficulty_filter_sources),
            "difficulty_keep_ratio": args.difficulty_keep_ratio,
            "tulu_source_names": sorted(tulu_source_names),
            "tulu_keyword_bias": args.tulu_keyword_bias,
            "tulu_keyword_fraction": args.tulu_keyword_fraction,
            "strict_tulu": args.strict_tulu,
            "tulu_max_turns": args.tulu_max_turns,
            "gsm8k_size": args.gsm8k_size,
            "metamath_size": args.metamath_size,
            "tulu_size": args.tulu_size,
            "code_size": args.code_size,
            "alpaca_size": args.alpaca_size,
            "mbpp_size": args.mbpp_size,
            "prepared_examples": {
                "gsm8k": len(data_by_task["gsm8k"]),
                "tulu": len(data_by_task["tulu"]),
                "code": len(data_by_task["code"]),
            },
        },
        "published": not args.no_publish,
        "saved_checkpoints": saved_checkpoints,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  python -m evaluation.eval_all --checkpoint_path \"{checkpoint_path}\" --base_model {args.model}")


if __name__ == "__main__":
    main()
