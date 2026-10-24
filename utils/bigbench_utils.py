from datasets import load_dataset
import time

import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download

def load_task(task_name, split="validation", n=20):
    """Load a small subset of a BIG-bench task by discovering the actual
    file path via the Hugging Face API, rather than guessing a URL pattern
    (which varies across tasks in this dataset)."""
    all_files = list_repo_files("tasksource/bigbench", repo_type="dataset")
    task_files = [f for f in all_files if f.startswith(f"{task_name}/")]

    split_files = [f for f in task_files if split in f]
    if not split_files:
        split_files = [f for f in task_files if "train" in f]
    if not split_files:
        raise FileNotFoundError(f"No parquet file found for task '{task_name}'")

    local_path = hf_hub_download(
        repo_id="tasksource/bigbench", repo_type="dataset", filename=split_files[0]
    )
    df = pd.read_parquet(local_path)
    df = df.head(n)
    return df.to_dict("records")

def build_prompt(example):
    choices = example["multiple_choice_targets"]
    if len(choices) == 0:
        # Generative task — no fixed choices to offer.
        return f"{example['inputs']}\n\nAnswer:"
    choice_text = "\n".join(f"- {c}" for c in choices)
    return (
        f"{example['inputs']}\n\n"
        f"Choices:\n{choice_text}\n\n"
        f"Answer with the exact text of the correct choice only."
    )

def get_correct_answers(example):
    targets = example["multiple_choice_targets"]
    scores = example["multiple_choice_scores"]
    if len(targets) == 0:
        # Generative task — the correct answer(s) live in `targets`, not the choice list.
        return set(example["targets"])
    return {t for t, s in zip(targets, scores) if s == 1}

def score_response(response_text, correct_answers):
    """Simple containment-based match: does the model's response
    include (or closely match) one of the correct answers?"""
    response_lower = response_text.strip().lower()
    return any(ans.strip().lower() in response_lower for ans in correct_answers)

def run_task_eval(task_name, ask_fn, n=20):
    """Run one BIG-bench task through a model, recording accuracy and timing.
    ask_fn: a function that takes a prompt string and returns the model's response string.
    """
    examples = load_task(task_name, n=n)
    correct_count = 0
    total_time = 0.0

    for example in examples:
        prompt = build_prompt(example)
        start = time.time()
        response = ask_fn(prompt)
        elapsed = time.time() - start
        total_time += elapsed

        correct_answers = get_correct_answers(example)
        if score_response(response, correct_answers):
            correct_count += 1

    accuracy = correct_count / len(examples)
    avg_time_per_example = total_time / len(examples)

    return {
        "task": task_name,
        "n_examples": len(examples),
        "accuracy": accuracy,
        "total_time_sec": round(total_time, 2),
        "avg_time_per_example_sec": round(avg_time_per_example, 2),
    }