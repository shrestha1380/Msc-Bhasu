import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_cpp import Llama
from utils.bigbench_utils import load_task, build_prompt, get_correct_answers, score_response

TASKS = ["date_understanding", "movie_recommendation", "causal_judgment", "logical_deduction"]
N_EXAMPLES = 20

print("Loading Llama-3-8B (GGUF)...")
llm = Llama.from_pretrained(
    repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=0,
    verbose=False,
)

def ask_llama3(prompt):
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0,
    )
    return output["choices"][0]["message"]["content"]

results = {}
for task in TASKS:
    print(f"\n=== Task: {task} ===")
    examples = load_task(task, n=N_EXAMPLES)
    correct_count = 0
    for i, example in enumerate(examples):
        prompt = build_prompt(example)
        response = ask_llama3(prompt)
        correct_answers = get_correct_answers(example)
        is_correct = score_response(response, correct_answers)
        correct_count += is_correct
        print(f"  [{i+1}/{len(examples)}] {'✓' if is_correct else '✗'}  Response: {response[:60]!r}")
    accuracy = correct_count / len(examples)
    results[task] = accuracy
    print(f"  Accuracy: {accuracy:.1%}")

print("\n=== Summary (Llama-3-8B) ===")
for task, acc in results.items():
    print(f"{task}: {acc:.1%}")