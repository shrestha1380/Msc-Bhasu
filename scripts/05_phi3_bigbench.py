import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.bigbench_utils import load_task, build_prompt, get_correct_answers, score_response

TASKS = ["date_understanding", "movie_recommendation", "causal_judgment", "logical_deduction"]
N_EXAMPLES = 20

print("Loading Phi-3-mini...")
model_name = "microsoft/Phi-3-mini-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cpu", torch_dtype="auto", attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def ask_phi3(prompt):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

results = {}
for task in TASKS:
    print(f"\n=== Task: {task} ===")
    examples = load_task(task, n=N_EXAMPLES)
    correct_count = 0
    for i, example in enumerate(examples):
        prompt = build_prompt(example)
        response = ask_phi3(prompt)
        correct_answers = get_correct_answers(example)
        is_correct = score_response(response, correct_answers)
        correct_count += is_correct
        print(f"  [{i+1}/{len(examples)}] {'✓' if is_correct else '✗'}  Response: {response[:60]!r}")
    accuracy = correct_count / len(examples)
    results[task] = accuracy
    print(f"  Accuracy: {accuracy:.1%}")

print("\n=== Summary (Phi-3-mini) ===")
for task, acc in results.items():
    print(f"{task}: {acc:.1%}")