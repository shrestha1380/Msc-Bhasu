import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from utils.bigbench_utils import load_task

MODEL_NAME = "facebook/nllb-200-1.3B"
print("Loading NLLB...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def translate(text, src_lang="eng_Latn", tgt_lang="fra_Latn"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated = model.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang), max_length=200
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

# --- Part A: BIG-bench conlang_translation task (expected to perform poorly) ---
print("\n=== BIG-bench: conlang_translation ===")
print("(Note: this task requires learning a fictional language from in-context")
print(" examples — NLLB is a supervised MT model and cannot do this. Included")
print(" for completeness / to document the limitation.)\n")

examples = load_task("conlang_translation", n=5)
for i, example in enumerate(examples):
    print(f"  [{i+1}] Input: {example['inputs'][:100]}...")
    print(f"      Expected: {example['targets']}")
    # NLLB has no sensible source language tag for this — skipped, documented only.

# --- Part B: standard translation sanity check (meaningful NLLB result) ---
print("\n=== Standard translation check (English -> French) ===")
test_sentences = [
    "The weather is beautiful today.",
    "I would like to book a table for two.",
    "Machine learning models can process large datasets.",
]
for sentence in test_sentences:
    translation = translate(sentence)
    print(f"  EN: {sentence}")
    print(f"  FR: {translation}\n")