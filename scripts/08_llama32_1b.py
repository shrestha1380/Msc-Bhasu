import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_TOKEN", "hf_xxxxxxxxxxxxxxxxxxxxx ")


model_name = "meta-llama/Llama-3.2-1B"

print("Loading Llama-3.2-1B...")

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cpu", torch_dtype="auto", attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Base model — no chat template. Use a plain completion-style prompt instead.
prompt = "Question: What is the capital of France?\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nGenerating response...\n")

outputs = model.generate(
    **inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)