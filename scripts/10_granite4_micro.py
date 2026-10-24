import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ibm-granite/granite-4.0-micro"

print("Loading granite-4.0-micro...")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nGenerating response...\n")
outputs = model.generate(
    **inputs, max_new_tokens=300, do_sample=False,
    temperature=None, top_p=None, top_k=None,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))