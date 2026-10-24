import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-3-mini-128k-instruct"

print("Loading model... (first time will download ~7-8 GB)")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype="auto",
#    trust_remote_code=True,
    attn_implementation="eager",          # Important fix
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# Convert messages to the format Phi-3 expects
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nGenerating response...\n")

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=False,
    temperature=0.0,
    pad_token_id=tokenizer.eos_token_id,
)

# Only print the newly generated part
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print(response)