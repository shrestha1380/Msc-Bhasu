from llama_cpp import Llama

print("Loading Llama-3 8B (GGUF, 4-bit quantized)...")

llm = Llama.from_pretrained(
    repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=0,
    verbose=True,
)

print("\nGenerating response...\n")

output = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Explain the benefits of running large language models locally:"}
    ],
    max_tokens=300,
    temperature=0.7,
)

print(output["choices"][0]["message"]["content"])