from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

MODEL_NAME = "facebook/nllb-200-1.3B"

print("Loading NLLB translation model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def translate(text, src_lang="eng_Latn", tgt_lang="fra_Latn"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=200
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

print("\n=== Translation Demo ===")
text = "Hello, how are you today?"
result = translate(text, src_lang="eng_Latn", tgt_lang="fra_Latn")
print(f"English : {text}")
print(f"French  : {result}")
