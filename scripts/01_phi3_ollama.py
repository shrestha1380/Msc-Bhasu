from ollama import chat, pull
import sys

MODEL = "phi3"

def main():
    print(f"Pulling model '{MODEL}' if not already present...")
    try:
        pull(MODEL)
    except Exception as e:
        print(f"Warning while pulling: {e}")
        print("Make sure Ollama is installed and running (ollama serve).")
        sys.exit(1)

    print(f"\n=== Chatting with {MODEL} ===")
    print("Type your message and press Enter. Type 'exit' or 'quit' to stop.\n")

    messages = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = chat(model=MODEL, messages=messages)
            assistant_reply = response.message.content
            print(f"\nPhi-3: {assistant_reply}\n")
            messages.append({"role": "assistant", "content": assistant_reply})
        except Exception as e:
            print(f"Error: {e}")
            print("Is Ollama running? Try 'ollama serve' in another terminal.")

if __name__ == "__main__":
    main()
