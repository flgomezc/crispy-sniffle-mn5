import argparse

SYSTEM_PROMPT = (
    "You are a sentiment classifier. "
    "Classify the sentiment of the movie review as exactly one word: "
    "Positive or Negative. Output nothing else."
)

LABEL_MAP = {0: "Negative", 1: "Positive"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--single", action="store_true", help="Run single inference only (Feature 3)")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    return parser.parse_args()


def build_prompt(tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_label(raw):
    raw = raw.strip().lower()
    if "positive" in raw:
        return "Positive"
    if "negative" in raw:
        return "Negative"
    return "Unknown"


def main():
    args = parse_args()
    print(f"Model path     : {args.model}")
    print(f"Tensor parallel: {args.tensor_parallel_size}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from datasets import load_dataset

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading model with vLLM...")
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    print("Model loaded successfully.")

    dataset = load_dataset("rotten_tomatoes", split=args.split)
    params = SamplingParams(temperature=0, max_tokens=8)

    if args.single:
        print("\n--- Feature 3: single inference ---")
        sample = dataset[0]
        text = sample["text"]
        true_label = LABEL_MAP[sample["label"]]
        prompt = build_prompt(tokenizer, text)
        outputs = llm.generate([prompt], params)
        raw = outputs[0].outputs[0].text
        predicted = parse_label(raw)
        print(f"Text           : {text[:80]}...")
        print(f"True label     : {true_label}")
        print(f"Raw output     : {repr(raw)}")
        print(f"Predicted      : {predicted}")
        print(f"Correct        : {predicted == true_label}")
        return

    print(f"\n--- Feature 4: batch inference ({len(dataset)} samples) ---")
    prompts = [build_prompt(tokenizer, sample["text"]) for sample in dataset]
    true_labels = [LABEL_MAP[sample["label"]] for sample in dataset]

    outputs = llm.generate(prompts, params)

    predicted_labels = [parse_label(o.outputs[0].text) for o in outputs]
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    unknown = predicted_labels.count("Unknown")

    print(f"Total samples  : {len(dataset)}")
    print(f"Correct        : {correct}")
    print(f"Unknown        : {unknown}")
    print(f"Accuracy       : {correct / len(dataset):.4f}")


if __name__ == "__main__":
    main()
