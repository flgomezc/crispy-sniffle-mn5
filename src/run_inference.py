import argparse
import csv
import json
import os
from datetime import datetime

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
    parser.add_argument("--output-dir", default="results", help="Directory to write results")
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


def save_results(output_dir, model_name, split, dataset, predicted_labels, raw_outputs):
    os.makedirs(output_dir, exist_ok=True)
    slug = model_name.replace("/", "_")

    csv_path = os.path.join(output_dir, f"{slug}_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "true_label", "predicted_label", "raw_output"])
        writer.writeheader()
        for i, (sample, predicted, raw) in enumerate(zip(dataset, predicted_labels, raw_outputs)):
            writer.writerow({
                "id": i,
                "text": sample["text"],
                "true_label": LABEL_MAP[sample["label"]],
                "predicted_label": predicted,
                "raw_output": raw,
            })

    true_labels = [LABEL_MAP[s["label"]] for s in dataset]
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    n = len(dataset)

    from sklearn.metrics import f1_score
    f1 = f1_score(true_labels, predicted_labels, average="macro", labels=["Positive", "Negative"])

    summary = {
        "model": model_name,
        "split": split,
        "total": n,
        "correct": correct,
        "accuracy": round(correct / n, 4),
        "f1_macro": round(f1, 4),
        "unknown": predicted_labels.count("Unknown"),
        "timestamp": datetime.now(tz=__import__("datetime").timezone.utc).isoformat(),
    }

    json_path = os.path.join(output_dir, f"{slug}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Predictions saved → {csv_path}")
    print(f"Summary saved     → {json_path}")
    return summary


def main():
    args = parse_args()
    model_name = os.path.basename(args.model.rstrip("/"))
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

    print(f"\n--- Batch inference ({len(dataset)} samples) ---")
    prompts = [build_prompt(tokenizer, sample["text"]) for sample in dataset]
    outputs = llm.generate(prompts, params)

    raw_outputs = [o.outputs[0].text for o in outputs]
    predicted_labels = [parse_label(r) for r in raw_outputs]

    true_labels = [LABEL_MAP[s["label"]] for s in dataset]
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    unknown = predicted_labels.count("Unknown")

    print(f"Total samples  : {len(dataset)}")
    print(f"Correct        : {correct}")
    print(f"Unknown        : {unknown}")
    print(f"Accuracy       : {correct / len(dataset):.4f}")

    summary = save_results(args.output_dir, model_name, args.split, dataset, predicted_labels, raw_outputs)
    print(f"\nAccuracy       : {summary['accuracy']}")
    print(f"F1 (macro)     : {summary['f1_macro']}")


if __name__ == "__main__":
    main()
