import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Model path     : {args.model}")
    print(f"Tensor parallel: {args.tensor_parallel_size}")

    print("Loading model with vLLM...")
    from vllm import LLM
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    print("Model loaded successfully.")

if __name__ == "__main__":
    main()
