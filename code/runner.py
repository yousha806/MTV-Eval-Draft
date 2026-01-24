import argparse
import json
from pathlib import Path
from tqdm import tqdm

from loader import load_all_examples
from gemini_client import evaluate_with_gemini, dry_run_evaluate
from utils import check_url

# List of countries to iterate over

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without calling Gemini")
    parser.add_argument("--data-root", type=str, default="data/part1", help="Root folder for part1 data")
    parser.add_argument("--prompt-path", type=str, default="prompts/concept.txt", help="Prompt template path")
    parser.add_argument("--output-path", type=str, default="outputs/gemini_results.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    # Absolute paths
    script_dir = Path(__file__).parent
    data_root = Path(args.data_root).resolve()
    prompt_path = Path(args.prompt_path).resolve()
    out_path = Path(args.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompt template
    prompt_template = prompt_path.read_text()

    # Load all examples efficiently
    examples = load_all_examples()

    # Choose evaluator function
    evaluator = dry_run_evaluate if args.dry_run else evaluate_with_gemini

    with out_path.open("w") as f:
        for idx, ex in enumerate(tqdm(examples)):
            # Optional URL sanity checks
            if args.dry_run:
                if not check_url(ex["src_image"]):
                    print(f"WARNING: src_image URL invalid: {ex['src_image']}")
                if not check_url(ex["adapted_image"]):
                    print(f"WARNING: adapted_image URL invalid: {ex['adapted_image']}")

            # Fill prompt template
            prompt = prompt_template.replace("{source_culture_text}", ex["source_culture"]).replace("{target_culture_text}", ex["target_culture"])

            # Call evaluator
            result_raw = evaluator(prompt, ex["src_image"], ex["adapted_image"])

            # Ensure result is a dict
            if isinstance(result_raw, str):
                try:
                    result = json.loads(result_raw)
                except json.JSONDecodeError:
                    # fallback in case of malformed output
                    result = {
                        "A_cultural_appropriateness": None,
                        "B_semantic_preservation": None,
                        "C_visual_coherence": None,
                        "overall_success": None
                    }
            elif isinstance(result_raw, dict):
                result = result_raw
            else:
                result = {}

            # Clean keys: remove whitespace and quotes
            result = {k.strip().strip('"'): v for k, v in result.items()}

            # Combine example data and evaluation
            record = {**ex, "evaluation": result}

            # Write JSONL
            f.write(json.dumps(record) + "\n")

            # Print first dry-run example
            if args.dry_run and idx == 0:
                print("\n✔ Dry-run example:\n")
                print(json.dumps(record, indent=2))

    print(f"\nDone. Results written to {out_path}")


if __name__ == "__main__":
    main()
