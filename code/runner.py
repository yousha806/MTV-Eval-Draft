import argparse
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # Concurrency limit (default 10)
    max_workers = 4

    print(f"Starting evaluation with {max_workers} workers...")
    
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, ex in enumerate(examples):
            # Optional URL sanity checks (only doing this in dry-run or before submission to avoid spamming logs in parallel)
            if args.dry_run:
                if not check_url(ex["src_image"]):
                    print(f"WARNING: src_image URL invalid: {ex['src_image']}")
                if not check_url(ex["adapted_image"]):
                    print(f"WARNING: adapted_image URL invalid: {ex['adapted_image']}")

            # Fill prompt template
            prompt = prompt_template.replace("{source_culture_text}", ex["source_culture"]).replace("{target_culture_text}", ex["target_culture"]).replace("{category}", ex["category"])

            # Submit task
            future = executor.submit(evaluator, prompt, ex["src_image"], ex["adapted_image"])
            futures[future] = ex

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures)):
            ex = futures[future]
            try:
                result_raw = future.result()
            except Exception as e:
                print(f"Error processing example: {e}")
                result_raw = {}

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
            f.flush() # Ensure it's written immediately

            # Print first dry-run example (we can't easily rely on idx==0 being first here, so just print the first one that returns)
            if args.dry_run and futures[future] == examples[0]:
                 # Note: this might not print if example[0] finishes late, but for dry-run it's fine
                 pass 

    print(f"\nDone. Results written to {out_path}")


if __name__ == "__main__":
    main()
