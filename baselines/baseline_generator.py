import argparse
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

from llm_generator import (
    generate_adapted_image_with_litellm,
    generate_adapted_image_with_nanobanana,
    extract_category,
    extract_source_country,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
from data_loader import load_all_examples, load_from_huggingface

def generate_baseline(example, generator_type="nanobanana"):
    try:
        src_url = example["src_image"]
        target_culture = example["target_culture"]
        category = example["category"]
        
        # Choose the appropriate generator function
        if generator_type == "nanobanana":
            adapted_url = generate_adapted_image_with_nanobanana(src_url, target_culture, category)
        else:  # gemini or default
            adapted_url = generate_adapted_image_with_litellm(src_url, target_culture, category)
        
        result = {
            **example,
            "adapted_image_url": adapted_url,
            "generator": generator_type,
        }
        return result
        
    except Exception as e:
        error_msg = str(e)
        return {
            **example,
            "adapted_image_url": None,
            "error": error_msg,
            "generator": generator_type,
        }


def main():
    parser = argparse.ArgumentParser(description="Generate baseline adapted images using LLM")
    parser.add_argument(
        "--generator",
        choices=["gemini", "nanobanana"],
        default="gemini",
        help="Which image generation API to use"
    )
    parser.add_argument(
        "--data-source",
        choices=["local", "huggingface"],
        default="local",
        help="Where to load data from: 'local' for local files, 'huggingface' for HF dataset"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["concept", "application"],
        default="concept",
        help="Dataset split when using --data-source huggingface (concept or application)"
    )
    parser.add_argument(
        "--target-countries",
        type=str,
        nargs="+",
        default=None,
        help="Target countries to filter by (only with --data-source huggingface). E.g., india japan brazil"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/part-1",
        help="Root folder for part1 data (only used with --data-source local)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/baseline_generated.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to existing output file to resume from (skips already processed examples)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually call the API, just show what would be generated"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of parallel workers (reduce if hitting API rate limits)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data from specified source
    if args.data_source == "huggingface":
        examples = load_from_huggingface(
            dataset_name=args.dataset_type,
            target_countries=args.target_countries
        )
    else:
        examples = load_all_examples(args.data_root)
    
    # Resume from existing file if specified
    processed_pairs = set()
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"Reading processed items from {args.resume_from}...")
            with resume_path.open("r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        src = record.get("src_image")
                        tgt = record.get("target_culture")
                        if src and tgt:
                            processed_pairs.add((src, tgt))
                    except json.JSONDecodeError:
                        continue
            
            # Filter out already processed examples
            original_count = len(examples)
            examples = [ex for ex in examples if (ex["src_image"], ex["target_culture"]) not in processed_pairs]
            print(f"Resuming: skipped {original_count - len(examples)} already processed, {len(examples)} remaining")
            
            # Use the resume file as output path
            output_path = resume_path
        else:
            print(f"WARNING: Resume file not found: {args.resume_from}")
    
    if args.max_samples:
        examples = examples[:args.max_samples]
    
    print(f"Processing {len(examples)} examples with {args.generator} generator...")
    
    if args.dry_run:
        print("DRY RUN MODE - No API calls will be made")
        with output_path.open("w") as f:
            for example in examples[:3]:
                result = {
                    **example,
                    "adapted_image_url": f"[DRY RUN] Would generate image for {example['src_image']}",
                    "generator": args.generator,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Dry run preview written to {output_path}")
        return
    
    with output_path.open("a" if args.resume_from else "w") as f, ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(generate_baseline, ex, args.generator): ex
            for ex in examples
        }
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"Error: {e}")
    
    print(f"\nDone. Results written to {output_path}")


if __name__ == "__main__":
    main()
