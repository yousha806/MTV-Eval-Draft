from pathlib import Path
import pandas as pd

#COUNTRIES = ["portugal","india","brazil","japan","nigeria","turkey","united-states"] 
COUNTRIES = ["india"]
def extract_source_country(url: str) -> str:
    parts = url.split("/")
    if "part1" in parts:
        idx = parts.index("part1")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"

def extract_category(url: str) -> str:
    """Extract category from URL (filename prefix before first underscore)"""
    parts = url.split("/")
    filename = parts[-1]
    return filename.split("_")[0]

def load_all_examples(root="data/part-1"):
    examples = []
    root_path = Path(root)

    for country in COUNTRIES:
        labels_path = root_path / country / "labels.csv"
        if not labels_path.exists():
            print(f"WARNING: CSV not found for country {country}: {labels_path}")
            continue

        df = pd.read_csv(labels_path)

        for row in df.itertuples(index=False):
            for i in range(1, 4):
                src_url = getattr(row, "src_image_path")
                examples.append({
                    "src_image": src_url,
                    "adapted_image": getattr(row, f"model_path_{i}"),
                    "model": getattr(row, f"model_{i}"),
                    "source_culture": extract_source_country(src_url),
                    "target_culture": country,
                    "category": extract_category(src_url),
                    "labels_file": str(labels_path),
                })

    print(f"Total examples found: {len(examples)}")
    return examples
