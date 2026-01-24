from pathlib import Path
import pandas as pd

COUNTRIES = ["portugal","india","brazil","japan","nigeria","turkey","united-states"] 

def extract_source_country(url: str) -> str:
    parts = url.split("/")
    if "part1" in parts:
        idx = parts.index("part1")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"

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
                examples.append({
                    "src_image": getattr(row, "src_image_path"),
                    "adapted_image": getattr(row, f"model_path_{i}"),
                    "model": getattr(row, f"model_{i}"),
                    "source_culture": extract_source_country(getattr(row, "src_image_path")),
                    "target_culture": country,
                    "labels_file": str(labels_path),
                })

    print(f"Total examples found: {len(examples)}")
    return examples
