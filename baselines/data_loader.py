from pathlib import Path
import pandas as pd

#COUNTRIES = ["brazil", "india", "portugal", "japan", "nigeria", "turkey", "united-states"]
COUNTRIES = [ "india"]

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
    """Load examples from baselines data folder"""
    examples = []
    root_path = Path(root)

    for country in COUNTRIES:
        labels_path = root_path / country / "labels.csv"
        if not labels_path.exists():
            print(f"WARNING: CSV not found for country {country}: {labels_path}")
            continue

        df = pd.read_csv(labels_path)

        for row in df.itertuples(index=False):
            src_url = getattr(row, "src_image_path")
            examples.append({
                "src_image": src_url,
                "source_culture": extract_source_country(src_url),
                "target_culture": country,
                "category": extract_category(src_url),
                "labels_file": str(labels_path),
            })

    print(f"Total examples found: {len(examples)}")
    return examples


def load_from_huggingface(dataset_name: str = "concept", target_countries=None):
    """
    Load examples from Hugging Face dataset (cmu-lti/machine-translation-for-vision)
    
    Args:
        dataset_name: "concept" or "application" (these are splits within the default config)
        target_countries: List of target countries to filter by (e.g., ["india", "japan"])
                         If None, uses all available target countries from each example
    
    Returns:
        List of examples with keys: src_image, source_culture, target_culture, category
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package not found. Install with: pip install datasets")
    
    if dataset_name not in ["concept", "application"]:
        raise ValueError("dataset_name must be 'concept' or 'application'")
    
    print(f"Loading dataset from Hugging Face...")
    ds = load_dataset("cmu-lti/machine-translation-for-vision", "default")
    
    if dataset_name not in ds:
        raise ValueError(f"Split '{dataset_name}' not found. Available: {list(ds.keys())}")
    
    dataset = ds[dataset_name]
    print(f"Processing split: {dataset_name}")
    
    examples = []
    
    for row in dataset:
        source_country = row.get("source_country", "unknown")
        image_path = row.get("image_path")
        category = row.get("category")
        target_countries_list = row.get("target_countries", [])
        
        # Skip if no image path or target countries
        if not image_path or not target_countries_list:
            continue
        
        # Parse target countries (they may be comma-separated string or list)
        if isinstance(target_countries_list, str):
            target_countries_list = [tc.strip() for tc in target_countries_list.split(",")]
        
        # Filter by target countries if specified
        if target_countries:
            target_countries_list = [tc for tc in target_countries_list if tc in target_countries]
        
        # Create an example for each target country
        for target_country in target_countries_list:
            examples.append({
                "src_image": image_path,
                "source_culture": source_country,
                "target_culture": target_country,
                "category": category,
                "split": dataset_name,
            })
    
    print(f"Total examples loaded from Hugging Face: {len(examples)}")
    return examples