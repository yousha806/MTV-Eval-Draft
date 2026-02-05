import os
import requests
from huggingface_hub import InferenceClient
from pathlib import Path



def load_image_bytes(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.content

def generate_adapted_image_with_litellm(src_url, target_culture, category):
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY environment variable not set!")
    
    input_image_bytes = load_image_bytes(src_url)
    
    client = InferenceClient(provider="replicate", api_key=api_key)
    
    prompt = f"""Edit the given image so that it reflects the cultural context of {target_culture},
while keeping the same category as the original image.

Make culturally appropriate changes to appearance, environment, materials, colors,
symbols, or presentation. Do not change the category of the object.

Target culture: {target_culture}
Category: {category}"""
    
    try:
        adapted_image = client.image_to_image(
            input_image_bytes,
            prompt=prompt,
            model="black-forest-labs/FLUX.2-dev"
        )
    except Exception as e:
        print(f"API error for {src_url}: {e}")
        raise
    
    output_dir = Path("outputs/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = src_url.split('/')[-1]
    output_path = output_dir / f"adapted_{filename}"
    
    try:
        if adapted_image is None:
            raise ValueError("API returned None")
        
        if hasattr(adapted_image, 'save'):
            adapted_image.save(output_path, quality=95)
        elif isinstance(adapted_image, bytes):
            with open(output_path, 'wb') as f:
                f.write(adapted_image)
        else:
            print(f"Warning: Unknown image type {type(adapted_image)}")
            with open(output_path, 'wb') as f:
                f.write(bytes(adapted_image))
        
        file_size = output_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Saved image is empty at {output_path}")
        
        print(f"✓ Saved {file_size} bytes to {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        raise

def generate_adapted_image_with_nanobanana(src_url, target_culture, category):
    return generate_adapted_image_with_litellm(src_url, target_culture, category)

def extract_category(url: str) -> str:
    parts = url.split("/")
    filename = parts[-1]
    return filename.split("_")[0]

def extract_source_country(url: str) -> str:
    parts = url.split("/")
    if "part1" in parts:
        idx = parts.index("part1")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"
