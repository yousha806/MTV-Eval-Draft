import os
import requests
from huggingface_hub import InferenceClient
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image
import io
from tenacity import retry, stop_after_attempt, wait_exponential
import unicodedata
import time


def sanitize_filename(filename):
    """Remove unicode/special characters from filename"""
    # Normalize unicode characters (decompose accented characters)
    normalized = unicodedata.normalize('NFKD', filename)
    # Keep only ASCII characters
    sanitized = normalized.encode('ASCII', 'ignore').decode('ASCII')
    return sanitized



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
    # Sanitize filename to remove unicode characters
    filename = sanitize_filename(filename)
    output_path = output_dir / f"adapted_{filename}"
    
    try:
        if adapted_image is None:
            raise ValueError("API returned None")
        
        if hasattr(adapted_image, 'save'):
            # Save with quality parameter only for formats that support it
            try:
                adapted_image.save(output_path, quality=95)
            except TypeError:
                # Format doesn't support quality parameter
                adapted_image.save(output_path)
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
    """Generate adapted image using Gemini's image generation capability (nanobanana model)"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key is None:
        raise ValueError("GEMINI_API_KEY environment variable not set!")
    
    client = genai.Client(api_key=gemini_key)
    
    # Add a small delay to avoid rate limiting
    time.sleep(0.5)
    
    # Load source image
    input_image_bytes = load_image_bytes(src_url)
    image_pil = Image.open(io.BytesIO(input_image_bytes))
    
    prompt = f"""Edit the given image so that it reflects the cultural context of {target_culture},
while keeping the same category as the original image.

Make culturally appropriate changes to appearance, environment, materials, colors,
symbols, or presentation. Do not change the category of the object.

Target culture: {target_culture}
Category: {category}"""
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call():
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, image_pil],
        )
        return response
    
    try:
        response = _call()
        
        # Extract generated image from response
        adapted_image = None
        for part in response.parts:
            if part.inline_data is not None:
                adapted_image = part.as_image()
                break
        
        if adapted_image is None:
            raise ValueError("No image generated in response")
        
        # Save the generated image
        output_dir = Path("outputs/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = src_url.split('/')[-1]
        # Sanitize filename to remove unicode characters
        filename = sanitize_filename(filename)
        output_path = output_dir / f"adapted_{filename}"
        
        # Save with quality parameter only for formats that support it
        try:
            adapted_image.save(output_path, quality=95)
        except TypeError:
            # Format doesn't support quality parameter
            adapted_image.save(output_path)
        
        file_size = output_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Saved image is empty at {output_path}")
        
        print(f"✓ Saved {file_size} bytes to {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"Error generating image with nanobanana for {src_url}: {e}")
        raise

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
