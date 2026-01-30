import json
import requests
import os
from tenacity import retry, stop_after_attempt, wait_fixed
from google import genai
from google.genai import types

def load_image_bytes(url):
    """Download image from URL and return bytes"""
    return requests.get(url).content

def extract_json(text):
    """Extract JSON from LLM output"""
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in response")
    return json.loads(match.group())

def dry_run_evaluate(prompt, src_url, adapted_url):
    return {
        "__dry_run__": True,
        "prompt_preview": prompt,
        "source_image_url": src_url,
        "adapted_image_url": adapted_url,
    }

def evaluate_with_gemini(prompt, src_url, adapted_url):
    """Evaluate images using Gemini flash models with a single prompt"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key is None:
        raise ValueError("GEMINI_API_KEY environment variable not set!")

    client = genai.Client(api_key=gemini_key)

    # Load images
    src_bytes = load_image_bytes(src_url)
    adapted_bytes = load_image_bytes(adapted_url)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call():
        response = client.models.generate_content(
            model="gemini-3-flash-preview",  # or "gemini-3-flash-preview" if available
            contents=[
                types.Part.from_bytes(data=src_bytes, mime_type="image/jpeg"),
                types.Part.from_bytes(data=adapted_bytes, mime_type="image/jpeg"),
                prompt  # single prompt for the model
            ]
        )
        return extract_json(response.text)

    return _call()
