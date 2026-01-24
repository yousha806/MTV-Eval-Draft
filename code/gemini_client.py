import json

#Dry run without API key
def dry_run_evaluate(prompt, src_url, adapted_url):
    return {
        "__dry_run__": True,
        "prompt_preview": prompt,
        "source_image_url": src_url,
        "adapted_image_url": adapted_url,
    }


#Actual Gemini Call
def evaluate_with_gemini(prompt, src_url, adapted_url):
    import google.generativeai as genai
    from tenacity import retry, stop_after_attempt, wait_fixed

    genai.configure()

    model = genai.GenerativeModel(
        "gemini-1.5-pro",
        generation_config={
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
        },
    )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call():
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": src_url},
            {"mime_type": "image/jpeg", "data": adapted_url},
        ])
        return json.loads(response.text)

    return _call()
