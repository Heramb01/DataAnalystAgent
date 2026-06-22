"""
Optional natural-language narration layer for the Autonomous Data Analyst Agent.

IMPORTANT — this fixes the "Failed to fetch" error:
That error happens when calling HuggingFace's hosted Inference API (a network
request to huggingface.co) and the model/token/route isn't valid, or the model
has been removed from free serverless inference, or you're calling it from a
browser (CORS). 

This module avoids that entirely by default: it runs a small open-source model
LOCALLY via the `transformers` library. No token, no internet call, no "Failed
to fetch" possible, because nothing is fetched over the network at inference time.

If you still want to use HuggingFace's hosted API instead of running locally,
a corrected, working version of that call is provided in `narrate_via_hf_api()`
below — read the comments there for exactly what was likely wrong.
"""

import os
import requests

_LOCAL_PIPE = None


def _get_local_pipeline():
    """
    Lazily load a small, free, open-source local text-generation model.
    flan-t5-small is ~80MB, runs on CPU, and needs no token or internet
    access after the first download (weights are cached locally).
    """
    global _LOCAL_PIPE
    if _LOCAL_PIPE is None:
        from transformers import pipeline
        _LOCAL_PIPE = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=-1,  # CPU; set to 0 if you have a CUDA GPU available
        )
    return _LOCAL_PIPE


def narrate_locally(insights: list, max_new_tokens: int = 200) -> str:
    """
    Turns a list of rule-based insight strings into a short narrative summary
    using a local, open-source model. No API key required.
    """
    if not insights:
        return "No insights were generated for this dataset."

    bullet_text = "\n".join(f"- {i}" for i in insights[:12])
    prompt = (
        "Summarize the following data analysis findings into a short, "
        "clear paragraph for a business audience:\n" + bullet_text
    )

    try:
        pipe = _get_local_pipeline()
        result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return result[0]["generated_text"].strip()
    except Exception as e:
        # Local model failed to load (e.g. first-time download blocked by
        # no internet). Gracefully fall back to the rule-based bullets
        # themselves, so the app never crashes because of the LLM layer.
        return (
            "Local narration model unavailable (" + str(e) + "). "
            "Here are the raw findings instead:\n" + bullet_text
        )


def narrate_via_hf_api(insights: list, hf_token: str,
                        model: str = "HuggingFaceH4/zephyr-7b-beta") -> str:
    """
    CORRECTED version of a HuggingFace hosted Inference API call.

    Common causes of your original "Failed to fetch" error, and the fixes
    applied here:
      1. Wrong endpoint shape — must be exactly:
         https://api-inference.huggingface.co/models/<org>/<model-name>
      2. Missing/incorrect Authorization header — must be:
         "Authorization": "Bearer hf_xxx..."  (note the "Bearer " prefix)
      3. Model not available on free serverless inference, or gated/private —
         you must visit the model page on huggingface.co and click
         "Agree and access repository" if it's gated, and verify the model
         supports the Inference API (look for the "Inference API" widget
         on the model's page).
      4. Calling this directly from browser JS (e.g. inside a React artifact)
         hits CORS restrictions — hence "Failed to fetch". This function is
         meant to be called from a Python backend (like this Streamlit app),
         which has no CORS restriction.
      5. Cold-start timeouts — the first call to a model can take 20-30s
         while it loads on HF's servers; we set a generous timeout and
         handle the 503 "loading" response correctly below.
    """
    if not hf_token:
        raise ValueError("No HuggingFace token provided. Get a free one at "
                          "https://huggingface.co/settings/tokens (read access is enough).")

    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    bullet_text = "\n".join(f"- {i}" for i in insights[:12])
    prompt = (
        "Summarize the following data analysis findings into a short, "
        "clear paragraph for a business audience:\n" + bullet_text
    )

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.3},
        "options": {"wait_for_model": True},  # handles the cold-start 503 automatically
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        # This is the Python equivalent of "Failed to fetch" — a network-level
        # failure (DNS, timeout, connection refused, SSL, etc.)
        raise RuntimeError(f"Network error calling HuggingFace API: {e}")

    if response.status_code == 401:
        raise RuntimeError("Invalid HuggingFace token (401 Unauthorized). "
                            "Double check the token at https://huggingface.co/settings/tokens")
    if response.status_code == 404:
        raise RuntimeError(f"Model '{model}' not found (404). Check the exact model name "
                            f"on huggingface.co — it's case-sensitive and must include the org prefix.")
    if response.status_code == 503:
        raise RuntimeError("Model is loading on HuggingFace's servers (503). "
                            "Try again in about 20 seconds.")
    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace API returned {response.status_code}: {response.text[:300]}")

    data = response.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    return str(data)


def narrate(insights: list, mode: str = "local", hf_token: str = None, hf_model: str = None) -> str:
    """Single entry point used by the app. mode = 'local' or 'hf_api'."""
    if mode == "hf_api":
        token = hf_token or os.environ.get("HF_TOKEN", "")
        model = hf_model or "HuggingFaceH4/zephyr-7b-beta"
        return narrate_via_hf_api(insights, token, model)
    return narrate_locally(insights)
