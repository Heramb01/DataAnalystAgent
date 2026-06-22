# Autonomous Data Analyst Agent

Upload a CSV/Excel file → get automatic data profiling, statistical insights,
and interactive charts. Fully open-source, runs free.

## What fixed your original error

Your `"Failed to fetch"` error came from calling HuggingFace's **hosted**
Inference API. That fails for one of these reasons:
1. The model isn't available on free serverless inference (HF has shrunk free coverage a lot).
2. The model is gated — you must click "Agree and access repository" on the model's HF page first.
3. The token is missing the `Bearer ` prefix or is invalid.
4. You called it from browser JavaScript (CORS blocks it) instead of from a backend.

**This project sidesteps the problem entirely by default**: the narrative
layer runs a small open-source model (`google/flan-t5-small`) **locally** —
no token, no network call at inference time, so "Failed to fetch" cannot
happen. If you specifically want to use HuggingFace's hosted API anyway, a
corrected version of that call is included (`app/narrate.py ->
narrate_via_hf_api`) with the fixes for all four issues above, selectable in
the app's sidebar.

## Project structure
```
data-analyst-agent/
├── app/
│   ├── main.py        # Streamlit UI — run this
│   ├── analyzer.py     # profiling + rule-based insights (no API needed)
│   ├── charts.py        # auto chart generation (Plotly)
│   └── narrate.py       # optional LLM narration (local model OR HF API)
├── requirements.txt
└── README.md
```

## 1. Run it locally (validate before deploying)

```bash
# Clone/copy this folder, then:
cd data-analyst-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app/main.py
```

Streamlit opens at `http://localhost:8501`. Upload a CSV/XLSX, or click
**"Generate sample sales dataset"** on the home screen to test instantly
without your own file.

### Validating it works
- Upload any CSV/Excel → you should see row/column counts, then tabs for
  **Insights**, **Charts**, **Numeric Stats**, **Raw Profile**.
- Click **"Generate narrative summary"** with the default "Local model"
  option selected — first run downloads ~80MB once, then works fully offline.
- If you want to test the HuggingFace API path instead: switch the sidebar
  radio to "HuggingFace Hosted API", paste a free token from
  https://huggingface.co/settings/tokens, and use a model confirmed to support
  free Inference API (e.g. `HuggingFaceH4/zephyr-7b-beta`, or check the
  "Inference API" widget is present on the model's page before using it).

## 2. Deploy it for free (pick one)

### Option A — Streamlit Community Cloud (easiest, fully free)
1. Push this folder to a public GitHub repo.
2. Go to https://streamlit.io/cloud → "New app".
3. Point it at your repo, set the main file path to `app/main.py`.
4. Deploy. You get a free public URL (`https://yourapp.streamlit.app`).
5. If using the HF API option, add your token as a secret in
   "Advanced settings → Secrets" as `HF_TOKEN = "hf_..."` instead of typing
   it into the UI each time.

### Option B — Hugging Face Spaces (free, open-source, supports Streamlit natively)
1. Create a free account at https://huggingface.co.
2. Go to "New Space" → choose **Streamlit** as the SDK → free CPU tier.
3. Upload these files (or `git push` to the Space's git repo) — make sure
   `app/main.py` is referenced, or rename it to `app.py` at the repo root
   (HF Spaces defaults to looking for `app.py`).
4. The Space auto-builds from `requirements.txt` and deploys at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.
5. If using the HF API narration option, add `HF_TOKEN` under
   Space Settings → "Repository secrets" rather than hardcoding it.

### Option C — Render.com free web service
1. Push to GitHub.
2. New "Web Service" on Render → connect repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0`
5. Free tier deploys automatically on every push.

## 3. Notes on the local LLM model
- `google/flan-t5-small` is intentionally small so it runs on a free CPU tier
  with no GPU. It's good for summarizing bullet-point insights into a
  paragraph, not for complex reasoning.
- If you want noticeably better narration quality and have more compute
  available (a paid Space tier, or a machine with a GPU), swap the model
  name in `narrate.py`'s `_get_local_pipeline()` for something like
  `google/flan-t5-base` or `google/flan-t5-large` — same code, just a
  bigger download.
- The app **never crashes** if the LLM layer fails for any reason (model
  download blocked, out of memory, etc.) — it automatically falls back to
  showing the raw rule-based insights instead.

## 4. Extending the agent
- `analyzer.py` is independent of Streamlit — you can call its functions
  from a script, a FastAPI backend, or a notebook.
- To add more chart types, add a function to `charts.py` and call it from
  the "Charts" tab in `main.py`.
- To support more file types (JSON, Parquet), extend `load_file()` in
  `analyzer.py`.
