# How to publish

> **Note (2026):** Hugging Face **Gradio Spaces often require PRO**.  
> Practical free options: **Render** (public Git repo) or run locally / Gradio `share=True`.

You only need the **`challenge/`** folder (~2 MB). Not the whole git monorepo.

## 1) Get the files on your PC

Option A — from git (if the repo is yours):

```bash
git clone <your-repo-url>
cd <repo>/challenge
```

Option B — download/copy the `challenge` directory from Cursor / zip.

Make sure these exist:

- `app.py`, `features.py`, `requirements.txt`, `README.md`
- `data/` (csv files)
- `model/model.joblib`, `model/metrics.json`
- `assets/orange_workflow.png`

You can delete before upload: `__pycache__/`, `model/last_batch_result.csv`, `ORANGE_GUIDE.md`, `PUBLISH.md` (optional).

## 2) Test locally

```bash
cd challenge
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7865 and click Research / Model / Predict.

## 3) Create a Hugging Face Space

1. Go to https://huggingface.co/spaces → **Create new Space**
2. **SDK:** Gradio  
3. Name e.g. `ai-exoplanets-koi`  
4. Visibility: Public (for teachers/peers) or Private  

## 4) Upload project files

Upload **contents of `challenge/`** into the Space root (not a nested `challenge/` folder):

```
app.py
features.py
requirements.txt
README.md          # already has Space YAML header
data/...
model/model.joblib
model/metrics.json
assets/orange_workflow.png
```

Ways to upload:

- Space → **Files** → Upload (simple), or  
- `huggingface-cli upload` / git clone the Space repo and push.

## 5) Wait for build

HF installs `requirements.txt` and starts `app.py`.  
First build may take a few minutes (CatBoost).  
If it sleeps on free tier, the first open can be slow — normal.

## 6) Share the link

Example: `https://huggingface.co/spaces/<your-username>/ai-exoplanets-koi`

That URL is what teachers/students can open asynchronously.
