# Running TARA on Windows

> Paths below assume you unzipped the project to `C:\TARA` (so `C:\TARA` contains
> `backend\`, `frontend\`, `start\`, `requirements.txt`, …). Adjust if you used a different folder.

## One-time setup
1. Install **Python 3.10 or 3.11** from python.org — during install, **tick "Add Python to PATH"**.
2. Unzip the project (e.g. to `C:\TARA\`), open Command Prompt:
   ```
   cd C:\TARA
   start\setup.bat
   ```
   This rebuilds the `.venv` environment from `requirements.txt` (needs internet once, ~10 min).

   **If it fails on `batman-package`** (needs a C compiler): delete that line from
   `requirements.txt` and run `start\setup.bat` again — it's only used by one Colab
   notebook, never by the app.

## Run it (two windows)
**Window 1 — the API:**
```
cd C:\TARA
.venv\Scripts\uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```
Allow it through Windows Firewall if prompted. First boot warms up for ~1–2 min.

**Window 2 — the website:**
```
cd C:\TARA\frontend
python -m http.server 5196
```
Open **http://localhost:5196/dash-workspace.html**

## Notes
- Demo stars answer instantly (their results ship in `backend\data\cache\`).
  Analyzing a NEW star needs internet (downloads from NASA MAST, ~15–90 s).
- `backend\precompute_cache.py` is Mac/Linux-only (SIGALRM) — not needed; the cache is included.
- No GPU needed — inference runs on CPU; training happens in the Colab notebooks under `notebooks\`.
- Node.js is only needed if you edit styles (`frontend\build-tw.sh` recompiles the CSS);
  the compiled CSS ships with the project.
