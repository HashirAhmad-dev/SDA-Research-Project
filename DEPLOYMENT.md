# Deploying the HybridSaaS-Sec dashboard

> **Important:** the dashboard is a **Streamlit** app (a long-running Python
> WebSocket server). It is **not** deployable to Vercel's serverless runtime.
> The recommended free hosts are:
>
> 1. **Streamlit Community Cloud** -- official, free, ~5 minutes. *Recommended.*
> 2. **Render** -- free web-service tier with auto-deploy from GitHub.
> 3. **Hugging Face Spaces** -- free, public, has a native "Streamlit" SDK.
> 4. **Railway / Fly.io** -- free trials, Docker-friendly.
>
> Vercel-the-platform only fits if you wrap the app in a Docker container and
> use a third-party Vercel-compatible host (e.g. Vercel + a Docker bridge),
> or rewrite the frontend as Next.js. See the last section for details.

---

## Prerequisites

1. A **GitHub account** with the project pushed as a public (or private) repo.
2. The repo root must contain:
   * `requirements.txt`
   * `frontend/app.py`
   * `backend/` package
   * The CSV / JSON / log artefacts
   * `.streamlit/config.toml` (already shipped)

Push the repo if you haven't:

```powershell
git init
git add .
git commit -m "Initial HybridSaaS-Sec dashboard"
git branch -M main
git remote add origin https://github.com/<your-user>/hybridsaas-sec.git
git push -u origin main
```

---

## Option 1 -- Streamlit Community Cloud (recommended)

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **New app**.
3. Fill in:
   * **Repository:** `<your-user>/hybridsaas-sec`
   * **Branch:** `main`
   * **Main file path:** `frontend/app.py`
4. Open **Advanced settings**:
   * **Python version:** 3.12
   * Leave secrets empty (the app uses no API keys).
5. Click **Deploy**.

After ~2 minutes you get a public URL like
`https://hybridsaas-sec-<hash>.streamlit.app`.

**Updating:** every `git push` to `main` triggers an auto-redeploy.

---

## Option 2 -- Render.com (Vercel-like UX, supports Streamlit)

1. Sign in to **https://render.com** with GitHub.
2. Click **New + > Web Service**, pick your repo.
3. Settings:
   * **Environment:** `Python 3`
   * **Build command:** `pip install -r requirements.txt`
   * **Start command:**
     ```
     streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
   * **Instance type:** Free.
4. Click **Create Web Service**. First build takes ~5 minutes.

Render gives you `https://<service-name>.onrender.com`.

> Free Render web services sleep after 15 minutes of inactivity; the first
> request after a sleep takes ~30 seconds to cold-start.

---

## Option 3 -- Hugging Face Spaces

1. Go to **https://huggingface.co/new-space**.
2. Name it, select **Streamlit** as the SDK, visibility = Public.
3. Create the space; HF gives you a git URL.
4. Push your code:

   ```powershell
   git remote add hf https://huggingface.co/spaces/<your-user>/hybridsaas-sec
   git push hf main
   ```

5. HF expects `app.py` at the **repo root**. Add a 2-line shim:

   ```python
   # app.py (repo root)
   from frontend.app import *  # noqa: F401,F403
   ```

Public URL: `https://huggingface.co/spaces/<your-user>/hybridsaas-sec`.

---

## Option 4 -- Railway

1. Sign in at **https://railway.app** with GitHub.
2. **New Project > Deploy from GitHub repo**.
3. Railway auto-detects Python; add a `Procfile` at the repo root:

   ```
   web: streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```

4. Click **Deploy**. You get a generated `https://*.up.railway.app` URL.

---

## Option 5 -- Vercel (only if you really must)

Vercel runs serverless functions; **it cannot host a Streamlit server**.
You have two viable paths:

### 5a. Containerize and host the container *elsewhere*, link from Vercel

This is the cleanest "Vercel-shaped" option:

1. Add a `Dockerfile` at the repo root:

   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   COPY . .
   RUN pip install --no-cache-dir -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "frontend/app.py", \
        "--server.port=8501", \
        "--server.address=0.0.0.0", \
        "--server.headless=true"]
   ```

2. Push the image to **Fly.io** (`flyctl launch`), **Render**, or a tiny VPS.
3. From your Vercel-hosted marketing site / landing page, just link to the
   Streamlit URL. The "Vercel" part is the landing page; the dashboard
   lives on the Docker host.

### 5b. Rewrite the frontend as a Next.js app on Vercel + FastAPI on Render

1. Keep `backend/main.py` (FastAPI) and deploy that to **Render** as a web
   service (same as Option 2 but starting `uvicorn backend.main:app`).
2. Scaffold a new Next.js project (`npx create-next-app@latest`) on Vercel.
3. From Next.js, fetch the same endpoints (`/score/{id}`, `/comparison`, ...)
   and render the charts with Recharts / Plotly.js.

That's roughly a week of frontend work and not what your current
deliverable asks for, so I'd only recommend it if Vercel is a hard
requirement.

---

## Post-deploy smoke test

After whichever host you pick, click through:

1. **System Overview tab** -- the four metric cards render with the paper's
   numbers (42.7%, 11.2%, 0.93, 0.91).
2. **Live Simulation tab** -- the default event is in the SENSITIVE band;
   drag the sidebar `beta` slider, the gauge needle moves.
3. **Comparative Metrics tab** -- the EWMA-vs-Hybrid summary table loads
   and the time-series chart's "Reset zoom" button works.
4. **About / Documentation tab** -- the Sankey diagram and latency bar
   chart render.

If anything is missing, check the host's build log for missing dependencies
(usually a forgotten line in `requirements.txt`).
