# Deploying the CFB Picks App to Streamlit Community Cloud

Free hosting — your dad opens a URL in any browser, no Python required.

---

## What you're pushing to GitHub

```
CFB-Betting-Model/
├── app.py                          ← the web app
├── src/                            ← model code (used by app.py)
├── models/                         ← trained model files (~570 KB total)
├── data/processed/                 ← historical ratings & stats (~12 MB total)
├── requirements.txt                ← packages Streamlit Cloud will install
├── .streamlit/
│   └── secrets_template.toml      ← safe template (no real keys)
└── .gitignore                      ← keeps secrets.toml off GitHub
```

Your real API keys go in **Streamlit Cloud's secrets UI** (Step 4) — never in the repo.

---

## Step 1 — Create a GitHub repository

1. Go to [github.com](https://github.com) → **New repository**
2. Name it `CFB-Betting-Model` (or anything you like)
3. Set it to **Private** (recommended — keeps your model logic private)
4. Do **not** initialize with a README (you already have files)
5. Click **Create repository**

---

## Step 2 — Push your files to GitHub

Open Terminal in your `CFB-Betting-Model` folder and run:

```bash
cd ~/Desktop/CFB-Betting-Model

git init
git add .
git commit -m "Initial commit — CFB betting model + Streamlit app"

# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/CFB-Betting-Model.git
git branch -M main
git push -u origin main
```

> **If you already have git set up** and just need to push new changes:
> ```bash
> git add .
> git commit -m "Add Streamlit app"
> git push
> ```

---

## Step 3 — Connect to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account
2. Click **New app**
3. Fill in:
   - **Repository**: `YOUR_USERNAME/CFB-Betting-Model`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**

Streamlit will install packages from `requirements.txt` and start the app.
First deploy takes ~2 minutes.

---

## Step 4 — Add your API keys (required)

After deploying (or before clicking Deploy), click **Advanced settings** → **Secrets**.

Paste this, replacing the placeholder values with your real keys:

```toml
CFB_API_KEY  = "your-cfbd-api-key-here"
ODDS_API_KEY = "your-odds-api-key-here"
```

Click **Save** — Streamlit will reboot the app with the keys loaded.

---

## Step 5 — Share the URL with your dad

Streamlit gives you a public URL like:

```
https://your-username-cfb-betting-model-app-xxxxxx.streamlit.app
```

That's it. He opens it in any browser, picks a season + week, and hits **Get This Week's Picks**.

---

## Keeping the app up to date

Each time you retrain the model or refresh data locally:

```bash
cd ~/Desktop/CFB-Betting-Model
git add models/ data/processed/
git commit -m "Updated models for Week X"
git push
```

Streamlit Cloud auto-redeploys within ~30 seconds of a push.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| App shows "Secrets not found" | Go to app Settings → Secrets and paste your keys |
| App shows "ModuleNotFoundError" | Check requirements.txt has all needed packages |
| Old data showing | Click the ⋮ menu → **Rerun** or clear cache |
| App crashes on load | Check the Streamlit Cloud logs (click **Manage app** → **Logs**) |
