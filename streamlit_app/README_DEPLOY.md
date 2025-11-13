# Deploy QUISK — Streamlit Demo

## Local

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

**Note:** If you encounter OpenCV errors (`libGL.so.1`), the app will still work - placeholder video generation will be skipped. Real videos will display normally.

## Streamlit Community Cloud (easiest)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and choose New app.
3. Select your repo & branch.
4. Main file path: `streamlit_app/app.py`
5. Advanced: leave defaults. Deploy.

## Render (or Railway/Heroku)

**Render:**

1. New Web Service → build from repo.
2. Runtime: Python 3.
3. Start command (auto from Procfile):
   ```
   sh setup.sh && streamlit run streamlit_app/app.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Set Build Command: `pip install -r requirements.txt`
5. Set environment to keep free tier alive optionally.

