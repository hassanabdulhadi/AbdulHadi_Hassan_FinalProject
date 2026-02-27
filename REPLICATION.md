# Replication Guide (Allowed GUI Type: Web-Based)

This project includes browser-based interfaces using Streamlit in:
- `gui.py` (main GUI entry)
- `streamlit_app.py` (same app logic)

## 1) Environment
- OS: Windows 10/11, macOS, or Linux
- Python: 3.11+ (tested on Python 3.13)
- Recommended: virtual environment

## 2) Setup
From `D:\DKU\Project`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you use your existing environment, only install missing packages:

```powershell
pip install -r requirements.txt
```

## 3) Run the Web GUI

```powershell
streamlit run gui.py
```

Streamlit prints a local URL such as:
- `http://localhost:8501`

Open that in your browser.

## 3.1 Optional Portable Asset/Data URLs (for Colab/other laptops)
If local files are missing, the app now supports URL fallbacks.

Set these environment variables before running:

```powershell
$env:UCL_COMPLETE_STATS_URL="https://.../UCL_Complete_Statistics_2022-2025.csv"
$env:UCL_TEAM_RECORDS_URL="https://.../UCL_Team_By_Team_Detailed_Records.csv"
$env:UCL_SPLASH_IMAGE_URL="https://.../Starting%20pic.jpeg"
$env:UCL_MUSIC_URL="https://.../fifa18_theme.mp3"
$env:UCL_LOGO_BASE_URL="https://.../team_logos"
streamlit run gui.py
```

`UCL_LOGO_BASE_URL` should point to a folder containing files like:
- `real_madrid.png`
- `manchester_city.png`

## 4) What to test
1. Language switch in sidebar (`English` / `中文`)
2. Pick different home/away teams
3. Click `Predict`
4. Verify:
   - Win/Draw/Win percentages are shown
   - Percentages are normalized and sum to ~100.0%
   - Metrics appear for both teams

## 5) Notes
- This is an allowed GUI type under your rule `2.1` because it is browser-rendered.
- The GUI is browser-rendered (no native desktop window).
