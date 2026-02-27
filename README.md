# AbdulHadi_Hassan_FinalProject

UCL Match Predictor built with Streamlit.

`streamlit_app.py` delegates to `gui.py`, so both local and hosted runs use the same full GUI.

## 1) Setup (Works Without `.venv`)

### Windows (PowerShell)
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Codespaces / Linux / macOS
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Optional: Use a Virtual Environment

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS/Codespaces:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 2) Run the App

### Windows
```powershell
python -m streamlit run streamlit_app.py
```

### Codespaces / Linux / macOS
```bash
python3 -m streamlit run streamlit_app.py
```

Open the app at `http://localhost:8501` (or open port `8501` from the Codespaces Ports tab).

## 3) Required Assets

These files must exist in the repo:
- `Starting pic.jpeg`
- `fifa18_theme.mp3`
- `assets/team_logos/*.png`

## 4) Quick Troubleshooting

### `streamlit: command not found`
Use:
```bash
python3 -m streamlit run streamlit_app.py
```

### UI looks outdated or not loading correctly
```bash
pkill -f streamlit || true
python3 -m streamlit cache clear
python3 -m streamlit run streamlit_app.py --server.port 8501
```
Then hard refresh browser: `Ctrl+Shift+R`.

### Verify entrypoint is correct
`streamlit_app.py` should contain:
```python
from gui import main

if __name__ == "__main__":
    main()
```
