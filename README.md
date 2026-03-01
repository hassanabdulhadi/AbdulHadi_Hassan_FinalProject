# AbdulHadi_Hassan_FinalProject

UCL Match Predictor built with Streamlit.

`streamlit_app.py` delegates to `gui.py`, so both local and hosted runs use the same full GUI.

Repository Link: https://github.com/hassanabdulhadi/AbdulHadi_Hassan_FinalProject

Project Description and Goal:

This project builds a machine learning–based football match prediction system that estimates the probability of a home win, draw, or away win using historical match data. The goal is to design a model that learns team strength, form, and performance trends over time and uses those patterns to make realistic predictions for future matches. The system combines statistical features, an Elo rating system, and a softmax regression classifier trained from scratch using NumPy.

Inputs and Outputs:

Inputs

1) CSV files containing historical match data (teams, goals scored, date, etc.)

2) Team names selected by the user in the interface

3) Historical match results from 2023–2024 for training

4) Matches from 2025 for evaluation/testing

Outputs

1) Predicted probabilities for:

2) Home win

3) Draw

4) Away win

4) Confidence-threshold evaluation table showing model accuracy at different certainty levels

5) Updated team statistics and ratings used internally for predictions



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


## Academic Integrity & AI Usage Disclosure

This project reflects my own original design decisions, system architecture, feature selection, model choice, and implementation logic. I took help and learn a lot of things from youtube in order to implement things in my code. All core components — including data processing, prediction methodology, interface design, and integration — were independently planned and developed by me while taking some help from AI.

I used AI assistance (Codex) in the following ways while completing this project:
•	I asked for help debugging my Streamlit GUI using the prompt:
“I am making a GUI using Streamlit. Can you tell where the problem is? Also, there is a song and starting picture in the file. Use these files as following: use songs in the background repeatedly and use the starting picture in the first display.”
•	I asked how to ensure my team logo image files work on other machines and not just locally:
“Will my team logos work on just my local computer or for others as well? If not, how can I make my logos work on other machines?”
•	I created the CSV dataset myself using official sources. NO AI used
•	I learned data parsing concepts from this video tutorial:
https://youtu.be/vmEHCJofslg?si=wbgQLuhHbbJUfI64 NO AI used
•	I used AI to improve my machine learning model by implementing Softmax Regression with Gradient Descent. The prompt used was:
“Write Python code that implements a 3-class softmax regression model from scratch using NumPy, including training with gradient descent and a function to predict probabilities.”


At the end, I told AI to use comments below some functions in order for me to connect things with each other. 
No AI output was copied blindly. Every implemented component was validated, modified, and integrated manually to align with project requirements and academic integrity standards.