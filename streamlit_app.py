"""
GitHub-hosted Streamlit environments often launch `streamlit_app.py` by default.
Delegate to `gui.py` so both entrypoints render the same full asset-enabled app.
"""

from gui import main


if __name__ == "__main__":
    main()
