from ml_model import available_teams, predict_proba, train_model  # Import model helpers for CLI usage.

MODEL = train_model()  # Train/load the model once at import time for faster repeated predictions.


def predict_match(home: str, away: str) -> str:  # Return a readable prediction summary for one matchup.
    if home not in MODEL.states or away not in MODEL.states:  # Validate that both teams exist in trained data.
        return "Team name not found in dataset."  # Return a clear error message when a team is unknown.

    p_home, p_draw, p_away = predict_proba(MODEL, home, away)  # Compute win/draw/win probabilities.
    return (  # Format the probabilities as a multi-line text block.
        f"{home} vs {away}\n"
        f"{home} win: {p_home * 100:.1f}%\n"
        f"Draw: {p_draw * 100:.1f}%\n"
        f"{away} win: {p_away * 100:.1f}%"
    )


def list_teams() -> None:  # Print all available team names for quick user lookup.
    for team in available_teams(MODEL):  # Iterate through sorted team list from the trained model.
        print(team)  # Output one team per line.


if __name__ == "__main__":  # Run lightweight CLI hint when script is launched directly.
    print("Model ready. Use predict_match(home, away) or list_teams().")  # Show available helper commands.
