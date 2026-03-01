from __future__ import annotations
# Enables forward references in type hints for cleaner annotations.

from collections import deque
# Provides a fixed-size queue for recent-form tracking.
from dataclasses import dataclass, field
# Defines lightweight data containers for team state.
from typing import Dict, List, Tuple
# Provides type hints for dictionaries, lists, and tuples.

import numpy as np
# Fast numerical arrays for training the model.
import pandas as pd
# Pandas is used for clear chronological train/test splitting and test summaries.

from data_parser import Match, load_matches
# Loads and structures match data from your CSV files.


@dataclass
class TeamState:
    elo: float = 1500.0  # Elo rating initialized at a neutral baseline.
    matches: int = 0  # Number of matches already processed for this team.
    points: float = 0.0  # Cumulative points (3/1/0 system).
    goals_for: int = 0  # Total goals scored.
    goals_against: int = 0  # Total goals conceded.
    recent_points: deque = field(default_factory=lambda: deque(maxlen=5))  # Points in last 5 matches.
    recent_gf: deque = field(default_factory=lambda: deque(maxlen=5))  # Goals scored in last 5 matches.
    recent_ga: deque = field(default_factory=lambda: deque(maxlen=5))  # Goals conceded in last 5 matches.


def _result_points(home_goals: int, away_goals: int) -> Tuple[float, float]:
    # Convert a scoreline into points for home/away.
    if home_goals > away_goals:
        return 3.0, 0.0
    if home_goals < away_goals:
        return 0.0, 3.0
    return 1.0, 1.0


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _update_elo(home: TeamState, away: TeamState, home_goals: int, away_goals: int) -> None:
    # Small home advantage to reflect typical home-field edge.
    home_advantage = 50.0
    exp_home = _expected_score(home.elo + home_advantage, away.elo)
    exp_away = 1.0 - exp_home

    home_score, away_score = _result_points(home_goals, away_goals)
    home_score /= 3.0
    away_score /= 3.0

    k = 20.0
    home.elo += k * (home_score - exp_home)
    away.elo += k * (away_score - exp_away)


def _team_features(team: TeamState) -> Tuple[float, float, float, float, float]:
    # Build per-team stats used to compare teams.
    if team.matches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ppm = team.points / team.matches
    gd_per_match = (team.goals_for - team.goals_against) / team.matches
    gf_per_match = team.goals_for / team.matches
    ga_per_match = team.goals_against / team.matches

    recent_games = max(1, len(team.recent_points))
    recent_ppm = sum(team.recent_points) / recent_games
    recent_gf = sum(team.recent_gf) / recent_games
    recent_ga = sum(team.recent_ga) / recent_games
    # Recent form blends points and goal balance.
    recent_form = recent_ppm + recent_gf - recent_ga
    return ppm, gd_per_match, gf_per_match, ga_per_match, recent_form


def _build_dataset(matches: List[Match]) -> Tuple[np.ndarray, np.ndarray, Dict[str, TeamState]]:
    # Turn match history into training features/labels.
    states: Dict[str, TeamState] = {}
    X: List[List[float]] = []
    y: List[int] = []

    for match in matches:
        home = states.setdefault(match.home, TeamState())
        away = states.setdefault(match.away, TeamState())

        home_feats = _team_features(home)
        away_feats = _team_features(away)

        # Feature differences: home minus away.
        features = [
            (home.elo - away.elo) / 400.0,
            home_feats[0] - away_feats[0],
            home_feats[1] - away_feats[1],
            home_feats[2] - away_feats[2],
            home_feats[3] - away_feats[3],
            home_feats[4] - away_feats[4],
            1.0,
        ]
        X.append(features)

        if match.home_goals > match.away_goals:
            y.append(0)
        elif match.home_goals == match.away_goals:
            y.append(1)
        else:
            y.append(2)

        # Update running team stats after this match.
        home_points, away_points = _result_points(match.home_goals, match.away_goals)
        home.matches += 1
        away.matches += 1
        home.points += home_points
        away.points += away_points
        home.goals_for += match.home_goals
        home.goals_against += match.away_goals
        away.goals_for += match.away_goals
        away.goals_against += match.home_goals
        home.recent_points.append(home_points)
        away.recent_points.append(away_points)
        home.recent_gf.append(match.home_goals)
        home.recent_ga.append(match.away_goals)
        away.recent_gf.append(match.away_goals)
        away.recent_ga.append(match.home_goals)
        _update_elo(home, away, match.home_goals, match.away_goals)

    return np.array(X, dtype=float), np.array(y, dtype=int), states


def _softmax(z: np.ndarray) -> np.ndarray:
    # Turn scores into probabilities.
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _train_softmax(X: np.ndarray, y: np.ndarray, epochs: int = 2000, lr: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    # Train a 3-class logistic regression model.
    n_samples, n_features = X.shape
    n_classes = 3

    W = np.zeros((n_features, n_classes))
    b = np.zeros((1, n_classes))

    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y] = 1.0

    # Simple gradient descent for a 3-class logistic regression.
    reg = 1e-3
    for _ in range(epochs):
        scores = X @ W + b
        probs = _softmax(scores)
        grad_W = (X.T @ (probs - y_onehot)) / n_samples + reg * W
        grad_b = np.mean(probs - y_onehot, axis=0, keepdims=True)
        W -= lr * grad_W
        b -= lr * grad_b

    return W, b.reshape(-1)


@dataclass
class Model:
    weights: np.ndarray  # Trained softmax weights.
    bias: np.ndarray  # Trained softmax bias.
    mean: np.ndarray  # Feature-wise mean used for normalization.
    std: np.ndarray  # Feature-wise standard deviation used for normalization.
    states: Dict[str, TeamState]  # Final team states after training-period updates.
    evaluation_summary: pd.DataFrame  # Threshold/accuracy report computed on held-out 2025 matches.


def _split_matches_chronologically(matches: List[Match]) -> Tuple[List[Match], List[Match]]:
    # Split matches by year using pandas:
    # training years -> 2023 and 2024, test year -> 2025.
    frame = pd.DataFrame(
        {
            "idx": np.arange(len(matches), dtype=int),
            "year": [m.date.year for m in matches],
        }
    )
    train_idx = frame.loc[frame["year"].isin([2023, 2024]), "idx"].to_numpy(dtype=int)
    test_idx = frame.loc[frame["year"] == 2025, "idx"].to_numpy(dtype=int)

    train_matches = [matches[i] for i in train_idx]
    test_matches = [matches[i] for i in test_idx]
    return train_matches, test_matches


def _clone_states(states: Dict[str, TeamState]) -> Dict[str, TeamState]:
    # Deep-copy team states so evaluation can update form/Elo without mutating
    # the model state used later by the GUI.
    copied: Dict[str, TeamState] = {}
    for name, state in states.items():
        cloned = TeamState()
        cloned.elo = state.elo
        cloned.matches = state.matches
        cloned.points = state.points
        cloned.goals_for = state.goals_for
        cloned.goals_against = state.goals_against
        cloned.recent_points = deque(state.recent_points, maxlen=5)
        cloned.recent_gf = deque(state.recent_gf, maxlen=5)
        cloned.recent_ga = deque(state.recent_ga, maxlen=5)
        copied[name] = cloned
    return copied


def _evaluate_on_test_set(model: Model, test_matches: List[Match]) -> pd.DataFrame:
    # Evaluate on the 2025 test set by confidence threshold.
    # For each threshold, we keep predictions where max class probability >= threshold
    # and compute accuracy only on that filtered subset.
    eval_states = _clone_states(model.states)
    prediction_rows: List[Dict[str, float]] = []

    for match in test_matches:
        home = eval_states.setdefault(match.home, TeamState())
        away = eval_states.setdefault(match.away, TeamState())

        home_feats = _team_features(home)
        away_feats = _team_features(away)
        features = np.array(
            [
                (home.elo - away.elo) / 400.0,
                home_feats[0] - away_feats[0],
                home_feats[1] - away_feats[1],
                home_feats[2] - away_feats[2],
                home_feats[3] - away_feats[3],
                home_feats[4] - away_feats[4],
                1.0,
            ],
            dtype=float,
        )
        x = ((features - model.mean) / model.std).reshape(1, -1)
        probs = _softmax(x @ model.weights + model.bias)[0]

        predicted_class = int(np.argmax(probs))
        predicted_prob = float(probs[predicted_class])

        if match.home_goals > match.away_goals:
            actual_class = 0
        elif match.home_goals == match.away_goals:
            actual_class = 1
        else:
            actual_class = 2

        prediction_rows.append(
            {
                "predicted_probability": predicted_prob,
                "is_correct": float(predicted_class == actual_class),
            }
        )

        # Update states after each test match so later test predictions remain chronological.
        home_points, away_points = _result_points(match.home_goals, match.away_goals)
        home.matches += 1
        away.matches += 1
        home.points += home_points
        away.points += away_points
        home.goals_for += match.home_goals
        home.goals_against += match.away_goals
        away.goals_for += match.away_goals
        away.goals_against += match.home_goals
        home.recent_points.append(home_points)
        away.recent_points.append(away_points)
        home.recent_gf.append(match.home_goals)
        home.recent_ga.append(match.away_goals)
        away.recent_gf.append(match.away_goals)
        away.recent_ga.append(match.home_goals)
        _update_elo(home, away, match.home_goals, match.away_goals)

    predictions = pd.DataFrame(prediction_rows)
    thresholds = np.array([0.5, 0.6, 0.7, 0.8], dtype=float)
    table_rows: List[Dict[str, float]] = []

    # Build threshold analysis table.
    for threshold in thresholds:
        filtered = predictions[predictions["predicted_probability"] >= threshold]
        matches_evaluated = int(len(filtered))
        accuracy = float(filtered["is_correct"].mean()) if matches_evaluated > 0 else np.nan
        table_rows.append(
            {
                "threshold": float(threshold),
                "matches_evaluated": int(matches_evaluated),
                "accuracy": accuracy,
            }
        )

    summary_table = pd.DataFrame(table_rows)

    # Print a clean, readable console table.
    printable = summary_table.copy()
    printable["threshold"] = printable["threshold"].map(lambda v: f"{v:.1f}")
    printable["matches_evaluated"] = printable["matches_evaluated"].astype(int)
    printable["accuracy"] = printable["accuracy"].map(
        lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
    )
    print("\n=== Confidence Threshold Analysis (2025 Test Set) ===")
    print(printable.to_string(index=False))
    return summary_table


def train_model() -> Model:
    # Train the model and return weights plus team states.
    matches = load_matches()
    if not matches:
        raise RuntimeError("No matches found in the provided data files.")

    # Chronological split:
    # train on 2023-2024, evaluate on 2025.
    train_matches, test_matches = _split_matches_chronologically(matches)
    if not train_matches:
        raise RuntimeError("No training matches found for years 2023-2024.")
    if not test_matches:
        raise RuntimeError("No test matches found for year 2025.")

    X, y, states = _build_dataset(train_matches)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    weights, bias = _train_softmax(Xn, y)
    model = Model(
        weights=weights,
        bias=bias,
        mean=mean,
        std=std,
        states=states,
        evaluation_summary=pd.DataFrame(),
    )

    # Evaluate strictly on the held-out 2025 test set.
    model.evaluation_summary = _evaluate_on_test_set(model, test_matches)
    return model


def _feature_vector(model: Model, home: str, away: str) -> np.ndarray:
    # Create a single feature vector for a given match.
    home_state = model.states.get(home, TeamState())
    away_state = model.states.get(away, TeamState())

    home_feats = _team_features(home_state)
    away_feats = _team_features(away_state)
    features = np.array(
        [
            (home_state.elo - away_state.elo) / 400.0,
            home_feats[0] - away_feats[0],
            home_feats[1] - away_feats[1],
            home_feats[2] - away_feats[2],
            home_feats[3] - away_feats[3],
            home_feats[4] - away_feats[4],
            1.0,
        ],
        dtype=float,
    )
    return (features - model.mean) / model.std


def predict_proba(model: Model, home: str, away: str) -> Tuple[float, float, float]:
    # Predict win/draw/loss probabilities for a match.
    x = _feature_vector(model, home, away).reshape(1, -1)
    scores = x @ model.weights + model.bias
    probs = _softmax(scores)[0]
    # home win, draw, away win
    return float(probs[0]), float(probs[1]), float(probs[2])


def available_teams(model: Model) -> List[str]:
    # Return all teams present in the training data.
    return sorted(model.states.keys())
