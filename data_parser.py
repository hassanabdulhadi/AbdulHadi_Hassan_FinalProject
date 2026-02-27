from __future__ import annotations
# Enables forward references in type hints for cleaner annotations.

import csv
# Reads CSV-style rows from the stats files.
import datetime as dt
# Handles parsing and storing dates.
import os
# Reads optional environment-variable URLs for remote dataset fallback.
from urllib.request import urlretrieve
# Downloads remote dataset files when local files are missing.
from dataclasses import dataclass
# Defines lightweight data containers for matches.
from pathlib import Path
# Builds OS-safe file paths relative to this script.
from typing import List, Optional
# Provides type hints for lists and optional values.


@dataclass(frozen=True)
class Match:
    date: dt.date  # Calendar date of the match.
    home: str  # Home team name.
    away: str  # Away team name.
    home_goals: int  # Goals scored by the home team.
    away_goals: int  # Goals scored by the away team.
    source: str  # Source label indicating which CSV this row came from.


ROOT = Path(__file__).resolve().parent  # Project folder used to resolve CSV paths.
DATA_CACHE_DIR = ROOT / ".asset_cache"  # Local cache folder for downloaded data files.

COMPLETE_STATS_FILE = "UCL_Complete_Statistics_2022-2025.csv"
TEAM_RECORDS_FILE = "UCL_Team_By_Team_Detailed_Records.csv"

COMPLETE_STATS_URL = os.getenv("UCL_COMPLETE_STATS_URL", "").strip()
TEAM_RECORDS_URL = os.getenv("UCL_TEAM_RECORDS_URL", "").strip()


def _parse_date(value: str) -> Optional[dt.date]:
    # Convert a YYYY-MM-DD string into a date object.
    value = value.strip()
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _parse_score(value: str) -> Optional[tuple[int, int]]:
    # Turn a "2-1" score string into integer goals.
    value = value.strip().lstrip("'")
    if "-" not in value:
        return None
    left, right = value.split("-", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def _clean_team(name: str) -> str:
    # Normalize team names by collapsing extra spaces.
    return " ".join(name.strip().split())


def _resolve_data_file(filename: str, remote_url: str) -> Path:
    # Resolve a dataset path with local-first and URL fallback behavior.
    direct_local = ROOT / filename
    if direct_local.exists():
        return direct_local

    if not remote_url:
        return direct_local

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = DATA_CACHE_DIR / filename
    if cached.exists():
        return cached

    try:
        urlretrieve(remote_url, cached)
    except Exception:
        # If download fails, return the non-existing direct path so caller can skip gracefully.
        return direct_local
    return cached


def _load_complete_statistics(path: Path) -> List[Match]:
    # Parse match results from the comprehensive statistics file.
    matches: List[Match] = []
    headers = {
        ("Matchday", "Date", "Home Team", "Away Team", "Score", "Winner"): "md",
        ("Group", "Matchday", "Date", "Home Team", "Away Team", "Score", "Winner"): "group",
        ("Round", "Date", "Home Team", "Away Team", "Score", "Aggregate", "Winner"): "round",
    }

    current_mode: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                current_mode = None
                continue

            first = row[0].strip()
            if first.startswith("SECTION") or first.startswith("UEFA"):
                current_mode = None
                continue

            header_key = tuple(cell.strip() for cell in row)
            if header_key in headers:
                # We just entered a table with match results.
                current_mode = headers[header_key]
                continue

            if first in {"Rank", "TEAM:", "Summary"}:
                current_mode = None
                continue

            if current_mode is None:
                continue

            if current_mode == "md":
                # Matchday,Date,Home Team,Away Team,Score,Winner
                if len(row) < 6:
                    continue
                date = _parse_date(row[1])
                score = _parse_score(row[4])
                if date is None or score is None:
                    continue
                home, away = _clean_team(row[2]), _clean_team(row[3])
                matches.append(
                    Match(date, home, away, score[0], score[1], "complete")
                )
            elif current_mode == "group":
                # Group,Matchday,Date,Home Team,Away Team,Score,Winner
                if len(row) < 7:
                    continue
                date = _parse_date(row[2])
                score = _parse_score(row[5])
                if date is None or score is None:
                    continue
                home, away = _clean_team(row[3]), _clean_team(row[4])
                matches.append(
                    Match(date, home, away, score[0], score[1], "complete")
                )
            elif current_mode == "round":
                # Round,Date,Home Team,Away Team,Score,Aggregate,Winner
                if len(row) < 6:
                    continue
                date = _parse_date(row[1])
                score = _parse_score(row[4])
                if date is None or score is None:
                    continue
                home, away = _clean_team(row[2]), _clean_team(row[3])
                matches.append(
                    Match(date, home, away, score[0], score[1], "complete")
                )

    return matches


def _load_team_records(path: Path) -> List[Match]:
    # Parse match results from the team-by-team records file.
    matches: List[Match] = []
    team: Optional[str] = None
    in_table = False

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                in_table = False
                continue

            first = row[0].strip()
            if first.startswith("TEAM:"):
                team = _clean_team(first.replace("TEAM:", "", 1))
                in_table = False
                continue

            if first.startswith("Season") and len(row) >= 7:
                # Start of a team's match list.
                in_table = True
                continue

            if first.startswith("Summary"):
                in_table = False
                continue

            if not in_table or team is None:
                continue

            if len(row) < 7:
                continue

            date = _parse_date(row[1])
            if date is None:
                continue
            venue = row[2].strip().lower()
            opponent = _clean_team(row[3])
            score = _parse_score(row[4])
            if score is None:
                continue

            if venue == "home":
                home, away = team, opponent
                home_goals, away_goals = score
            elif venue == "away":
                home, away = opponent, team
                # In this file the score is already written as home-away.
                home_goals, away_goals = score
            else:
                # Treat other venues as neutral, keep team as home for consistency.
                home, away = team, opponent
                home_goals, away_goals = score

            matches.append(
                Match(date, home, away, home_goals, away_goals, "team_records")
            )

    return matches


def load_matches() -> List[Match]:
    # Load, combine, and de-duplicate all match results.
    complete_path = _resolve_data_file(COMPLETE_STATS_FILE, COMPLETE_STATS_URL)
    team_records_path = _resolve_data_file(TEAM_RECORDS_FILE, TEAM_RECORDS_URL)

    matches: List[Match] = []
    if complete_path.exists():
        # Full competition results tables.
        matches.extend(_load_complete_statistics(complete_path))
    if team_records_path.exists():
        # Team-by-team logs. These can overlap with the full tables.
        matches.extend(_load_team_records(team_records_path))

    # De-duplicate by date/home/away/score
    unique = {}
    for match in matches:
        key = (
            match.date,
            match.home.lower(),
            match.away.lower(),
            match.home_goals,
            match.away_goals,
        )
        if key not in unique:
            unique[key] = match

    ordered = sorted(unique.values(), key=lambda m: (m.date, m.home, m.away))

    # Canonicalize team-name casing (e.g., "REAL MADRID" -> "Real Madrid")
    # by picking the variant with more lowercase characters for each lowercase key.
    preferred_name: dict[str, str] = {}
    for match in ordered:
        for name in (match.home, match.away):
            key = name.lower().strip()
            prev = preferred_name.get(key)
            if prev is None:
                preferred_name[key] = name
            elif sum(c.islower() for c in name) > sum(c.islower() for c in prev):
                preferred_name[key] = name

    normalized: List[Match] = []
    for match in ordered:
        home = preferred_name.get(match.home.lower().strip(), match.home)
        away = preferred_name.get(match.away.lower().strip(), match.away)
        normalized.append(
            Match(
                date=match.date,
                home=home,
                away=away,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                source=match.source,
            )
        )
    return normalized
