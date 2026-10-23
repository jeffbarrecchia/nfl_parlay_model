from pathlib import Path
import pandas as pd
import numpy as np

TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

REQUIRED_COLUMNS = {
    "season", "week", "gameday", "home_team", "away_team",
    "home_score", "away_score",
}


def load_historical_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Historical data missing columns: {sorted(missing)}")

    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    for col in ["home_score", "away_score", "spread_line",
                "home_moneyline", "away_moneyline",
                "home_spread_odds", "away_spread_odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[
        "gameday", "home_score", "away_score"
    ]).copy()

    return df.sort_values(["gameday", "week"]).reset_index(drop=True)


def build_team_features(games: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Build pregame features using only previously completed games."""
    rows = []
    history = {}

    for row in games.sort_values(["gameday", "week"]).itertuples(index=False):
        home, away = row.home_team, row.away_team

        def snapshot(team):
            prior = history.get(team, [])
            recent = prior[-window:]

            if not recent:
                return {
                    "win_rate": 0.5,
                    "points_for": 22.0,
                    "points_against": 22.0,
                    "point_diff": 0.0,
                    "games_played": 0,
                }

            return {
                "win_rate": float(np.mean([x["win"] for x in recent])),
                "points_for": float(np.mean([x["pf"] for x in recent])),
                "points_against": float(np.mean([x["pa"] for x in recent])),
                "point_diff": float(np.mean([x["pf"] - x["pa"] for x in recent])),
                "games_played": len(prior),
            }

        h, a = snapshot(home), snapshot(away)
        margin = float(row.home_score - row.away_score)

        item = {
            "season": int(row.season),
            "week": int(row.week),
            "gameday": row.gameday,
            "home_team": home,
            "away_team": away,
            "home_score": float(row.home_score),
            "away_score": float(row.away_score),
            "home_margin": margin,
            "home_win": int(margin > 0),
            "home_win_rate": h["win_rate"],
            "away_win_rate": a["win_rate"],
            "home_points_for": h["points_for"],
            "away_points_for": a["points_for"],
            "home_points_against": h["points_against"],
            "away_points_against": a["points_against"],
            "home_recent_diff": h["point_diff"],
            "away_recent_diff": a["point_diff"],
            "home_games_played": h["games_played"],
            "away_games_played": a["games_played"],
            "win_rate_diff": h["win_rate"] - a["win_rate"],
            "points_for_diff": h["points_for"] - a["points_for"],
            "points_against_diff": a["points_against"] - h["points_against"],
            "recent_diff_diff": h["point_diff"] - a["point_diff"],
            "is_home": 1.0,
        }

        # Moneyline outcome: ties are excluded from model training later.
        item["home_moneyline"] = getattr(row, "home_moneyline", np.nan)
        item["away_moneyline"] = getattr(row, "away_moneyline", np.nan)

        # In this dataset, spread_line is the home team's handicap:
        # positive values mean the home team is receiving points.
        spread = getattr(row, "spread_line", np.nan)
        item["spread_line"] = spread
        item["home_spread_odds"] = getattr(row, "home_spread_odds", np.nan)
        item["away_spread_odds"] = getattr(row, "away_spread_odds", np.nan)

        if pd.notna(spread):
            item["home_cover"] = int(margin + float(spread) > 0)
            item["away_cover"] = int(margin + float(spread) < 0)
        else:
            item["home_cover"] = np.nan
            item["away_cover"] = np.nan

        rows.append(item)

        # Update team histories only after creating this game's features.
        home_win = 1 if margin > 0 else 0
        away_win = 1 if margin < 0 else 0

        history.setdefault(home, []).append({
            "win": home_win, "pf": float(row.home_score),
            "pa": float(row.away_score),
        })
        history.setdefault(away, []).append({
            "win": away_win, "pf": float(row.away_score),
            "pa": float(row.home_score),
        })

    return pd.DataFrame(rows)

def build_latest_team_features(games: pd.DataFrame, window: int = 5) -> dict:
    """Calculate each team's recent performance using completed games only."""
    history = {}

    completed = games.dropna(
        subset=["home_score", "away_score", "gameday"]
    ).sort_values(["gameday", "week"])

    for row in completed.itertuples(index=False):
        margin = float(row.home_score - row.away_score)

        history.setdefault(row.home_team, []).append({
            "win": 1 if margin > 0 else 0,
            "pf": float(row.home_score),
            "pa": float(row.away_score),
        })

        history.setdefault(row.away_team, []).append({
            "win": 1 if margin < 0 else 0,
            "pf": float(row.away_score),
            "pa": float(row.home_score),
        })

    def snapshot(team):
        prior = history.get(team, [])
        recent = prior[-window:]

        if not recent:
            return {
                "win_rate": 0.5,
                "points_for": 22.0,
                "points_against": 22.0,
                "point_diff": 0.0,
                "games_played": 0,
            }

        return {
            "win_rate": float(np.mean([x["win"] for x in recent])),
            "points_for": float(np.mean([x["pf"] for x in recent])),
            "points_against": float(np.mean([x["pa"] for x in recent])),
            "point_diff": float(np.mean([x["pf"] - x["pa"] for x in recent])),
            "games_played": len(prior),
        }

    return {team: snapshot(team) for team in history}

def make_matchup_features(home_team: str, away_team: str,
                          team_features: dict) -> dict:
    """Create model input features for an upcoming matchup."""

    # Convert full sportsbook names to historical team abbreviations.
    home_team = TEAM_NAME_MAP.get(home_team, home_team)
    away_team = TEAM_NAME_MAP.get(away_team, away_team)

    h = team_features.get(home_team)
    a = team_features.get(away_team)

    if h is None or a is None:
        raise ValueError(
            f"Missing historical team data for {home_team} or {away_team}"
        )

    return {
        "win_rate_diff": h["win_rate"] - a["win_rate"],
        "points_for_diff": h["points_for"] - a["points_for"],
        "points_against_diff": a["points_against"] - h["points_against"],
        "recent_diff_diff": h["point_diff"] - a["point_diff"],
        "home_games_played": h["games_played"],
        "away_games_played": a["games_played"],
    }