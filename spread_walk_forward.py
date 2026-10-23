from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

HISTORICAL_PATH = DATA_DIR / "historical_games.csv"
EFFICIENCY_PATH = DATA_DIR / "team_efficiency_features.csv"

TARGET_SEASONS = list(range(2018, 2027))

BASELINE_FEATURES = [
    "win_rate_diff",
    "points_for_diff",
    "points_against_diff",
    "recent_diff_diff",
    "home_games_played",
    "away_games_played",
]

EFFICIENCY_METRICS = [
    "offensive_epa",
    "offensive_success_rate",
    "passing_epa_per_play",
    "rushing_epa_per_play",
    "defensive_epa_allowed",
    "defensive_success_allowed",
]

EFFICIENCY_FEATURES = (
    [f"home_pregame_{m}" for m in EFFICIENCY_METRICS]
    + [f"away_pregame_{m}" for m in EFFICIENCY_METRICS]
    + [f"diff_pregame_{m}" for m in EFFICIENCY_METRICS]
)

STATISTICAL_FEATURES = BASELINE_FEATURES + EFFICIENCY_FEATURES


def american_to_probability(odds):
    """Convert American odds into raw implied probability."""
    odds = pd.to_numeric(odds, errors="coerce")
    result = pd.Series(np.nan, index=odds.index, dtype=float)

    positive = odds > 0
    negative = odds < 0

    result.loc[positive] = 100 / (odds.loc[positive] + 100)
    result.loc[negative] = (
        odds.loc[negative].abs()
        / (odds.loc[negative].abs() + 100)
    )

    return result


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, random_state=42),
    )


def evaluate(actual, probability):
    actual = np.asarray(actual, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)

    return {
        "games": len(actual),
        "accuracy": accuracy_score(actual, probability >= 0.5),
        "brier": brier_score_loss(actual, probability),
        "log_loss": log_loss(actual, probability, labels=[0, 1]),
    }

def add_efficiency_features(features):
    efficiency = pd.read_csv(EFFICIENCY_PATH)

    metrics = [
        "pregame_offensive_epa",
        "pregame_offensive_success_rate",
        "pregame_passing_epa_per_play",
        "pregame_rushing_epa_per_play",
        "pregame_defensive_epa_allowed",
        "pregame_defensive_success_allowed",
    ]

    home = efficiency[["game_id", "team"] + metrics].rename(
        columns={
            "team": "home_team",
            **{m: f"home_{m}" for m in metrics},
        }
    )

    away = efficiency[["game_id", "team"] + metrics].rename(
        columns={
            "team": "away_team",
            **{m: f"away_{m}" for m in metrics},
        }
    )

    result = features.merge(
        home,
        on=["game_id", "home_team"],
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        away,
        on=["game_id", "away_team"],
        how="left",
        validate="many_to_one",
    )

    for metric in metrics:
        result[f"diff_{metric}"] = (
            result[f"home_{metric}"] - result[f"away_{metric}"]
        )

    return result

def main():
    if not HISTORICAL_PATH.exists():
        raise FileNotFoundError(f"Missing {HISTORICAL_PATH}")

    if not EFFICIENCY_PATH.exists():
        raise FileNotFoundError(f"Missing {EFFICIENCY_PATH}")

    print("Loading historical games...")

    games = pd.read_csv(HISTORICAL_PATH)

    required_game_columns = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
    ]

    missing = [c for c in required_game_columns if c not in games.columns]
    if missing:
        raise ValueError(f"Historical data is missing columns: {missing}")

    games["season"] = pd.to_numeric(games["season"], errors="coerce")

    for column in [
        "home_score",
        "away_score",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
    ]:
        games[column] = pd.to_numeric(games[column], errors="coerce")

    # Exclude games without final scores or a usable spread line.
    games = games.dropna(
        subset=["home_score", "away_score", "spread_line"]
    ).copy()

    # Restore the project's existing historical team-feature logic.
    from nfl_parlay.features import build_team_features

    features = build_team_features(games)

    # Restore the actual game ID because build_team_features may omit it.
    game_keys = games[
        ["game_id", "season", "week", "home_team", "away_team"]
    ].drop_duplicates(
        subset=["season", "week", "home_team", "away_team"]
    )

    features = features.merge(
        game_keys,
        on=["season", "week", "home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )

    if features["game_id"].isna().any():
        missing_ids = int(features["game_id"].isna().sum())
        raise ValueError(
            f"Could not match {missing_ids} feature rows to game IDs."
        )

    features = add_efficiency_features(features)

    # Add historical spread lines and prices once, using game ID.
    game_data = games[
        [
            "game_id",
            "home_score",
            "away_score",
            "spread_line",
            "home_spread_odds",
            "away_spread_odds",
        ]
    ].drop_duplicates(subset=["game_id"])

    df = features.merge(
        game_data,
        on="game_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_schedule"),
    )

    # If the feature builder retained scores or lines, use the schedule
    # version to avoid ambiguous duplicate columns.
    for column in [
        "home_score",
        "away_score",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
    ]:
        schedule_column = f"{column}_schedule"
        if schedule_column in df.columns:
            df[column] = df[schedule_column]
            df.drop(columns=[schedule_column], inplace=True)

    df["home_margin"] = df["home_score"] - df["away_score"]

    # Positive spread_line means home team is favored.
    df["cover_margin"] = df["home_margin"] - df["spread_line"]

    # Pushes are excluded from binary cover prediction.
    df["spread_target"] = (df["cover_margin"] > 0).astype(float)
    df.loc[df["cover_margin"] == 0, "spread_target"] = np.nan

    # Raw probabilities from both sides, normalized to remove the vig.
    home_raw = american_to_probability(df["home_spread_odds"])
    away_raw = american_to_probability(df["away_spread_odds"])
    total_raw = home_raw + away_raw

    df["market_home_cover_probability"] = home_raw / total_raw

    missing_features = [
        c for c in STATISTICAL_FEATURES if c not in df.columns
    ]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    predictions = []
    evaluation_rows = []

    for season in TARGET_SEASONS:
        train = df[df["season"] < season].copy()
        test = df[df["season"] == season].copy()

        # Evaluate only games with final outcomes, a spread line,
        # and both spread prices.
        test = test.dropna(
            subset=[
                "spread_target",
                "spread_line",
                "home_spread_odds",
                "away_spread_odds",
                "market_home_cover_probability",
            ]
        ).copy()

        train = train.dropna(
            subset=["spread_target", "spread_line"]
        ).copy()

        if train.empty or test.empty:
            print(f"Skipping {season}: insufficient training or test data.")
            continue

        print(
            f"Season {season}: {len(train)} training games, "
            f"{len(test)} test games"
        )

        model = make_model()
        model.fit(
            train[STATISTICAL_FEATURES],
            train["spread_target"].astype(int),
        )

        test["team_probability"] = model.predict_proba(
            test[STATISTICAL_FEATURES]
        )[:, 1]

        test["market_probability"] = test["market_home_cover_probability"]

        # Save genuinely out-of-sample predictions and the prices needed
        # for the subsequent betting ROI analysis.
        keep_columns = [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
            "home_spread_odds",
            "away_spread_odds",
            "cover_margin",
            "spread_target",
            "team_probability",
            "market_probability",
        ]

        predictions.append(test[keep_columns].copy())

        market_metrics = evaluate(
            test["spread_target"].astype(int),
            test["market_probability"],
        )

        team_metrics = evaluate(
            test["spread_target"].astype(int),
            test["team_probability"],
        )

        evaluation_rows.append({
            "season": season,
            "model": "Market",
            **market_metrics,
        })

        evaluation_rows.append({
            "season": season,
            "model": "Statistical",
            **team_metrics,
        })

    if not predictions:
        print("No out-of-sample predictions were produced.")
        return

    predictions_df = pd.concat(predictions, ignore_index=True)
    evaluation_df = pd.DataFrame(evaluation_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "spread_walk_forward_results.csv"
    evaluation_path = OUTPUT_DIR / "spread_walk_forward_metrics.csv"

    predictions_df.to_csv(predictions_path, index=False)
    evaluation_df.to_csv(evaluation_path, index=False)

    print("\n===== OVERALL SPREAD PREDICTION RESULTS =====")

    overall = (
        evaluation_df.groupby("model")
        .agg(
            seasons=("season", "nunique"),
            games=("games", "sum"),
            accuracy=("accuracy", "mean"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
        .round(4)
    )

    print(overall.to_string())

    print("\n===== SEASON-BY-SEASON BRIER SCORES =====")
    print(
        evaluation_df.pivot(
            index="season",
            columns="model",
            values="brier",
        ).round(4).to_string()
    )

    print(f"\nSaved predictions: {predictions_path}")
    print(f"Saved metrics: {evaluation_path}")


if __name__ == "__main__":
    main()