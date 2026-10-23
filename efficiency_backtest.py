from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from nfl_parlay.features import load_historical_games, build_team_features


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# Existing baseline features used by your original model.
BASELINE_FEATURES = [
    "win_rate_diff",
    "points_for_diff",
    "points_against_diff",
    "recent_diff_diff",
    "home_games_played",
    "away_games_played",
]

# New play-by-play efficiency metrics.
EFFICIENCY_METRICS = [
    "offensive_epa",
    "offensive_success_rate",
    "passing_epa_per_play",
    "rushing_epa_per_play",
    "defensive_epa_allowed",
    "defensive_success_allowed",
]

EFFICIENCY_FEATURES = [
    f"home_pregame_{metric}" for metric in EFFICIENCY_METRICS
] + [
    f"away_pregame_{metric}" for metric in EFFICIENCY_METRICS
] + [
    f"diff_pregame_{metric}" for metric in EFFICIENCY_METRICS
]

TARGET_SEASONS = list(range(2018, 2027))


def american_to_probability(odds):
    """Convert American odds into raw implied probabilities."""
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


def add_market_probabilities(df):
    """Remove the bookmaker's two-way margin by normalizing both sides."""
    home_ml = american_to_probability(df["home_moneyline"])
    away_ml = american_to_probability(df["away_moneyline"])

    total_ml = home_ml + away_ml
    df["market_home_win_probability"] = home_ml / total_ml

    home_spread = american_to_probability(df["home_spread_odds"])
    away_spread = american_to_probability(df["away_spread_odds"])

    total_spread = home_spread + away_spread
    df["market_home_cover_probability"] = home_spread / total_spread

    return df

def add_efficiency_features(features):
    """Join pregame efficiency metrics using each game's actual identifier."""
    efficiency_path = DATA_DIR / "team_efficiency_features.csv"

    if not efficiency_path.exists():
        raise FileNotFoundError(
            f"Missing {efficiency_path}. Run build_efficiency_data.py first."
        )

    efficiency = pd.read_csv(efficiency_path)

    # Keep the original historical game ID as the join key.
    # Each game has one row for the home team and one for the away team.
    metric_columns = [
        "pregame_offensive_epa",
        "pregame_offensive_success_rate",
        "pregame_passing_epa_per_play",
        "pregame_rushing_epa_per_play",
        "pregame_defensive_epa_allowed",
        "pregame_defensive_success_allowed",
    ]

    home = efficiency[["game_id", "team"] + metric_columns].copy()
    home = home.rename(
        columns={
            "team": "home_team",
            **{
                metric: "home_" + metric
                for metric in metric_columns
            },
        }
    )

    away = efficiency[["game_id", "team"] + metric_columns].copy()
    away = away.rename(
        columns={
            "team": "away_team",
            **{
                metric: "away_" + metric
                for metric in metric_columns
            },
        }
    )

    # Match by game ID and verify that the team identity agrees too.
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

    for metric in metric_columns:
        result["diff_" + metric] = (
            result["home_" + metric] - result["away_" + metric]
        )

    return result

def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )


def evaluate_predictions(actual, probability):
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    actual = np.asarray(actual).astype(int)

    return {
        "games": len(actual),
        "accuracy": accuracy_score(actual, probability >= 0.5),
        "brier": brier_score_loss(actual, probability),
        "log_loss": log_loss(actual, probability, labels=[0, 1]),
    }


def print_metrics(label, actual, probability):
    metrics = evaluate_predictions(actual, probability)
    print(
        f"{label:<24} "
        f"N={metrics['games']:>5}  "
        f"Accuracy={metrics['accuracy']:.3f}  "
        f"Brier={metrics['brier']:.3f}  "
        f"LogLoss={metrics['log_loss']:.3f}"
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading historical games...")
    games = load_historical_games(DATA_DIR / "historical_games.csv")

    print("Building baseline pregame features...")

    # Preserve game identifiers because build_team_features() drops game_id.
    game_keys = games[
        ["game_id", "season", "week", "home_team", "away_team"]
    ].copy()

    features = build_team_features(games)

    # Restore game_id by matching each historical game to its feature row.
    features = features.merge(
        game_keys,
        on=["season", "week", "home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )

    print("Adding efficiency features...")
    df = add_efficiency_features(features)    

    # Use only rows with completed scores.
    df["home_margin"] = df["home_score"] - df["away_score"]
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    # Exclude ties from binary moneyline evaluation.
    df["moneyline_target"] = (df["home_margin"] > 0).astype(int)
    df.loc[df["home_margin"] == 0, "moneyline_target"] = np.nan

    # nflverse convention: positive spread_line means the home team
    # was favored. Home covers when margin - spread_line is positive.

    df["cover_margin"] = df["home_margin"] - df["spread_line"]
    df["spread_target"] = (df["cover_margin"] > 0).astype(int)
    df.loc[df["cover_margin"] == 0, "spread_target"] = np.nan

    df = add_market_probabilities(df)

    # Require actual two-sided odds for fair market comparisons.
    df["moneyline_odds_available"] = (
        df["home_moneyline"].notna()
        & df["away_moneyline"].notna()
        & df["market_home_win_probability"].notna()
    )

    df["spread_odds_available"] = (
        df["home_spread_odds"].notna()
        & df["away_spread_odds"].notna()
        & df["spread_line"].notna()
        & df["market_home_cover_probability"].notna()
    )

    all_results = []

    print("\nRunning chronological, season-by-season evaluation...")
    print("Each test season is predicted using only earlier seasons.")

    seasons = sorted(df["season"].dropna().astype(int).unique())

    for season in TARGET_SEASONS:
        if season not in seasons:
            continue

        train = df[df["season"] < season].copy()
        test = df[df["season"] == season].copy()

        if train.empty or test.empty:
            continue

        print(
            f"\nSeason {season}: "
            f"{len(train)} training games, {len(test)} test games"
        )

        # ---------------- MONEYLINE ----------------
        ml_train = train.dropna(subset=["moneyline_target"])
        ml_test = test.dropna(subset=["moneyline_target"]).copy()

        if len(ml_train) and len(ml_test):
            ml_feature_sets = {
                "Baseline": BASELINE_FEATURES,
                "Efficiency": EFFICIENCY_FEATURES,
                "Combined": BASELINE_FEATURES + EFFICIENCY_FEATURES,
            }

            for model_name, columns in ml_feature_sets.items():
                model = make_model()
                model.fit(
                    ml_train[columns],
                    ml_train["moneyline_target"].astype(int),
                )

                ml_test[f"{model_name.lower()}_moneyline_probability"] = (
                    model.predict_proba(ml_test[columns])[:, 1]
                )

                metrics = evaluate_predictions(
                    ml_test["moneyline_target"],
                    ml_test[f"{model_name.lower()}_moneyline_probability"],
                )

                all_results.append({
                    "season": season,
                    "market": "moneyline",
                    "model": model_name,
                    **metrics,
                })

            if ml_test["market_home_win_probability"].notna().all():
                metrics = evaluate_predictions(
                    ml_test["moneyline_target"],
                    ml_test["market_home_win_probability"],
                )
                all_results.append({
                    "season": season,
                    "market": "moneyline",
                    "model": "Market",
                    **metrics,
                })

        # ---------------- SPREAD ----------------
        ats_train = train.dropna(
            subset=["spread_target", "spread_line"]
        ).copy()

        ats_test = test.dropna(
            subset=[
                "spread_target",
                "spread_line",
                "market_home_cover_probability",
            ]
        ).copy()

        if len(ats_train) and len(ats_test):
            ats_feature_sets = {
                "Baseline": BASELINE_FEATURES,
                "Efficiency": EFFICIENCY_FEATURES,
                "Combined": BASELINE_FEATURES + EFFICIENCY_FEATURES,
            }

            for model_name, columns in ats_feature_sets.items():
                model = make_model()
                model.fit(
                    ats_train[columns],
                    ats_train["spread_target"].astype(int),
                )

                ats_test[f"{model_name.lower()}_spread_probability"] = (
                    model.predict_proba(ats_test[columns])[:, 1]
                )

                metrics = evaluate_predictions(
                    ats_test["spread_target"],
                    ats_test[f"{model_name.lower()}_spread_probability"],
                )

                all_results.append({
                    "season": season,
                    "market": "spread",
                    "model": model_name,
                    **metrics,
                })

            if not ats_test.empty:
                metrics = evaluate_predictions(
                    ats_test["spread_target"],
                    ats_test["market_home_cover_probability"],
                )
                all_results.append({
                    "season": season,
                    "market": "spread",
                    "model": "Market",
                    **metrics,
                })

    if not all_results:
        print("No evaluation results were produced.")
        return

    results = pd.DataFrame(all_results)

    results_path = OUTPUT_DIR / "efficiency_backtest_results.csv"
    results.to_csv(results_path, index=False)

    print("\n===== AVERAGE RESULTS BY MARKET AND MODEL =====")

    summary = (
        results.groupby(["market", "model"])
        .agg(
            seasons=("season", "nunique"),
            games=("games", "sum"),
            accuracy=("accuracy", "mean"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
        .round(4)
    )

    print(summary.to_string())

    print("\n===== SEASON-BY-SEASON BRIER SCORES =====")
    print(
        results.pivot_table(
            index=["market", "season"],
            columns="model",
            values="brier",
        ).round(4).to_string()
    )

    print(f"\nDetailed results saved to: {results_path}")
    print(
        "\nReminder: these results evaluate probabilities, not betting profit. "
        "A profitable strategy requires a separate, properly timed betting "
        "backtest with realistic odds and staking."
    )


if __name__ == "__main__":
    main()