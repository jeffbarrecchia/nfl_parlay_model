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
    """Convert American odds into implied probabilities."""
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
    """Calculate no-vig market probabilities for both markets."""
    df = df.copy()

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
    """Join each team's pregame efficiency metrics using game ID and team."""
    efficiency = pd.read_csv(EFFICIENCY_PATH)

    metric_columns = [
        "pregame_" + metric for metric in EFFICIENCY_METRICS
    ]

    home = efficiency[["game_id", "team"] + metric_columns].copy()
    home = home.rename(
        columns={
            "team": "home_team",
            **{m: "home_" + m for m in metric_columns},
        }
    )

    away = efficiency[["game_id", "team"] + metric_columns].copy()
    away = away.rename(
        columns={
            "team": "away_team",
            **{m: "away_" + m for m in metric_columns},
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

    for metric in metric_columns:
        result["diff_" + metric] = (
            result["home_" + metric] - result["away_" + metric]
        )

    return result


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, random_state=42),
    )


def evaluate_predictions(actual, probability):
    actual = np.asarray(actual, dtype=int)
    probability = np.asarray(probability, dtype=float)
    probability = np.clip(probability, 1e-6, 1 - 1e-6)

    return {
        "games": len(actual),
        "accuracy": accuracy_score(actual, probability >= 0.5),
        "brier": brier_score_loss(actual, probability),
        "log_loss": log_loss(actual, probability, labels=[0, 1]),
    }


def run_backtest():
    if not HISTORICAL_PATH.exists():
        raise FileNotFoundError(f"Missing historical data: {HISTORICAL_PATH}")

    if not EFFICIENCY_PATH.exists():
        raise FileNotFoundError(f"Missing efficiency data: {EFFICIENCY_PATH}")

    print("Loading historical games and team features...")

    games = pd.read_csv(HISTORICAL_PATH)

    # Rebuild the same historical team features used by the project.
    from nfl_parlay.features import build_team_features

    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")

    # Only games with final scores can be used as labeled outcomes.
    games = games.dropna(subset=["home_score", "away_score"]).copy()

    features = build_team_features(games)

    # Restore game identifiers lost during feature engineering.
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

    features = add_efficiency_features(features)

    # Join historical odds by game ID so odds cannot be misaligned by row order.

    df = features.copy()  

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["home_margin"] = df["home_score"] - df["away_score"]

    # Moneyline outcomes: exclude tied games.
    df["moneyline_target"] = (df["home_margin"] > 0).astype(float)
    df.loc[df["home_margin"] == 0, "moneyline_target"] = np.nan

    # nflverse convention: positive spread_line means the home team is favored.
    df["cover_margin"] = df["home_margin"] - df["spread_line"]
    df["spread_target"] = (df["cover_margin"] > 0).astype(float)
    df.loc[df["cover_margin"] == 0, "spread_target"] = np.nan

    df = add_market_probabilities(df)

    missing_features = [
        column for column in STATISTICAL_FEATURES if column not in df.columns
    ]
    if missing_features:
        raise ValueError(
            "Missing model features: " + ", ".join(missing_features)
        )

    all_results = []

    for season in TARGET_SEASONS:
        train = df[df["season"] < season].copy()
        test = df[df["season"] == season].copy()

        if train.empty or test.empty:
            continue

        print(
            f"\nSeason {season}: "
            f"{len(train)} training games, {len(test)} test games"
        )

        for market, target_column, market_column in [
            (
                "moneyline",
                "moneyline_target",
                "market_home_win_probability",
            ),
            (
                "spread",
                "spread_target",
                "market_home_cover_probability",
            ),
        ]:
            # Compare all approaches on exactly the same test games.
            test_rows = test.dropna(
                subset=[target_column, market_column]
            ).copy()

            train_rows = train.dropna(subset=[target_column]).copy()

            if train_rows.empty or test_rows.empty:
                print(f"  Skipping {market}: insufficient labeled data.")
                continue

            actual = test_rows[target_column].astype(int)
            market_probability = test_rows[market_column].astype(float)

            # Market-only benchmark.
            metrics = evaluate_predictions(actual, market_probability)
            all_results.append({
                "season": season,
                "market": market,
                "model": "Market",
                **metrics,
            })

            # Statistical-only model.
            statistical_model = make_model()
            statistical_model.fit(
                train_rows[STATISTICAL_FEATURES],
                train_rows[target_column].astype(int),
            )

            statistical_probability = statistical_model.predict_proba(
                test_rows[STATISTICAL_FEATURES]
            )[:, 1]

            metrics = evaluate_predictions(actual, statistical_probability)
            all_results.append({
                "season": season,
                "market": market,
                "model": "Statistical",
                **metrics,
            })

            # Market-aware model: add market probability as a model feature.
            market_feature = "market_probability_input"

            train_market = train_rows.dropna(subset=[market_column]).copy()
            test_market = test_rows.copy()

            train_market[market_feature] = train_market[market_column]
            test_market[market_feature] = test_market[market_column]

            market_aware_features = STATISTICAL_FEATURES + [market_feature]

            market_model = make_model()
            market_model.fit(
                train_market[market_aware_features],
                train_market[target_column].astype(int),
            )

            market_aware_probability = market_model.predict_proba(
                test_market[market_aware_features]
            )[:, 1]

            metrics = evaluate_predictions(actual, market_aware_probability)
            all_results.append({
                "season": season,
                "market": market,
                "model": "Market-aware",
                **metrics,
            })

    if not all_results:
        print("No results were produced.")
        return

    results = pd.DataFrame(all_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "market_aware_backtest_results.csv"
    results.to_csv(output_path, index=False)

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

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_backtest()