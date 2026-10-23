from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from nfl_parlay.features import load_historical_games, build_team_features
from nfl_parlay.models import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "historical_games.csv"
OUTPUT_PATH = ROOT / "outputs" / "backtest_results.csv"


def american_to_implied_probability(odds):
    """Convert American moneyline odds to raw implied probability."""
    odds = pd.to_numeric(odds, errors="coerce")

    if pd.isna(odds) or odds == 0:
        return np.nan

    if odds > 0:
        return 100 / (odds + 100)

    return abs(odds) / (abs(odds) + 100)


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )


def evaluate(name, actual, probabilities):
    probabilities = np.clip(np.asarray(probabilities), 0.001, 0.999)
    actual = np.asarray(actual).astype(int)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy:   {accuracy_score(actual, probabilities >= 0.5):.3f}")
    print(f"Brier:      {brier_score_loss(actual, probabilities):.3f}")
    print(f"Log loss:   {log_loss(actual, probabilities, labels=[0, 1]):.3f}")


def calibration_table(actual, probabilities):
    result = pd.DataFrame({
        "actual": np.asarray(actual).astype(int),
        "probability": np.asarray(probabilities),
    })

    result["probability_band"] = pd.cut(
        result["probability"],
        bins=[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        include_lowest=True,
    )

    return result.groupby(
        "probability_band", observed=False
    ).agg(
        games=("actual", "size"),
        average_predicted_probability=("probability", "mean"),
        actual_home_win_rate=("actual", "mean"),
    )


def main():
    print("Loading historical NFL data...")

    games = load_historical_games(DATA_PATH)
    features = build_team_features(games)

    required_odds = ["home_moneyline", "away_moneyline"]
    missing = [column for column in required_odds if column not in features.columns]

    if missing:
        raise ValueError(f"Historical data is missing columns: {missing}")

    df = features.copy()

    for column in required_odds:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Exclude ties and games without both moneyline prices.
    df = df.dropna(
        subset=["home_margin", "home_moneyline", "away_moneyline"]
    ).copy()
    df = df[df["home_margin"] != 0].copy()
    df = df[
        (df["home_moneyline"] != 0)
        & (df["away_moneyline"] != 0)
    ].copy()

    df["home_win"] = (df["home_margin"] > 0).astype(int)

    # Remove the bookmaker's margin by normalizing both sides.
    df["home_raw_probability"] = df["home_moneyline"].apply(
        american_to_implied_probability
    )
    df["away_raw_probability"] = df["away_moneyline"].apply(
        american_to_implied_probability
    )

    total = df["home_raw_probability"] + df["away_raw_probability"]

    df["market_home_probability"] = df["home_raw_probability"] / total

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["market_home_probability"])

    # Use all completed 2026 games as part of the test period.
    # All seasons before the first test season are training data.
    seasons = sorted(df["season"].dropna().unique())

    if len(seasons) < 3:
        raise ValueError("Not enough seasons to run a chronological backtest.")

    test_start = max(2024, int(seasons[-1]) - 2)

    train = df[df["season"] < test_start].copy()
    test = df[df["season"] >= test_start].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test data is empty.")

    print(f"Training seasons: {int(train['season'].min())}–{int(train['season'].max())}")
    print(f"Test seasons: {int(test['season'].min())}–{int(test['season'].max())}")
    print(f"Training games: {len(train)}")
    print(f"Test games: {len(test)}")
    print("Test includes only completed games with valid moneylines.")

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]
    y_train = train["home_win"]
    y_test = test["home_win"]

    # Model 1: Existing team-statistics features only.
    team_model = make_model()
    team_model.fit(X_train, y_train)

    test["team_model_probability"] = team_model.predict_proba(X_test)[:, 1]

    # Model 2: Team features plus the market's probability.
    # This is trained only on historical training seasons.
    market_features = FEATURE_COLUMNS + ["market_home_probability"]

    market_model = make_model()
    market_model.fit(train[market_features], y_train)

    test["market_model_probability"] = market_model.predict_proba(
        test[market_features]
    )[:, 1]

    # Model 3: Market baseline, using margin-adjusted implied probability.
    test["market_baseline_probability"] = test["market_home_probability"]

    print("\n===== HELD-OUT PERFORMANCE =====")

    evaluate(
        "1. Team-statistics model",
        y_test,
        test["team_model_probability"],
    )

    evaluate(
        "2. Market-aware model",
        y_test,
        test["market_model_probability"],
    )

    evaluate(
        "3. Market baseline",
        y_test,
        test["market_baseline_probability"],
    )

    print("\n===== CALIBRATION: MARKET-AWARE MODEL =====")
    print(
        calibration_table(
            y_test,
            test["market_model_probability"],
        ).to_string()
    )

    test["team_model_edge"] = (
        test["team_model_probability"] - test["market_home_probability"]
    )

    test["market_model_edge"] = (
        test["market_model_probability"] - test["market_home_probability"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()