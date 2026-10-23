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
OUTPUT_PATH = ROOT / "outputs" / "walk_forward_results.csv"


def american_to_probability(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan

    odds = float(odds)

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

    return {
        "model": name,
        "games": len(actual),
        "accuracy": accuracy_score(actual, probabilities >= 0.5),
        "brier_score": brier_score_loss(actual, probabilities),
        "log_loss": log_loss(actual, probabilities, labels=[0, 1]),
    }


def main():
    print("Loading historical NFL data...")

    games = load_historical_games(DATA_PATH)
    features = build_team_features(games)

    required = [
        "season",
        "home_margin",
        "home_moneyline",
        "away_moneyline",
    ]

    missing = [column for column in required if column not in features.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = features.copy()

    for column in ["home_moneyline", "away_moneyline"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["season", "home_margin", "home_moneyline", "away_moneyline"]
    ).copy()

    # Exclude ties and invalid moneyline prices.
    df = df[
        (df["home_margin"] != 0)
        & (df["home_moneyline"] != 0)
        & (df["away_moneyline"] != 0)
    ].copy()

    df["home_win"] = (df["home_margin"] > 0).astype(int)

    df["home_raw_prob"] = df["home_moneyline"].apply(
        american_to_probability
    )
    df["away_raw_prob"] = df["away_moneyline"].apply(
        american_to_probability
    )

    probability_total = df["home_raw_prob"] + df["away_raw_prob"]

    df["market_probability"] = df["home_raw_prob"] / probability_total

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["market_probability"])

    seasons = sorted(int(s) for s in df["season"].unique())

    # Start testing in 2018, with at least eight prior seasons for training.
    test_seasons = [season for season in seasons if season >= 2018]

    if not test_seasons:
        raise ValueError("No eligible test seasons found.")

    all_predictions = []
    all_metrics = []

    market_features = FEATURE_COLUMNS + ["market_probability"]

    for test_season in test_seasons:
        train = df[df["season"] < test_season].copy()
        test = df[df["season"] == test_season].copy()

        if len(train) < 500 or len(test) < 50:
            print(f"Skipping {test_season}: insufficient data.")
            continue

        print(
            f"Testing {test_season}: "
            f"{len(train)} training games, {len(test)} test games"
        )

        X_train = train[FEATURE_COLUMNS]
        y_train = train["home_win"].astype(int)

        # Original team-statistics model.
        team_model = make_model()
        team_model.fit(X_train, y_train)

        team_prob = team_model.predict_proba(
            test[FEATURE_COLUMNS]
        )[:, 1]

        # Market-aware model. The market probability is available for both
        # historical training games and the test season.
        market_model = make_model()
        market_model.fit(
            train[market_features],
            y_train,
        )

        market_aware_prob = market_model.predict_proba(
            test[market_features]
        )[:, 1]

        actual = test["home_win"].astype(int).to_numpy()
        market_prob = test["market_probability"].to_numpy()

        season_predictions = test[
            ["season", "gameday", "home_team", "away_team", "home_win"]
        ].copy()

        season_predictions["team_probability"] = team_prob
        season_predictions["market_aware_probability"] = market_aware_prob
        season_predictions["market_probability"] = market_prob

        all_predictions.append(season_predictions)

        for model_name, probabilities in [
            ("Team model", team_prob),
            ("Market-aware model", market_aware_prob),
            ("Market baseline", market_prob),
        ]:
            metrics = evaluate(model_name, actual, probabilities)
            metrics["season"] = test_season
            all_metrics.append(metrics)

    if not all_predictions:
        raise ValueError("No seasons could be evaluated.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    print("\n===== RESULTS BY SEASON =====")
    print(
        metrics_df.pivot(
            index="season",
            columns="model",
            values="brier_score",
        ).round(3).to_string()
    )

    print("\n===== AGGREGATE WALK-FORWARD RESULTS =====")

    for model_name in [
        "Team model",
        "Market-aware model",
        "Market baseline",
    ]:
        subset = predictions.copy()

        if model_name == "Team model":
            probability_column = "team_probability"
        elif model_name == "Market-aware model":
            probability_column = "market_aware_probability"
        else:
            probability_column = "market_probability"

        result = evaluate(
            model_name,
            subset["home_win"],
            subset[probability_column],
        )

        print(f"\n{model_name} ({result['games']} games)")
        print(f"Accuracy:   {result['accuracy']:.3f}")
        print(f"Brier:      {result['brier_score']:.3f}")
        print(f"Log loss:   {result['log_loss']:.3f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_PATH, index=False)
    metrics_df.to_csv(
        ROOT / "outputs" / "walk_forward_metrics.csv",
        index=False,
    )

    print(f"\nSaved predictions to: {OUTPUT_PATH}")
    print(f"Saved season metrics to: {ROOT / 'outputs' / 'walk_forward_metrics.csv'}")


if __name__ == "__main__":
    main()
