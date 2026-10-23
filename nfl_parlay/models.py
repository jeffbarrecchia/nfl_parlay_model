from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)


FEATURE_COLUMNS = [
    "win_rate_diff",
    "points_for_diff",
    "points_against_diff",
    "recent_diff_diff",
    "home_games_played",
    "away_games_played",
]


def prepare_training_data(features: pd.DataFrame) -> pd.DataFrame:
    """Prepare moneyline and spread labels from historical games."""
    df = features.copy()

    # Convert required numeric fields safely.
    numeric_columns = FEATURE_COLUMNS + [
        "home_win",
        "home_margin",
        "spread_line",
        "home_moneyline",
        "away_moneyline",
        "home_spread_odds",
        "away_spread_odds",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # A home win is a win, not a tie.
    # Tied games are excluded from binary moneyline training.
    df["moneyline_target"] = (df["home_margin"] > 0).astype(float)
    df.loc[df["home_margin"] == 0, "moneyline_target"] = float("nan")

    # nflverse convention:
    # spread_line: positive means home team is favored.
    # Home covers when home_margin - spread_line > 0.

    cover_margin = df["home_margin"] - df["spread_line"]

    df["spread_target"] = (cover_margin > 0).astype(float)
    df.loc[cover_margin == 0, "spread_target"] = float("nan")

    # Exclude games without the relevant odds from betting evaluation.
    df["moneyline_odds_available"] = (
        df["home_moneyline"].notna() & df["away_moneyline"].notna()
    )
    df["spread_odds_available"] = (
        df["home_spread_odds"].notna()
        & df["away_spread_odds"].notna()
        & df["spread_line"].notna()
    )

    return df


def _make_model(random_state: int = 42):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=random_state),
    )


def train_models(
    features: pd.DataFrame,
    test_seasons: int = 2,
    random_state: int = 42,
):
    """Train on earlier seasons and evaluate on the latest seasons."""
    df = prepare_training_data(features)
    seasons = sorted(df["season"].dropna().unique())

    if len(seasons) <= test_seasons:
        raise ValueError("Not enough seasons for a chronological split.")

    cutoff_season = seasons[-test_seasons]

    train = df[df["season"] < cutoff_season].copy()
    test = df[df["season"] >= cutoff_season].copy()

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    # Moneyline model: probability that the home team wins.
    moneyline_train = train.dropna(subset=["moneyline_target"])
    moneyline_test = test.dropna(subset=["moneyline_target"])

    moneyline_model = _make_model(random_state)
    moneyline_model.fit(
        moneyline_train[FEATURE_COLUMNS],
        moneyline_train["moneyline_target"].astype(int),
    )

    # Spread model: probability that the home team covers.
    spread_train = train.dropna(subset=["spread_target"])
    spread_test = test.dropna(subset=["spread_target"])

    spread_model = _make_model(random_state)
    spread_model.fit(
        spread_train[FEATURE_COLUMNS],
        spread_train["spread_target"].astype(int),
    )

    # Report held-out results only; do not report training performance as
    # evidence of predictive ability.
    print(f"Training seasons: {min(seasons)}–{cutoff_season - 1}")
    print(f"Test seasons: {cutoff_season}–{max(seasons)}")
    print(f"Moneyline training games: {len(moneyline_train)}")
    print(f"Spread training games: {len(spread_train)}")

    if len(moneyline_test):
        ml_prob = moneyline_model.predict_proba(
            moneyline_test[FEATURE_COLUMNS]
        )[:, 1]
        ml_actual = moneyline_test["moneyline_target"].astype(int)

        print("\nMoneyline holdout results (home win):")
        print(f"  Accuracy: {accuracy_score(ml_actual, ml_prob >= 0.5):.3f}")
        print(f"  Brier score: {brier_score_loss(ml_actual, ml_prob):.3f}")
        print(f"  Log loss: {log_loss(ml_actual, ml_prob, labels=[0, 1]):.3f}")

    if len(spread_test):
        ats_prob = spread_model.predict_proba(
            spread_test[FEATURE_COLUMNS]
        )[:, 1]
        ats_actual = spread_test["spread_target"].astype(int)

        print("\nSpread holdout results (home covers):")
        print(f"  Accuracy: {accuracy_score(ats_actual, ats_prob >= 0.5):.3f}")
        print(f"  Brier score: {brier_score_loss(ats_actual, ats_prob):.3f}")
        print(f"  Log loss: {log_loss(ats_actual, ats_prob, labels=[0, 1]):.3f}")

    return moneyline_model, spread_model, FEATURE_COLUMNS


def save_models(path: Path, moneyline_model, spread_model, feature_columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "moneyline_model": moneyline_model,
            "spread_model": spread_model,
            "feature_columns": feature_columns,
        },
        path,
    )


def load_models(path: Path):
    payload = joblib.load(path)
    return (
        payload["moneyline_model"],
        payload["spread_model"],
        payload["feature_columns"],
    )