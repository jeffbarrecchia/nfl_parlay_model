from itertools import combinations
import math

import pandas as pd

from nfl_parlay.features import (
    build_latest_team_features,
    make_matchup_features,
)


def american_to_decimal(odds):
    odds = float(odds)

    if odds > 0:
        return 1 + odds / 100

    if odds < 0:
        return 1 + 100 / abs(odds)

    raise ValueError("American odds cannot be zero.")


def break_even_probability(odds):
    return 1 / american_to_decimal(odds)


def expected_roi(probability, odds):
    return probability * american_to_decimal(odds) - 1


def build_candidates(
    odds,
    features,
    moneyline_model,
    spread_model,
    feature_columns,
    min_edge=0.02,
    min_expected_roi=0.03,
):
    if odds.empty:
        return pd.DataFrame()

    # Use only the latest completed historical team statistics.
    historical = features.copy()
    historical["gameday"] = pd.to_datetime(historical["gameday"])

    team_features = build_latest_team_features(historical)

    candidates = []

    for event_id, game_rows in odds.groupby("event_id"):
        game_rows = game_rows.copy()

        # We request FanDuel only, but retain this check as a safeguard.
        game_rows = game_rows[
            game_rows["bookmaker"].astype(str).str.lower() == "fanduel"
        ]

        if game_rows.empty:
            continue

        game = game_rows.iloc[0]
        home = game["home_team"]
        away = game["away_team"]

        try:
            matchup = make_matchup_features(
                home,
                away,
                team_features,
            )
        except ValueError:
            print(f"Skipping {away} at {home}: missing historical team data.")
            continue

        X = pd.DataFrame([matchup], columns=feature_columns)

        home_win_probability = float(
            moneyline_model.predict_proba(X)[0, 1]
        )

        # The spread model estimates home-cover probability at a zero
        # spread. We do not yet have a model conditioned on the actual
        # spread, so do not use it to recommend spread bets.
        home_cover_probability = None

        # Moneyline candidates.
        for team, odds_value, probability in [
            (home, game.get("home_ml"), home_win_probability),
            (away, game.get("away_ml"), 1 - home_win_probability),
        ]:
            if pd.isna(odds_value):
                continue

            odds_value = float(odds_value)

            # Filter implausibly extreme odds until the data is validated.
            if odds_value < -2000 or odds_value > 1500 or odds_value == 0:
                continue

            implied_probability = break_even_probability(odds_value)
            edge = probability - implied_probability
            roi = expected_roi(probability, odds_value)

            if edge >= min_edge and roi >= min_expected_roi:
                candidates.append({
                    "event_id": event_id,
                    "event_key": event_id,
                    "commence_time": game.get("commence_time"),
                    "home_team": home,
                    "away_team": away,
                    "market": "moneyline",
                    "selection": team,
                    "odds": odds_value,
                    "probability": probability,
                    "break_even_probability": implied_probability,
                    "edge": edge,
                    "expected_roi": roi,
                    "bookmaker": game.get("bookmaker_title", "FanDuel"),
                })

        # Spread recommendations are deliberately disabled until we train
        # a model that predicts cover probability conditional on the actual
        # point spread. A zero-spread prediction is not sufficient.
        _ = home_cover_probability

    columns = [
        "event_id",
        "event_key",
        "commence_time",
        "home_team",
        "away_team",
        "market",
        "selection",
        "odds",
        "probability",
        "break_even_probability",
        "edge",
        "expected_roi",
        "bookmaker",
    ]

    return pd.DataFrame(candidates, columns=columns)


def select_parlay(
    candidates,
    min_parlay_probability=0.05,
    max_candidates=35,
):
    if candidates is None or candidates.empty:
        return None

    # Retain the strongest candidates first.
    candidates = candidates.sort_values(
        ["expected_roi", "probability"],
        ascending=False,
    ).head(max_candidates).copy()

    best = None

    for combo in combinations(range(len(candidates)), 3):
        legs = candidates.iloc[list(combo)]

        # Never select multiple bets from the same game.
        if legs["event_key"].duplicated().any():
            continue

        probabilities = legs["probability"].astype(float).tolist()
        parlay_probability = math.prod(probabilities)

        if parlay_probability < min_parlay_probability:
            continue

        decimal_odds = math.prod(
            american_to_decimal(x) for x in legs["odds"]
        )

        expected_return = parlay_probability * decimal_odds
        roi = expected_return - 1

        # Balance expected value with the chance of hitting the parlay.
        score = roi + 0.25 * parlay_probability

        if best is None or score > best["score"]:
            best = {
                "legs": legs,
                "probability": parlay_probability,
                "decimal_odds": decimal_odds,
                "american_odds": decimal_to_american(decimal_odds),
                "expected_roi": roi,
                "score": score,
            }

    return best


def decimal_to_american(decimal_odds):
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")

    if decimal_odds >= 2:
        return round((decimal_odds - 1) * 100)

    return round(-100 / (decimal_odds - 1))