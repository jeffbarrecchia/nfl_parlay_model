from pathlib import Path
import json
import pandas as pd
import nflreadpy as nfl
from datetime import datetime
from nfl_parlay.features import (
    load_historical_games,
    build_team_features,
    build_latest_team_features,
    make_matchup_features,
)
from nfl_parlay.models import train_models, save_models, load_models
from nfl_parlay.odds import load_odds
from nfl_parlay.selector import build_candidates, select_parlay

ROOT = Path(__file__).resolve().parent

def rank_upcoming_games(
    odds, games, moneyline_model, feature_columns, schedule=None
):
    """Rank upcoming games and compare model predictions with FanDuel."""
    import pandas as pd

    from nfl_parlay.selector import break_even_probability

    team_features = build_latest_team_features(games)
    rankings = []

    for event_id, game_rows in odds.groupby("event_id"):
        game_rows = game_rows[
            game_rows["bookmaker"].astype(str).str.lower() == "fanduel"
        ]

        if game_rows.empty:
            continue

        game = game_rows.iloc[0]
        home = game["home_team"]
        away = game["away_team"]

        try:
            matchup = make_matchup_features(home, away, team_features)
        except ValueError:
            print(f"Skipping {away} at {home}: missing historical team data.")
            continue

        X = pd.DataFrame([matchup], columns=feature_columns)
        home_probability = float(moneyline_model.predict_proba(X)[0, 1])
        away_probability = 1.0 - home_probability

        home_odds = game.get("home_ml")
        away_odds = game.get("away_ml")

        home_implied = (
            break_even_probability(home_odds)
            if pd.notna(home_odds) and float(home_odds) != 0
            else None
        )
        away_implied = (
            break_even_probability(away_odds)
            if pd.notna(away_odds) and float(away_odds) != 0
            else None
        )

        # Remove the sportsbook margin when both sides have valid odds.
        if home_implied is not None and away_implied is not None:
            implied_total = home_implied + away_implied
            home_no_vig = home_implied / implied_total
            away_no_vig = away_implied / implied_total
        else:
            home_no_vig = None
            away_no_vig = None

        if pd.notna(home_odds) and pd.notna(away_odds):
            favorite = (
                home if float(home_odds) < float(away_odds) else away
            )
        else:
            favorite = "Unavailable"

        model_winner = (
            home if home_probability >= away_probability else away
        )

        rankings.append({
            "event_id": event_id,
            "commence_time": game.get("commence_time"),
            "away_team": away,
            "home_team": home,
            "model_winner": model_winner,
            "model_win_probability": max(
                home_probability, away_probability
            ),
            "home_win_probability": home_probability,
            "away_win_probability": away_probability,
            "fanduel_favorite": favorite,
            "model_agrees_with_favorite": (
                model_winner == favorite if favorite != "Unavailable" else None
            ),
            "home_moneyline": home_odds,
            "away_moneyline": away_odds,
            "home_implied_probability": home_implied,
            "away_implied_probability": away_implied,
            "home_no_vig_probability": home_no_vig,
            "away_no_vig_probability": away_no_vig,
            "home_model_vs_market": (
                home_probability - home_no_vig
                if home_no_vig is not None else None
            ),
            "away_model_vs_market": (
                away_probability - away_no_vig
                if away_no_vig is not None else None
            ),
        })

    results = pd.DataFrame(rankings)

    if not results.empty:
        results["commence_time"] = pd.to_datetime(
            results["commence_time"], errors="coerce", utc=True
        )

        # Keep all upcoming games; sort by predicted winner probability.
        results = results.sort_values(
            "model_win_probability", ascending=False
        ).reset_index(drop=True)

        results.insert(0, "rank", range(1, len(results) + 1))

        # Match upcoming odds to the official NFL schedule.
        from nfl_parlay.features import TEAM_NAME_MAP

        if schedule is None:
            schedule = games.copy()
        else:
            schedule = schedule.copy()

        schedule["gameday"] = pd.to_datetime(
            schedule["gameday"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        results["game_date"] = (
            results["commence_time"].dt.strftime("%Y-%m-%d")
        )

        results["home_abbr"] = results["home_team"].map(
            lambda team: TEAM_NAME_MAP.get(team, team)
        )
        results["away_abbr"] = results["away_team"].map(
            lambda team: TEAM_NAME_MAP.get(team, team)
        )

        results["nfl_week"] = pd.NA

        for idx, result in results.iterrows():
            matching_game = schedule[
                (schedule["gameday"] == result["game_date"])
                & (
                    (
                        (schedule["home_team"] == result["home_abbr"])
                        & (schedule["away_team"] == result["away_abbr"])
                    )
                    | (
                        (schedule["home_team"] == result["away_abbr"])
                        & (schedule["away_team"] == result["home_abbr"])
                    )
                )
            ]

            if not matching_game.empty:
                # Prefer the current NFL season if multiple matches exist.
                current_season_matches = matching_game[
                    matching_game["season"] == datetime.now().year
                ]

                if not current_season_matches.empty:
                    matching_game = current_season_matches

                results.at[idx, "nfl_week"] = int(
                    matching_game.iloc[0]["week"]
                )

        results["nfl_week"] = results["nfl_week"].astype("Int64")

        results.drop(
            columns=["game_date", "home_abbr", "away_abbr"],
            inplace=True,
        )

    return results

def main():
    with open(ROOT / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    # Automatically refresh NFL data before generating parlays.
    data_path = ROOT / "data" / "historical_games.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    current_year = datetime.now().year
    seasons = list(range(2010, current_year + 1))

    print("\nRefreshing historical NFL data...")

    try:
        fresh_data = nfl.load_schedules(seasons).to_pandas()
        fresh_data.to_csv(data_path, index=False)
        print(f"Updated historical data: {len(fresh_data)} games.")
    except Exception as e:
        if data_path.exists():
            print(f"WARNING: Data refresh failed: {e}")
            print("Using previously saved historical data.")
        else:
            raise RuntimeError(
                "Could not download NFL data and no saved data exists."
            ) from e

    # Keep the complete schedule for NFL-week matching.
    raw_schedule = pd.read_csv(data_path)

    games = load_historical_games(data_path)
    features = build_team_features(games)

    model_path = ROOT / "models" / "models.joblib"

    if config.get("retrain_models", False) or not model_path.exists():
        moneyline_model, spread_model, feature_columns = train_models(
            features,
            test_seasons=config.get("test_seasons", 2),
            random_state=config.get("random_state", 42),
        )
        save_models(model_path, moneyline_model, spread_model, feature_columns)
    else:
        moneyline_model, spread_model, feature_columns = load_models(model_path)

    print("\nFetching FanDuel odds...")
    odds = load_odds(config)

    if odds.empty:
        print("No FanDuel odds returned.")
        return

    print(f"Retrieved {len(odds)} odds rows across {odds['event_id'].nunique()} games.")

    # Rank all upcoming games by predicted moneyline win probability.
    rankings = rank_upcoming_games(
        odds=odds,
        games=games,
        moneyline_model=moneyline_model,
        feature_columns=feature_columns,
        schedule=raw_schedule,
    )

    if not rankings.empty:

        print("\nAvailable NFL weeks:")
        print(
            rankings.groupby("nfl_week").size().rename("games").to_string()
        )

        week_input = input(
            "\nEnter an NFL week to display, or press Enter for all games: "
        ).strip()

        display_rankings = rankings.copy()

        if week_input:
            try:
                selected_week = int(week_input)

                display_rankings = rankings[
                    rankings["nfl_week"] == selected_week
                ].copy()

                if display_rankings.empty:
                    print(f"No games found for NFL Week {selected_week}.")
                    display_rankings = rankings.copy()

            except ValueError:
                print("Invalid week; displaying all upcoming games.")

        print("\nUpcoming Games Ranked by Model Win Probability:")
        print(
            display_rankings[
                [
                    "rank",
                    "nfl_week",
                    "commence_time",
                    "away_team",
                    "home_team",
                    "model_winner",
                    "model_win_probability",
                    "fanduel_favorite",
                    "model_agrees_with_favorite",
                    "home_no_vig_probability",
                    "away_no_vig_probability",
                    "home_model_vs_market",
                    "away_model_vs_market",
                ]
            ].to_string(
                index=False,
                formatters={
                    "model_win_probability": "{:.1%}".format,
                    "home_no_vig_probability": (
                        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                    ),
                    "away_no_vig_probability": (
                        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                    ),
                    "home_model_vs_market": (
                        lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"
                    ),
                    "away_model_vs_market": (
                        lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"
                    ),
                },
            )
        )

        output_path = ROOT / "outputs"
        output_path.mkdir(exist_ok=True)

        rankings.to_csv(
            output_path / "weekly_game_rankings.csv",
            index=False,
        )

        print(
            f"\nFull weekly rankings saved to "
            f"{output_path / 'weekly_game_rankings.csv'}"
        )
    else:
        print("\nNo upcoming games could be ranked.")

    candidates = build_candidates(
        odds=odds,
        features=features,
        moneyline_model=moneyline_model,
        spread_model=spread_model,
        feature_columns=feature_columns,
        min_edge=config.get("min_probability_edge", 0.02),
        min_expected_roi=config.get("min_expected_roi", 0.03),
    )

    if candidates.empty:
        print("No bets passed the current probability and value filters.")
        return

    candidates = candidates.sort_values("expected_roi", ascending=False)

    print("\nTop candidate bets:")
    print(candidates[
        ["commence_time", "selection", "market", "odds",
         "probability", "edge", "expected_roi"]
    ].head(15).to_string(index=False))

    parlay = select_parlay(
        candidates,
        min_parlay_probability=config.get("min_parlay_probability", 0.05),
        max_candidates=config.get("max_candidates_for_combinations", 35),
    )

    if parlay is None:
        print("\nNo qualifying three-leg parlay found.")
        return

    print("\nSuggested three-leg parlay (experimental):")
    print(parlay["legs"][
        ["selection", "market", "odds", "probability", "expected_roi"]
    ].to_string(index=False))
    print(f"\nEstimated hit probability: {parlay['probability']:.1%}")
    print(f"Combined American odds: {parlay['american_odds']:+}")
    print(f"Estimated ROI: {parlay['expected_roi']:.1%}")

    output_path = ROOT / "outputs"
    output_path.mkdir(exist_ok=True)
    candidates.to_csv(output_path / "all_candidates.csv", index=False)
    print(f"\nCandidates saved to {output_path / 'all_candidates.csv'}")


if __name__ == "__main__":
    main()