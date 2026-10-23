from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "outputs" / "spread_walk_forward_results.csv"
OUTPUT_DIR = ROOT / "outputs"

EDGE_THRESHOLDS = [0.02, 0.03, 0.05, 0.07, 0.10]
STAKE = 10.00


def american_profit_per_dollar(odds):
    """Net profit on a winning bet per dollar staked."""
    if odds > 0:
        return odds / 100.0
    if odds < 0:
        return 100.0 / abs(odds)
    return np.nan


def american_break_even_probability(odds):
    """Break-even probability at the offered American odds."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return np.nan


def calculate_drawdown(profits):
    """Maximum peak-to-trough drawdown in cumulative profit."""
    if len(profits) == 0:
        return 0.0

    cumulative = np.cumsum(profits)
    running_peak = np.maximum.accumulate(
        np.insert(cumulative, 0, 0.0)
    )[1:]

    return float((running_peak - cumulative).max())


def prepare_bets(df, threshold):
    """
    Select at most one spread side per game.

    The model's edge is measured against the no-vig market probability.
    Expected value is calculated separately using the actual offered odds.
    """
    rows = []

    for _, game in df.iterrows():
        p_home = game["team_probability"]
        p_market_home = game["market_probability"]

        if pd.isna(p_home) or pd.isna(p_market_home):
            continue

        p_away = 1.0 - p_home
        p_market_away = 1.0 - p_market_home

        home_edge = p_home - p_market_home
        away_edge = p_away - p_market_away

        if home_edge >= threshold and home_edge >= away_edge:
            side = "home"
            probability = p_home
            market_probability = p_market_home
            odds = game["home_spread_odds"]
            outcome = int(game["spread_target"])

        elif away_edge >= threshold:
            side = "away"
            probability = p_away
            market_probability = p_market_away
            odds = game["away_spread_odds"]
            outcome = 1 - int(game["spread_target"])

        else:
            continue

        if pd.isna(odds) or odds == 0:
            continue

        odds = float(odds)

        profit_if_win = STAKE * american_profit_per_dollar(odds)
        break_even_probability = american_break_even_probability(odds)

        # Expected profit if the model's probability estimate is correct.
        expected_profit = (
            probability * profit_if_win
            - (1.0 - probability) * STAKE
        )

        # Actual result from the historical game.
        if outcome == 1:
            actual_profit = profit_if_win
            result = "win"
        else:
            actual_profit = -STAKE
            result = "loss"

        rows.append({
            "edge_threshold": threshold,
            "season": int(game["season"]),
            "game_id": game["game_id"],
            "gameday": game["gameday"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "bet_side": side,
            "spread_line": game["spread_line"],
            "american_odds": odds,
            "model_probability": probability,
            "no_vig_market_probability": market_probability,
            "break_even_probability": break_even_probability,
            "estimated_edge": probability - market_probability,
            "expected_profit": expected_profit,
            "actual_result": result,
            "actual_profit": actual_profit,
            "stake": STAKE,
        })

    return pd.DataFrame(rows)


def summarize_bets(bets, group_columns):
    """Summarize expected and realized betting performance."""
    if bets.empty:
        return pd.DataFrame()

    grouped = bets.groupby(group_columns, dropna=False)

    summary = grouped.agg(
        bets=("actual_result", "size"),
        wins=("actual_result", lambda x: (x == "win").sum()),
        expected_profit=("expected_profit", "sum"),
        actual_profit=("actual_profit", "sum"),
        total_staked=("stake", "sum"),
    ).reset_index()

    summary["win_rate"] = summary["wins"] / summary["bets"]
    summary["expected_roi"] = (
        summary["expected_profit"] / summary["total_staked"]
    )
    summary["actual_roi"] = (
        summary["actual_profit"] / summary["total_staked"]
    )

    return summary


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run spread_walk_forward.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "game_id",
        "season",
        "gameday",
        "home_team",
        "away_team",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
        "spread_target",
        "team_probability",
        "market_probability",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = [
        "season",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
        "spread_target",
        "team_probability",
        "market_probability",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "season",
            "spread_line",
            "home_spread_odds",
            "away_spread_odds",
            "spread_target",
            "team_probability",
            "market_probability",
        ]
    ).copy()

    # The walk-forward script excludes pushes. Validate the labels here.
    df = df[df["spread_target"].isin([0, 1])].copy()

    print(f"Loaded {len(df)} games with usable predictions and odds.")
    print(f"Flat stake per bet: ${STAKE:.2f}")

    all_bets = []

    for threshold in EDGE_THRESHOLDS:
        bets = prepare_bets(df, threshold)

        if not bets.empty:
            all_bets.append(bets)

    if not all_bets:
        print("No bets qualified at any threshold.")
        return

    bets_df = pd.concat(all_bets, ignore_index=True)

    overall = summarize_bets(bets_df, ["edge_threshold"])
    by_season = summarize_bets(
        bets_df,
        ["edge_threshold", "season"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bets_path = OUTPUT_DIR / "spread_ev_bets.csv"
    overall_path = OUTPUT_DIR / "spread_ev_summary.csv"
    season_path = OUTPUT_DIR / "spread_ev_by_season.csv"

    bets_df.to_csv(bets_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_season.to_csv(season_path, index=False)

    display_columns = [
        "edge_threshold",
        "bets",
        "wins",
        "win_rate",
        "expected_profit",
        "actual_profit",
        "expected_roi",
        "actual_roi",
        "total_staked",
    ]

    print("\n===== OVERALL EXPECTED VALUE VS ACTUAL RESULTS =====")
    print(overall[display_columns].round(4).to_string(index=False))

    print("\n===== RESULTS BY SEASON =====")
    print(
        by_season[
            [
                "edge_threshold",
                "season",
                "bets",
                "win_rate",
                "expected_profit",
                "actual_profit",
                "expected_roi",
                "actual_roi",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nSaved individual bets: {bets_path}")
    print(f"Saved overall summary: {overall_path}")
    print(f"Saved seasonal summary: {season_path}")


if __name__ == "__main__":
    main()