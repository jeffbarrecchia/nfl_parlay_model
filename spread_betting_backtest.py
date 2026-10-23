from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "outputs" / "spread_walk_forward_results.csv"
OUTPUT_DIR = ROOT / "outputs"

EDGE_THRESHOLDS = [0.02, 0.03, 0.05, 0.07, 0.10]
STAKE = 10.00


def american_profit_per_dollar(odds):
    """Net profit on a winning bet for each $1 staked."""
    if odds > 0:
        return odds / 100.0
    if odds < 0:
        return 100.0 / abs(odds)
    return np.nan


def calculate_drawdown(profits):
    """Calculate maximum peak-to-trough drawdown in cumulative profit."""
    if len(profits) == 0:
        return 0.0

    cumulative = np.cumsum(profits)
    running_peak = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))[1:]
    drawdowns = running_peak - cumulative

    return float(drawdowns.max())


def simulate_strategy(df, probability_column, edge_threshold, strategy_name):
    """
    Select home or away spread bets when the model's estimated probability
    exceeds the corresponding no-vig market probability by the threshold.

    At most one bet is placed per game.
    """
    rows = []

    for _, game in df.iterrows():
        model_probability = game[probability_column]
        market_home_probability = game["market_probability"]

        if pd.isna(model_probability) or pd.isna(market_home_probability):
            continue

        # Derive away probability as the complement of the home probability.
        market_away_probability = 1.0 - market_home_probability
        model_away_probability = 1.0 - model_probability

        home_edge = model_probability - market_home_probability
        away_edge = model_away_probability - market_away_probability

        # Select the side with the larger edge, but only if it clears
        # the threshold. Otherwise, skip the game.
        if home_edge >= edge_threshold and home_edge >= away_edge:
            side = "home"
            edge = home_edge
            odds = game["home_spread_odds"]
            actual_cover = int(game["spread_target"])
        elif away_edge >= edge_threshold:
            side = "away"
            edge = away_edge
            odds = game["away_spread_odds"]
            actual_cover = 1 - int(game["spread_target"])
        else:
            continue

        if pd.isna(odds) or odds == 0:
            continue

        odds = float(odds)

        if actual_cover == 1:
            profit = STAKE * american_profit_per_dollar(odds)
            result = "win"
        else:
            profit = -STAKE
            result = "loss"

        rows.append({
            "strategy": strategy_name,
            "edge_threshold": edge_threshold,
            "season": game["season"],
            "game_id": game["game_id"],
            "gameday": game["gameday"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "bet_side": side,
            "spread_line": game["spread_line"],
            "odds": odds,
            "model_probability": (
                model_probability if side == "home" else model_away_probability
            ),
            "market_probability": (
                market_home_probability
                if side == "home"
                else market_away_probability
            ),
            "estimated_edge": edge,
            "result": result,
            "stake": STAKE,
            "profit": profit,
        })

    bets = pd.DataFrame(rows)

    if bets.empty:
        return bets, {
            "strategy": strategy_name,
            "edge_threshold": edge_threshold,
            "bets": 0,
            "wins": 0,
            "win_rate": np.nan,
            "total_staked": 0.0,
            "net_profit": 0.0,
            "roi": np.nan,
            "max_drawdown": 0.0,
        }

    # Use chronological order for the drawdown calculation.
    bets = bets.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    wins = int((bets["result"] == "win").sum())
    total_staked = float(bets["stake"].sum())
    net_profit = float(bets["profit"].sum())

    summary = {
        "strategy": strategy_name,
        "edge_threshold": edge_threshold,
        "bets": len(bets),
        "wins": wins,
        "win_rate": wins / len(bets),
        "total_staked": total_staked,
        "net_profit": net_profit,
        "roi": net_profit / total_staked,
        "max_drawdown": calculate_drawdown(bets["profit"].to_numpy()),
    }

    return bets, summary


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run spread_walk_forward.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
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

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in [
        "season",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
        "spread_target",
        "team_probability",
        "market_probability",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Retain only games with a valid outcome, model prediction, market
    # probability, and both available spread prices.
    df = df.dropna(
        subset=[
            "spread_target",
            "team_probability",
            "market_probability",
            "home_spread_odds",
            "away_spread_odds",
        ]
    ).copy()

    # Pushes were excluded when the walk-forward predictions were generated.
    # Validate that the labels are binary before simulating bets.
    df = df[df["spread_target"].isin([0, 1])].copy()

    print(f"Loaded {len(df)} games with usable spread predictions and odds.")
    print(f"Flat stake per bet: ${STAKE:.2f}")

    all_bets = []
    summaries = []

    for strategy_name, probability_column in [
        ("Statistical", "team_probability"),
    ]:
        for threshold in EDGE_THRESHOLDS:
            bets, summary = simulate_strategy(
                df,
                probability_column,
                threshold,
                strategy_name,
            )

            summaries.append(summary)

            if not bets.empty:
                all_bets.append(bets)

    summary_df = pd.DataFrame(summaries)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "spread_betting_summary.csv"
    bets_path = OUTPUT_DIR / "spread_betting_bets.csv"

    summary_df.to_csv(summary_path, index=False)

    if all_bets:
        bets_df = pd.concat(all_bets, ignore_index=True)
        bets_df.to_csv(bets_path, index=False)
    else:
        pd.DataFrame().to_csv(bets_path, index=False)

    display = summary_df.copy()
    display["win_rate"] = display["win_rate"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
    )
    display["roi"] = display["roi"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
    )
    display["total_staked"] = display["total_staked"].map(
        lambda x: f"${x:,.2f}"
    )
    display["net_profit"] = display["net_profit"].map(
        lambda x: f"${x:,.2f}"
    )
    display["max_drawdown"] = display["max_drawdown"].map(
        lambda x: f"${x:,.2f}"
    )

    print("\n===== SPREAD BETTING RESULTS =====")
    print(display.to_string(index=False))

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved individual bets: {bets_path}")


if __name__ == "__main__":
    main()