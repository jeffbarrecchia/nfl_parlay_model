from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "outputs" / "walk_forward_results.csv"
OUTPUT_DIR = ROOT / "outputs"

STAKE = 10.0

# Minimum predicted probability advantage over the market.
EDGE_THRESHOLDS = [0.02, 0.03, 0.05, 0.07, 0.10]


def american_to_decimal(odds):
    odds = float(odds)

    if odds > 0:
        return 1 + odds / 100

    if odds < 0:
        return 1 + 100 / abs(odds)

    raise ValueError("American odds cannot be zero.")


def american_to_probability(odds):
    odds = float(odds)

    if odds > 0:
        return 100 / (odds + 100)

    if odds < 0:
        return abs(odds) / (abs(odds) + 100)

    return np.nan


def add_market_probabilities(df):
    """Calculate no-vig home and away probabilities from both prices."""
    df = df.copy()

    df["home_raw_probability"] = df["home_moneyline"].apply(
        american_to_probability
    )
    df["away_raw_probability"] = df["away_moneyline"].apply(
        american_to_probability
    )

    total = (
        df["home_raw_probability"]
        + df["away_raw_probability"]
    )

    df["market_home_probability"] = (
        df["home_raw_probability"] / total
    )

    df["market_away_probability"] = (
        df["away_raw_probability"] / total
    )

    return df


def calculate_profit(won, odds, stake=STAKE):
    """Net profit for a flat-stake American-odds wager."""
    decimal_odds = american_to_decimal(odds)

    if won:
        return stake * (decimal_odds - 1)

    return -stake


def summarize_bets(bets, strategy, threshold):
    if bets.empty:
        return {
            "strategy": strategy,
            "edge_threshold": threshold,
            "bets": 0,
            "wins": 0,
            "win_rate": np.nan,
            "total_staked": 0.0,
            "net_profit": 0.0,
            "roi": np.nan,
            "max_drawdown": 0.0,
        }

    bets = bets.sort_values(["season", "gameday"]).copy()

    bets["profit"] = [
        calculate_profit(won, odds)
        for won, odds in zip(bets["won"], bets["odds"])
    ]

    bets["cumulative_profit"] = bets["profit"].cumsum()

    # Bankroll starts at zero profit. Drawdown measures a decline
    # from the highest cumulative profit reached.
    running_peak = bets["cumulative_profit"].cummax()
    drawdown = running_peak - bets["cumulative_profit"]

    total_staked = len(bets) * STAKE
    net_profit = bets["profit"].sum()

    return {
        "strategy": strategy,
        "edge_threshold": threshold,
        "bets": len(bets),
        "wins": int(bets["won"].sum()),
        "win_rate": bets["won"].mean(),
        "total_staked": total_staked,
        "net_profit": net_profit,
        "roi": net_profit / total_staked,
        "max_drawdown": drawdown.max(),
    }


def simulate_strategy(df, strategy, edge_threshold):
    """
    Select a side only when the model's estimated probability exceeds
    that side's no-vig market probability by the required edge.

    This is a diagnostic backtest. Historical prices may not represent
    the exact odds available when a real wager could have been placed.
    """
    if strategy == "Team model":
        home_probability = df["team_probability"]
    elif strategy == "Market-aware model":
        home_probability = df["market_aware_probability"]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    home_edge = (
        home_probability - df["market_home_probability"]
    )

    away_probability = 1 - home_probability

    away_edge = (
        away_probability - df["market_away_probability"]
    )

    # Select the side with a qualifying edge. Never place two bets
    # on the same game.
    home_pick = home_edge >= edge_threshold
    away_pick = away_edge >= edge_threshold

    # If both sides qualify due to model/market inconsistencies,
    # skip the game rather than double-bet.
    conflicting = home_pick & away_pick

    home_pick = home_pick & ~conflicting
    away_pick = away_pick & ~conflicting

    selected = df.loc[home_pick | away_pick].copy()

    if selected.empty:
        return selected

    selected["selection"] = np.where(
        home_pick.loc[selected.index],
        selected["home_team"],
        selected["away_team"],
    )

    selected["odds"] = np.where(
        home_pick.loc[selected.index],
        selected["home_moneyline"],
        selected["away_moneyline"],
    )

    selected["predicted_probability"] = np.where(
        home_pick.loc[selected.index],
        home_probability.loc[selected.index],
        away_probability.loc[selected.index],
    )

    selected["market_probability"] = np.where(
        home_pick.loc[selected.index],
        selected["market_home_probability"],
        selected["market_away_probability"],
    )

    selected["edge"] = (
        selected["predicted_probability"]
        - selected["market_probability"]
    )

    selected["won"] = np.where(
        home_pick.loc[selected.index],
        selected["home_win"] == 1,
        selected["home_win"] == 0,
    )

    selected["strategy"] = strategy
    selected["edge_threshold"] = edge_threshold

    return selected


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}.\n"
            "Run python walk_forward.py first."
        )

    print("Loading walk-forward predictions...")

    df = pd.read_csv(INPUT_PATH)

    required = [
        "season",
        "gameday",
        "home_team",
        "away_team",
        "home_win",
        "team_probability",
        "market_aware_probability",
        "market_probability",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # The original walk-forward output does not currently include
    # historical American odds. Recover them from historical data.
    historical_path = ROOT / "data" / "historical_games.csv"
    historical = pd.read_csv(historical_path)

    odds_columns = [
        "season",
        "gameday",
        "home_team",
        "away_team",
        "home_moneyline",
        "away_moneyline",
    ]

    missing_odds = [
        column for column in odds_columns
        if column not in historical.columns
    ]

    if missing_odds:
        raise ValueError(
            "Historical data is missing odds columns: "
            + ", ".join(missing_odds)
        )

    historical["gameday"] = pd.to_datetime(
        historical["gameday"], errors="coerce"
    )
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")

    for column in ["home_moneyline", "away_moneyline"]:
        historical[column] = pd.to_numeric(
            historical[column], errors="coerce"
        )

    # Match on season, date, and both team names to avoid accidentally
    # joining different games.
    df = df.merge(
        historical[odds_columns],
        on=["season", "gameday", "home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )

    df = df.dropna(
        subset=[
            "home_moneyline",
            "away_moneyline",
            "team_probability",
            "market_aware_probability",
            "home_win",
        ]
    ).copy()

    df = df[
        (df["home_moneyline"] != 0)
        & (df["away_moneyline"] != 0)
    ].copy()

    df = add_market_probabilities(df)

    # The walk-forward market probabilities and the newly calculated
    # probabilities should be consistent. Report the difference.
    probability_difference = (
        df["market_home_probability"] - df["market_probability"]
    ).abs().mean()

    print(f"Games with usable odds: {len(df)}")
    print(
        "Mean absolute difference between market probabilities: "
        f"{probability_difference:.4f}"
    )

    if probability_difference > 0.02:
        print(
            "WARNING: Market probabilities differ materially. "
            "Inspect the odds source and matching before interpreting ROI."
        )

    all_bets = []
    summaries = []

    for strategy in ["Team model", "Market-aware model"]:
        for threshold in EDGE_THRESHOLDS:
            bets = simulate_strategy(df, strategy, threshold)

            summaries.append(
                summarize_bets(bets, strategy, threshold)
            )

            if not bets.empty:
                all_bets.append(bets)

    summary_df = pd.DataFrame(summaries)

    print("\n===== STRATEGY RESULTS =====")
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "win_rate": lambda x: f"{x:.1%}" if pd.notna(x) else "N/A",
                "roi": lambda x: f"{x:.1%}" if pd.notna(x) else "N/A",
                "net_profit": lambda x: f"${x:.2f}",
                "total_staked": lambda x: f"${x:.2f}",
                "max_drawdown": lambda x: f"${x:.2f}",
            },
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "betting_backtest_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if all_bets:
        bets_df = pd.concat(all_bets, ignore_index=True)
        bets_path = OUTPUT_DIR / "betting_backtest_bets.csv"
        bets_df.to_csv(bets_path, index=False)
        print(f"\nSaved individual bets to: {bets_path}")
    else:
        print("\nNo bets qualified at the tested edge thresholds.")

    print(f"Saved summary to: {summary_path}")
    print(
        "\nImportant: This simulation is not proof of future profitability. "
        "Verify historical odds timing and avoid tuning thresholds to the "
        "same test period used to report results."
    )


if __name__ == "__main__":
    main()
