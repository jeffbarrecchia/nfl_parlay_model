from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "data" / "team_efficiency_features.csv"

# Include enough seasons to establish team form before the walk-forward
# evaluation period. The current season may be incomplete.
SEASONS = list(range(2010, 2027))


def safe_mean(series):
    return series.mean() if len(series) else np.nan


def main():
    print(f"Downloading play-by-play data for {SEASONS[0]}–{SEASONS[-1]}...")
    pbp = nfl.load_pbp(SEASONS).to_pandas()

    required = [
        "game_id",
        "game_date",
        "season",
        "week",
        "posteam",
        "defteam",
        "play_type",
        "epa",
        "success",
        "pass_attempt",
        "rush_attempt",
    ]

    missing = [column for column in required if column not in pbp.columns]
    if missing:
        raise ValueError(f"Play-by-play data missing columns: {missing}")

    pbp["game_date"] = pd.to_datetime(pbp["game_date"], errors="coerce")

    # Keep ordinary offensive plays. Exclude kickoffs, punts, field goals,
    # kneels, spikes, and plays without a meaningful offensive team.
    pbp = pbp[
        pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & pbp["game_date"].notna()
    ].copy()

    pbp["success"] = pd.to_numeric(pbp["success"], errors="coerce")
    pbp["epa"] = pd.to_numeric(pbp["epa"], errors="coerce")

    pbp["pass_attempt"] = pd.to_numeric(
        pbp["pass_attempt"], errors="coerce"
    ).fillna(0)

    pbp["rush_attempt"] = pd.to_numeric(
        pbp["rush_attempt"], errors="coerce"
    ).fillna(0)

    print(f"Usable offensive plays: {len(pbp):,}")

    # Offensive statistics: aggregate the plays each team ran.
    offense = (
        pbp.groupby(["game_id", "game_date", "season", "week", "posteam"])
        .agg(
            offensive_epa=("epa", "mean"),
            offensive_success_rate=("success", "mean"),
            offensive_plays=("epa", "size"),
            pass_epa_sum=("epa", lambda s: s[pbp.loc[s.index, "pass_attempt"].eq(1)].sum()),
            pass_attempts=("pass_attempt", "sum"),
            rush_epa_sum=("epa", lambda s: s[pbp.loc[s.index, "rush_attempt"].eq(1)].sum()),
            rush_attempts=("rush_attempt", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    offense["passing_epa_per_play"] = (
        offense["pass_epa_sum"] / offense["pass_attempts"].replace(0, np.nan)
    )
    offense["rushing_epa_per_play"] = (
        offense["rush_epa_sum"] / offense["rush_attempts"].replace(0, np.nan)
    )

    # Defensive statistics: EPA and success rate allowed by each defense.
    defense = (
        pbp.groupby(["game_id", "game_date", "season", "week", "defteam"])
        .agg(
            defensive_epa_allowed=("epa", "mean"),
            defensive_success_allowed=("success", "mean"),
            defensive_plays=("epa", "size"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # Join the offense and defense views to form one row per team per game.
    stats = offense.merge(
        defense,
        on=["game_id", "game_date", "season", "week", "team"],
        how="outer",
    )

    stats = stats.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    # Rolling pregame statistics use only previous games.
    metric_columns = [
        "offensive_epa",
        "offensive_success_rate",
        "passing_epa_per_play",
        "rushing_epa_per_play",
        "defensive_epa_allowed",
        "defensive_success_allowed",
    ]

    for metric in metric_columns:
        stats[f"pregame_{metric}"] = (
            stats.groupby("team")[metric]
            .transform(
                lambda s: s.shift(1).rolling(window=8, min_periods=1).mean()
            )
        )

    # Include prior-game counts so the model can distinguish early-season
    # estimates from estimates based on a full recent sample.
    stats["prior_games"] = stats.groupby("team").cumcount()

    stats = stats[
        [
            "game_id",
            "game_date",
            "season",
            "week",
            "team",
            "prior_games",
        ] + [f"pregame_{metric}" for metric in metric_columns]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(stats):,} team-game rows to:")
    print(OUTPUT_PATH)
    print("\nSample:")
    print(stats.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()