import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from nfl_parlay.features import (
    load_historical_games,
    build_team_features,
)
from nfl_parlay.models import load_models
from nfl_parlay.odds import load_odds
from nfl_parlay.selector import build_candidates, select_parlay
from main import rank_upcoming_games


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
DATA_PATH = ROOT / "data" / "historical_games.csv"
MODEL_PATH = ROOT / "models" / "models.joblib"
OUTPUT_PATH = ROOT / "outputs"

st.set_page_config(
    page_title="NFL Model Dashboard",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 NFL Prediction Dashboard")
st.caption("Model predictions, FanDuel comparisons, and experimental parlays")


@st.cache_data(ttl=3600)
def load_config():
    config = {}

    # Load local configuration when available.
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config.update(json.load(f))

    # On Streamlit Cloud, load the API key from app secrets.
    try:
        api_key = st.secrets["odds_api_key"]
        if api_key:
            config["odds_api_key"] = api_key
    except (KeyError, FileNotFoundError):
        pass

    return config

@st.cache_data
def load_historical_data():
    games = load_historical_games(DATA_PATH)
    features = build_team_features(games)
    schedule = pd.read_csv(DATA_PATH)
    return games, features, schedule


@st.cache_resource
def load_saved_models():
    return load_models(MODEL_PATH)


def get_odds(config, refresh=False):
    if refresh:
        st.cache_data.clear()

    return load_odds(config)


if not DATA_PATH.exists():
    st.error("Historical NFL data was not found.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Saved model not found. Train the model locally first.")
    st.stop()

try:
    config = load_config()
    games, features, schedule = load_historical_data()
    moneyline_model, spread_model, feature_columns = load_saved_models()
except Exception as exc:
    st.error(f"Could not initialize the dashboard: {exc}")
    st.stop()


with st.sidebar:
    st.header("Controls")

    refresh_button = st.button(
        "🔄 Refresh FanDuel Odds",
        use_container_width=True,
    )

    st.caption("Odds refreshes use your configured Odds API account.")

    selected_week = st.selectbox(
        "NFL Week",
        options=list(range(1, 19)),
        index=6,
    )

    st.divider()

    st.caption(
        "Predictions are experimental. Model probabilities and estimated "
        "returns are not guarantees of betting value."
    )


@st.cache_data(ttl=300)
def fetch_odds():
    return load_odds(config)


try:
    if refresh_button:
        fetch_odds.clear()

    with st.spinner("Loading FanDuel odds..."):
        odds = fetch_odds()
except Exception as exc:
    st.error(f"Could not retrieve odds: {exc}")
    st.stop()


if odds.empty:
    st.warning("No FanDuel odds are currently available.")
    st.stop()


st.success(
    f"Loaded {len(odds)} odds rows across "
    f"{odds['event_id'].nunique()} games."
)


try:
    rankings = rank_upcoming_games(
        odds=odds,
        games=games,
        moneyline_model=moneyline_model,
        feature_columns=feature_columns,
        schedule=schedule,
    )
except Exception as exc:
    st.error(f"Could not generate game rankings: {exc}")
    st.stop()


if rankings.empty:
    st.warning("No upcoming games could be ranked.")
    st.stop()


if "nfl_week" not in rankings.columns:
    st.error("The rankings do not contain NFL week assignments.")
    st.stop()


rankings["nfl_week"] = pd.to_numeric(
    rankings["nfl_week"],
    errors="coerce",
).astype("Int64")

weekly_rankings = rankings[
    rankings["nfl_week"] == selected_week
].copy()

if weekly_rankings.empty:
    st.warning(f"No games with available odds were found for NFL Week {selected_week}.")
else:
    tab_rankings, tab_market, tab_bets, tab_parlay = st.tabs(
        [
            "🏈 Weekly Rankings",
            "📊 Model vs. Market",
            "💰 Potential Bets",
            "🎲 Three-Leg Parlay",
        ]
    )

    with tab_rankings:
        st.subheader(f"NFL Week {selected_week}")

        st.metric("Games available", len(weekly_rankings))

        display = weekly_rankings[
            [
                "rank",
                "commence_time",
                "away_team",
                "home_team",
                "model_winner",
                "model_win_probability",
                "fanduel_favorite",
                "model_agrees_with_favorite",
            ]
        ].copy()

        # Match each game to its FanDuel moneyline odds.
        odds_lookup = odds[
            ["home_team", "away_team", "commence_time", "home_ml", "away_ml"]
        ].copy()

        odds_lookup["commence_time"] = pd.to_datetime(
            odds_lookup["commence_time"], errors="coerce", utc=True
        )

        display["commence_time"] = pd.to_datetime(
            display["commence_time"], errors="coerce", utc=True
        )

        display = display.merge(
            odds_lookup,
            on=["home_team", "away_team", "commence_time"],
            how="left",
        )

        display["Model Pick Odds"] = display.apply(
            lambda row: (
                row["home_ml"]
                if row["model_winner"] == row["home_team"]
                else row["away_ml"]
                if row["model_winner"] == row["away_team"]
                else None
            ),
            axis=1,
        )

        display["Model Pick Odds"] = display["Model Pick Odds"].apply(
            lambda value: (
                f"{value:+.0f}" if pd.notna(value) else "N/A"
            )
        )

        display["commence_time"] = pd.to_datetime(
            display["commence_time"], utc=True
        ).dt.strftime("%a %b %d, %I:%M %p UTC")

        display["model_win_probability"] = (
            display["model_win_probability"] * 100
        ).round(1)

        display = display.rename(
            columns={
                "rank": "Overall Rank",
                "commence_time": "Kickoff",
                "away_team": "Away Team",
                "home_team": "Home Team",
                "model_winner": "Model Pick",
                "model_win_probability": "Win Probability (%)",
                "Model Pick Odds": "Model Pick Odds",
                "fanduel_favorite": "FanDuel Favorite",
                "model_agrees_with_favorite": "Agrees With Favorite",
            }
        )

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download weekly rankings CSV",
            data=weekly_rankings.to_csv(index=False).encode("utf-8"),
            file_name=f"nfl_week_{selected_week}_rankings.csv",
            mime="text/csv",
        )

    with tab_market:
        st.subheader("Model vs. FanDuel")

        market_columns = [
            "rank",
            "away_team",
            "home_team",
            "home_no_vig_probability",
            "away_no_vig_probability",
            "home_model_vs_market",
            "away_model_vs_market",
        ]

        market = weekly_rankings[market_columns].copy()

        market = market.rename(
            columns={
                "rank": "Rank",
                "away_team": "Away Team",
                "home_team": "Home Team",
                "home_no_vig_probability": "Home Market Probability",
                "away_no_vig_probability": "Away Market Probability",
                "home_model_vs_market": "Home Model Difference",
                "away_model_vs_market": "Away Model Difference",
            }
        )

        st.dataframe(
            market.style.format(
                {
                    "Home Market Probability": "{:.1%}",
                    "Away Market Probability": "{:.1%}",
                    "Home Model Difference": "{:+.1%}",
                    "Away Model Difference": "{:+.1%}",
                },
                na_rep="N/A",
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Positive differences mean the model assigns a higher win "
            "probability than the no-vig market estimate. This does not "
            "establish that a bet is profitable."
        )

    with tab_bets:
        st.subheader("Potential Moneyline Bets")

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
            st.info("No bets passed your configured filters.")
        else:
            week_dates = set(
                pd.to_datetime(
                    weekly_rankings["commence_time"], utc=True
                ).dt.strftime("%Y-%m-%d")
            )

            candidates["game_date"] = pd.to_datetime(
                candidates["commence_time"], utc=True
            ).dt.strftime("%Y-%m-%d")

            weekly_candidates = candidates[
                candidates["game_date"].isin(week_dates)
            ].copy()

            weekly_candidates = weekly_candidates.sort_values(
                "expected_roi",
                ascending=False,
            )

            st.dataframe(
                weekly_candidates[
                    [
                        "commence_time",
                        "selection",
                        "odds",
                        "probability",
                        "edge",
                        "expected_roi",
                    ]
                ].style.format(
                    {
                        "odds": "{:+.0f}",
                        "probability": "{:.1%}",
                        "edge": "{:+.1%}",
                        "expected_roi": "{:+.1%}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Expected ROI is model-based and unvalidated. It should not "
                "be interpreted as evidence of a profitable betting strategy."
            )

    with tab_parlay:
        st.subheader("Experimental Three-Leg Parlay")

        candidates = build_candidates(
            odds=odds,
            features=features,
            moneyline_model=moneyline_model,
            spread_model=spread_model,
            feature_columns=feature_columns,
            min_edge=config.get("min_probability_edge", 0.02),
            min_expected_roi=config.get("min_expected_roi", 0.03),
        )

        if not candidates.empty:
            week_dates = set(
                pd.to_datetime(
                    weekly_rankings["commence_time"], utc=True
                ).dt.strftime("%Y-%m-%d")
            )

            candidates["game_date"] = pd.to_datetime(
                candidates["commence_time"], utc=True
            ).dt.strftime("%Y-%m-%d")

            candidates = candidates[
                candidates["game_date"].isin(week_dates)
            ].copy()

        if candidates.empty:
            st.info("No candidates passed your configured filters.")
        else:
            parlay = select_parlay(
                candidates,
                min_parlay_probability=config.get(
                    "min_parlay_probability", 0.05
                ),
                max_candidates=config.get(
                    "max_candidates_for_combinations", 35
                ),
            )

            if parlay is None:
                st.info("No qualifying three-leg parlay was found.")
            else:
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Estimated Hit Probability",
                    f"{parlay['probability']:.1%}",
                )
                col2.metric(
                    "Combined American Odds",
                    f"{parlay['american_odds']:+}",
                )
                col3.metric(
                    "Estimated ROI",
                    f"{parlay['expected_roi']:+.1%}",
                )

                st.dataframe(
                    parlay["legs"][
                        [
                            "selection",
                            "market",
                            "odds",
                            "probability",
                            "expected_roi",
                        ]
                    ].style.format(
                        {
                            "odds": "{:+.0f}",
                            "probability": "{:.1%}",
                            "expected_roi": "{:+.1%}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                st.warning(
                    "This is an experimental recommendation. Its estimated "
                    "probability and ROI have not been validated as a "
                    "profitable betting strategy."
                )


st.divider()
st.caption(
    f"Last page load: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
    "| Odds are subject to change."
)