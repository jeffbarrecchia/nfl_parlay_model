import requests
import pandas as pd


ODDS_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"


def load_odds(config) -> pd.DataFrame:
    api_key = config.get("odds_api_key", "")
    if not api_key or api_key == "PUT_YOUR_ODDS_API_KEY_HERE":
        print("Add your Odds API key to config.json before requesting live odds.")
        return pd.DataFrame()

    params = {
        "apiKey": api_key,
        "regions": config.get("regions", "us"),
        "markets": ",".join(config.get("markets", ["h2h", "spreads"])),
        "oddsFormat": config.get("odds_format", "american"),
        "dateFormat": "iso",
    }
    response = requests.get(
        ODDS_URL.format(sport=config.get("odds_api_sport_key", "americanfootball_nfl")),
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    events = response.json()
    rows = []

    for event in events:
        for bookmaker in event.get("bookmakers", []):
            if config.get("preferred_bookmakers") and bookmaker["key"] not in config["preferred_bookmakers"]:
                continue
            markets = {m["key"]: m for m in bookmaker.get("markets", [])}
            h2h = markets.get("h2h", {}).get("outcomes", [])
            spreads = markets.get("spreads", {}).get("outcomes", [])
            home = event.get("home_team")
            away = event.get("away_team")
            team_h2h = {x["name"]: x.get("price") for x in h2h}
            team_spread = {x["name"]: x for x in spreads}
            rows.append({
                "event_id": event.get("id"),
                "commence_time": event.get("commence_time"),
                "home_team": home,
                "away_team": away,
                "bookmaker": bookmaker.get("key"),
                "bookmaker_title": bookmaker.get("title"),
                "home_ml": team_h2h.get(home),
                "away_ml": team_h2h.get(away),
                "home_spread": team_spread.get(home, {}).get("point"),
                "home_spread_odds": team_spread.get(home, {}).get("price"),
                "away_spread": team_spread.get(away, {}).get("point"),
                "away_spread_odds": team_spread.get(away, {}).get("price"),
            })

    return pd.DataFrame(rows)
