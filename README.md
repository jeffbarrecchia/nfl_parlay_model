# NFL Parlay Model — starter scaffold

This is the beginning of a standalone Python project for NFL moneyline and spread-cover probability research, with a three-leg parlay selector and plain-text/CSV output.

**Important:** this is a scaffold, not a working betting model yet. It intentionally refuses to produce picks until key pieces are implemented. In particular, the current starter does not yet have historical spread labels/lines or a complete upcoming-game feature pipeline. Do not use it to place bets.

## 1. Requirements

- Python 3.11 or newer recommended
- An API key from an odds provider that supports NFL moneyline and spread markets
- Internet connection when requesting current odds

## 2. Setup on Windows

Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

## 3. Historical data

The next development step is to fetch and prepare historical NFL schedules/results and historical spreads. A starter input CSV is expected at:

`data/historical_games.csv`

Required columns:

- `season`
- `week`
- `gameday`
- `home_team`
- `away_team`
- `home_score`
- `away_score`

The project should be extended to load historical lines and construct ATS outcomes before the spread-cover model is used.

## 4. Configure odds

Copy your odds API key into `config.json` under `odds_api_key`. Do not commit the key to a public repository. For better security, move it to an environment variable before using a remote runner.

## 5. Run

```powershell
python main.py
```

The eventual output will show three selected legs, each leg's model probability and available odds, plus combined parlay odds and estimated hit probability. It will also write candidate rows to `outputs/all_candidates.csv`.

## 6. Modeling safeguards

- Features for a game must use information available before kickoff.
- Use chronological/walk-forward validation, not random train/test splits.
- Track moneyline and spread-cover probability calibration separately.
- For parlays, multiplying leg probabilities assumes independence. The starter restricts to one selection per game but still needs a dependence caveat for selections from different games.
- A positive model-estimated ROI is not evidence of a real edge unless it persists on untouched historical data and prospective tracking.
- Odds and line movement must be timestamped; don't backtest with today's lines.
