from datetime import datetime
from pathlib import Path

import nflreadpy as nfl

current_year = datetime.now().year
seasons = list(range(2010, current_year + 1))

print(f"Downloading NFL data for seasons {seasons[0]}–{seasons[-1]}...")

df = nfl.load_schedules(seasons).to_pandas()

output_path = Path("data") / "historical_games.csv"
df.to_csv(output_path, index=False)

print(f"Saved {len(df)} games to {output_path}")
print(f"2026 games: {len(df[df['season'] == 2026])}")