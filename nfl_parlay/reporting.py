from pathlib import Path
import pandas as pd


def print_report(parlay, config):
    print("\n" + "=" * 48)
    print("       NFL WEEKLY 3-LEG PARLAY MODEL")
    print("=" * 48)
    if not parlay:
        print("No qualifying 3-leg combination was found.")
        print("This starter still needs historical spread labels and upcoming-game features.")
        return

    for i, leg in enumerate(parlay["legs"].itertuples(index=False), start=1):
        print(f"\nLEG {i}: {leg.selection}")
        print(f"  Market: {leg.market}")
        print(f"  Odds: {leg.odds:+g}")
        print(f"  Model probability: {leg.probability:.1%}")
        print(f"  Break-even probability: {leg.break_even_probability:.1%}")
        print(f"  Probability edge: {leg.edge:+.1%}")
        print(f"  Estimated ROI: {leg.expected_roi:+.1%}")

    print("\n" + "-" * 48)
    print("3-LEG PARLAY")
    print(f"Estimated hit probability (independence assumption): {parlay['probability']:.2%}")
    print(f"Combined American odds: {parlay['american_odds']:+d}")
    print(f"Combined decimal odds: {parlay['decimal_odds']:.3f}")
    print(f"Break-even probability at combined odds: {1/parlay['decimal_odds']:.2%}")
    print(f"Estimated ROI per $1 stake: {parlay['expected_roi']:+.2%}")
    print("\nNOTE: Starter scaffold only; do not use its picks for betting.")


def save_candidates_csv(candidates, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(path, index=False)
    print(f"\nSaved candidate list to: {path}")
