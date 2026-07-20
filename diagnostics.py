"""
diagnostics.py
checks the following:

  1. Basic data sanity on the CURRENT file (duplicates, nulls, weird values)
  2. A diff against the backup file, if present, to see exactly what changed
  3. A walk-forward hit-rate backtest, split into "older" vs "most recent"
     races, so we can see whether accuracy is uniformly worse or just
     worse on the newest rounds

Usage:
    python diagnostics.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "f1_results_2022_2026.csv")
BACKUP_PATH = os.path.join(SCRIPT_DIR, "f1_results_2022_2026_backup.csv")


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_sanity(df: pd.DataFrame):
    section("1. DATA SANITY")
    print(f"Total rows: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")

    dupes = df.duplicated(subset=["season", "round", "driver_id"], keep=False)
    print(f"Duplicate (season, round, driver_id) rows: {dupes.sum()}")
    if dupes.sum() > 0:
        print(df[dupes].sort_values(["season", "round", "driver_id"]).head(10))

    for col in ["grid", "position", "status", "points"]:
        n_null = df[col].isna().sum()
        print(f"Nulls in '{col}': {n_null} ({n_null / len(df):.1%})")

    bad_grid = df[(df["grid"] < 0) | (df["grid"] > 25)]
    if len(bad_grid) > 0:
        print(f"Suspicious grid values (outside 0-25): {len(bad_grid)} rows")
        print(bad_grid[["season", "round", "driver_id", "grid"]].head(10))

    rows_per_race = df.groupby(["season", "round"]).size()
    print(f"\nDrivers per race — min: {rows_per_race.min()}, "
          f"max: {rows_per_race.max()}, median: {rows_per_race.median()}")
    thin_races = rows_per_race[rows_per_race < 15]
    if len(thin_races) > 0:
        print(f"Races with fewer than 15 classified drivers ({len(thin_races)}):")
        print(thin_races)


def check_diff(df: pd.DataFrame):
    section("2. DIFF VS BACKUP")
    if not os.path.exists(BACKUP_PATH):
        print("No backup file found — skipping (this is fine on a fresh run).")
        return
    backup = pd.read_csv(BACKUP_PATH)
    print(f"Backup rows: {len(backup)}  |  Current rows: {len(df)}  "
          f"|  Diff: {len(df) - len(backup):+d}")

    key = ["season", "round", "driver_id"]
    merged = backup.merge(df, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    added = merged[merged["_merge"] == "right_only"]
    removed = merged[merged["_merge"] == "left_only"]
    print(f"Rows added: {len(added)}  |  Rows removed: {len(removed)}")

    both = merged[merged["_merge"] == "both"]
    changed_pos = both[both["position_old"] != both["position_new"]]
    changed_grid = both[both["grid_old"] != both["grid_new"]]
    print(f"Rows with changed 'position': {len(changed_pos)}")
    print(f"Rows with changed 'grid': {len(changed_grid)}")
    if len(changed_pos) > 0:
        print(changed_pos[key + ["position_old", "position_new"]].head(10))


def check_backtest(df: pd.DataFrame):
    section("3. WALK-FORWARD HIT RATE: OLDER VS RECENT")
    try:
        from predict_race_stacking import predict_race
    except ImportError:
        print("Couldn't import predict_race_stacking.py — run this script "
              "from the same folder as your project files.")
        return

    races = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    races = races[races["season"] >= 2023]  # need at least a season of history first
    n = len(races)
    split = int(n * 0.7)
    older = races.iloc[:split]
    recent = races.iloc[split:]

    def hit_rate(race_subset, label):
        hits, total = 0, 0
        for _, row in race_subset.iterrows():
            try:
                result = predict_race(df, int(row["season"]), int(row["round"]), live=False)
            except Exception:
                continue
            result = result[result["position"].notna()]
            if len(result) == 0:
                continue
            hits += (result["predicted_position"] == result["position"]).sum()
            total += len(result)
        rate = hits / total if total else float("nan")
        print(f"{label}: {hits}/{total} exact hits ({rate:.1%}) across "
              f"{len(race_subset)} races")
        return rate

    hit_rate(older, "Older races  ")
    hit_rate(recent, "Recent races ")


def main():
    df = pd.read_csv(RESULTS_PATH)
    check_sanity(df)
    check_diff(df)
    check_backtest(df)


if __name__ == "__main__":
    main()