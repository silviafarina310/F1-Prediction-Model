"""
update_results.py

Refreshes f1_results_2022_2026.csv with the latest completed races,
without re-fetching the entire multi-season history from scratch.

What it does:
  - Re-fetches the most recent season already in the CSV, in full --
    cheap (a handful of API calls), and it self-heals any results that
    changed after a post-race steward decision, not just adds new rounds.
  - Then checks whether the next season has started yet, and the one
    after that, repeating until it hits a season with no data.
  - Leaves every older season completely untouched.
  - If the API can't be reached or returns nothing for the current
    season, the existing file is left alone rather than risking data loss.

Usage:
    python update_results.py
"""

import os
import time
import requests
import pandas as pd

BASE_URL = "https://api.jolpi.ca/ergast/f1"
RESULTS_PATH = "f1_results_2022_2026.csv"
PAGE_SIZE = 100
REQUEST_DELAY = 0.3


def fetch_season_results(season: int) -> list[dict]:
    rows = []
    offset = 0
    while True:
        url = f"{BASE_URL}/{season}/results/"
        params = {"limit": PAGE_SIZE, "offset": offset}
        resp = _get_with_retry(url, params)
        data = resp.json()["MRData"]
        total = int(data["total"])
        if total == 0:
            return []
        for race in data["RaceTable"]["Races"]:
            for result in race["Results"]:
                rows.append(_flatten(season, race, result))
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY)
        if offset >= total:
            break
    return rows


def _get_with_retry(url: str, params: dict, max_retries: int = 5) -> requests.Response:
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries")


def _flatten(season: int, race: dict, result: dict) -> dict:
    circuit = race["Circuit"]
    driver = result["Driver"]
    constructor = result["Constructor"]
    position = result.get("position")
    return {
        "season": season,
        "round": int(race["round"]),
        "race_name": race["raceName"],
        "circuit_id": circuit["circuitId"],
        "circuit_name": circuit["circuitName"],
        "country": circuit["Location"]["country"],
        "latitude": float(circuit["Location"]["lat"]),
        "longitude": float(circuit["Location"]["long"]),
        "driver_id": driver["driverId"],
        "driver_code": driver.get("code"),
        "given_name": driver["givenName"],
        "family_name": driver["familyName"],
        "dob": driver["dateOfBirth"],
        "nationality": driver["nationality"],
        "constructor_id": constructor["constructorId"],
        "constructor_name": constructor["name"],
        "grid": int(result["grid"]),
        "position": int(position) if position and position.isdigit() else None,
        "position_text": result.get("positionText"),
        "points": float(result["points"]),
        "laps": int(result["laps"]),
        "status": result["status"],
    }


def main():
    if not os.path.exists(RESULTS_PATH):
        raise SystemExit(f"{RESULTS_PATH} not found -- run fetch_f1_data.py first to create the initial dataset.")

    existing = pd.read_csv(RESULTS_PATH)
    last_season = int(existing["season"].max())
    print(f"Existing data covers through season {last_season} ({len(existing)} rows).")

    fresh_seasons = {}
    season = last_season
    while True:
        print(f"Fetching season {season}...")
        try:
            rows = fetch_season_results(season)
        except Exception as e:
            print(f"  Error fetching season {season}: {e}")
            rows = []
        if not rows:
            print(f"  No data available for season {season}.")
            break
        fresh_seasons[season] = rows
        print(f"  -> {len(rows)} rows")
        season += 1

    if last_season not in fresh_seasons:
        print("\nCould not refresh the current season (API issue?) -- leaving the existing file untouched.")
        return

    # Safety net: keep a backup of the previous file before overwriting it.
    backup_path = RESULTS_PATH.replace(".csv", "_backup.csv")
    existing.to_csv(backup_path, index=False)

    kept = existing[existing["season"] < last_season]
    fresh_df = pd.DataFrame([row for rows in fresh_seasons.values() for row in rows])
    updated = pd.concat([kept, fresh_df], ignore_index=True).sort_values(["season", "round"]).reset_index(drop=True)
    updated.to_csv(RESULTS_PATH, index=False)

    new_seasons = sorted(s for s in fresh_seasons if s > last_season)
    print(f"\nSaved {len(updated)} rows to {RESULTS_PATH} ({len(updated) - len(existing):+d} vs before).")
    if new_seasons:
        print(f"New season(s) detected and added: {new_seasons}")
    print(f"Previous version backed up to {backup_path}.")


if __name__ == "__main__":
    main()