"""
Pulls Formula 1 race results for the 2022-2026 seasons from the Jolpica F1
API (https://github.com/jolpica/jolpica-f1).
 
Each row in the output is one driver's result in one race, including their
starting grid position, finishing position, points, status, and basic
driver/constructor/circuit info
 
"""
import os
import requests
import pandas as pd
import time
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASONS = [2022, 2023, 2024, 2025, 2026]
PAGE_SIZE = 100  # Max number of results per page
REQUEST_DELAY = 0.5  # Delay between API requests in seconds
 
def fetch_season_results(season):
    rows = []
    offset = 0
    while True:
        url1 = f"{BASE_URL}/{season}/results"
        params = {"limit": PAGE_SIZE, "offset": offset}
        response = _get_with_retry(url1, params=params)
        data = response.json()["MRData"]
        total = int(data["total"])
        races = data["RaceTable"]["Races"]
 
        for race in races:
            for result in race["Results"]:
                rows.append(_flatten(season, race, result))
 
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY)  # Be polite to the API
        if offset >= total:
            break
    
    return rows
 
def _get_with_retry(url, params, max_retries=5):
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout= 30)
        if resp.status_code == 429:
            wait = 2 ** attempt 
            print(f"Rate limit hit. Waiting {wait}s before retrying...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"Failed to fetch data from {url} after {max_retries} attempts")
 
def _flatten(season, race, result):
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
        "latitude": circuit["Location"]["lat"],
        "longitude": circuit["Location"]["long"],
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
        "position_text": result["positionText"],
        "points": float(result["points"]),
        "laps": int(result["laps"]),
        "status": result["status"]
    }
 
def main():
    all_rows = []
    for season in SEASONS:
        print(f"Fetching results for season {season}...")
        rows = fetch_season_results(season)
        print(f"{len(rows)} rows fetched for season {season}")
        all_rows.extend(rows)
    
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(SCRIPT_DIR, "f1_results_2022_2026.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
 
if __name__ == "__main__":
    main()