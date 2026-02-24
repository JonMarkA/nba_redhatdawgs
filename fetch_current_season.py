"""
fetch_current_season.py
────────────────────────────────────────────────────────────────────────────
Fetches live 2025-26 NBA season stats using the nba_api package and produces
a CSV with these columns (features only — MadePlayoffs is NOT included
because that is the TARGET OUTPUT predicted by Azure AutoML):

  TEAM_ID, TEAM_NAME, W_PCT, OFF_RATING, DEF_RATING, NET_RATING,
  AST_PCT, REB_PCT, TS_PCT, PACE, Season

Usage:
    pip install nba_api
    python fetch_current_season.py

Output:
    current_season.csv  — fed into Azure AutoML for prediction

The script pulls from two official NBA Stats endpoints:
  1. leaguestandingsv3      → TEAM_ID, TEAM_NAME, W_PCT
  2. leaguedashteamstats    → OFF_RATING, DEF_RATING, NET_RATING,
                              AST_PCT, REB_PCT, TS_PCT, PACE
────────────────────────────────────────────────────────────────────────────
"""

import time
import argparse
import csv
import sys

try:
    from nba_api.stats.endpoints import leaguedashteamstats, leaguestandingsv3
except ImportError:
    print("ERROR: nba_api is not installed.")
    print("Run:  pip install nba_api")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────
SEASON        = "2025-26"
SEASON_TYPE   = "Regular Season"
OUTPUT_FILE   = "current_season.csv"
REQUEST_DELAY = 1.0   # seconds between API calls (avoid rate limiting)

# Feature columns only — MadePlayoffs is the Azure ML output, not an input
CSV_COLUMNS = [
    "TEAM_ID", "TEAM_NAME", "W_PCT",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "REB_PCT", "TS_PCT", "PACE",
    "Season"
]


def fetch_standings() -> dict:
    """
    Pull standings from leaguestandingsv3.
    Returns dict keyed by TEAM_ID (str) → { TEAM_NAME, W_PCT }
    """
    print(f"  [1/2] Fetching standings for {SEASON}...")
    time.sleep(REQUEST_DELAY)

    resp = leaguestandingsv3.LeagueStandingsV3(
        season=SEASON,
        season_type="Regular Season",
        timeout=30,
    )

    df = resp.get_data_frames()[0]
    df.columns = [c.upper() for c in df.columns]

    standings = {}
    for _, row in df.iterrows():
        tid   = str(int(row["TEAMID"]))
        wins  = int(row["WINS"])
        losses = int(row["LOSSES"])
        w_pct = round(wins / (wins + losses), 3) if (wins + losses) > 0 else 0.0

        standings[tid] = {
            "TEAM_NAME": row["TEAMNAME"],
            "W_PCT":     w_pct,
        }

    print(f"  ✓ Standings fetched — {len(standings)} teams")
    return standings


def fetch_advanced_stats() -> dict:
    """
    Pull advanced team stats from leaguedashteamstats.
    Returns dict keyed by TEAM_ID (str) → stat dict.
    """
    print(f"  [2/2] Fetching advanced stats for {SEASON}...")
    time.sleep(REQUEST_DELAY)

    resp = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense="Advanced",
        per_mode_simple="PerGame",
        timeout=30,
    )

    df = resp.get_data_frames()[0]
    df.columns = [c.upper() for c in df.columns]

    stats = {}
    for _, row in df.iterrows():
        tid = str(int(row["TEAM_ID"]))
        stats[tid] = {
            "OFF_RATING": round(float(row["OFF_RATING"]), 1),
            "DEF_RATING": round(float(row["DEF_RATING"]), 1),
            "NET_RATING": round(float(row["NET_RATING"]), 1),
            "AST_PCT":    round(float(row["AST_PCT"]), 3),
            "REB_PCT":    round(float(row["REB_PCT"]), 3),
            "TS_PCT":     round(float(row["TS_PCT"]), 3),
            "PACE":       round(float(row["PACE"]), 2),
        }

    print(f"  ✓ Advanced stats fetched — {len(stats)} teams")
    return stats


def build_rows(standings: dict, advanced: dict) -> list:
    """Merge standings + advanced stats into CSV-ready rows."""
    rows = []

    for tid, s in standings.items():
        if tid not in advanced:
            print(f"  ⚠ Skipping {s['TEAM_NAME']} — missing from advanced stats")
            continue

        adv = advanced[tid]
        rows.append({
            "TEAM_ID":    tid,
            "TEAM_NAME":  s["TEAM_NAME"],
            "W_PCT":      s["W_PCT"],
            "OFF_RATING": adv["OFF_RATING"],
            "DEF_RATING": adv["DEF_RATING"],
            "NET_RATING": adv["NET_RATING"],
            "AST_PCT":    adv["AST_PCT"],
            "REB_PCT":    adv["REB_PCT"],
            "TS_PCT":     adv["TS_PCT"],
            "PACE":       adv["PACE"],
            "Season":     SEASON,
        })

    rows.sort(key=lambda r: r["TEAM_NAME"])
    return rows


def write_csv(rows: list, filepath: str):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ Saved → {filepath}  ({len(rows)} teams)")


def preview(rows: list):
    print(f"\n  {'TEAM':<32} {'W_PCT':>5}  {'NET':>5}  {'DEF':>5}  {'PACE':>5}")
    print(f"  {'─' * 55}")
    for r in rows[:5]:
        print(f"  {r['TEAM_NAME']:<32} {r['W_PCT']:>5}  "
              f"{r['NET_RATING']:>5}  {r['DEF_RATING']:>5}  {r['PACE']:>5}")
    if len(rows) > 5:
        print(f"  ... and {len(rows) - 5} more teams")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help="Output CSV file path")
    args = parser.parse_args()

    print(f"\n{'─' * 60}")
    print(f"  NBA Stats Fetcher  |  Season: {SEASON}")
    print(f"  Output columns (features only — no MadePlayoffs)")
    print(f"{'─' * 60}\n")

    try:
        standings = fetch_standings()
        advanced  = fetch_advanced_stats()
    except Exception as e:
        print(f"\n  ERROR: NBA API call failed — {e}")
        print("  Possible causes:")
        print("    • No internet connection")
        print("    • Rate limited — wait 60s and retry")
        print("    • Season endpoint not yet available")
        sys.exit(1)

    rows = build_rows(standings, advanced)

    if not rows:
        print("  ERROR: No rows produced — check API responses above")
        sys.exit(1)

    write_csv(rows, args.output)
    preview(rows)

    print(f"\n  ✅ {args.output} is ready to be uploaded to Azure ML for prediction\n")


if __name__ == "__main__":
    main()
