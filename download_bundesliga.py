"""Download the 7 IDSSE Bundesliga matches via kloppy and save them as CSV files.
Run:  python download_bundesliga.py          (saves into the folder  data/bundesliga/)
Add   --test   to download only 500 frames of one match first."""
import os, sys
from kloppy import sportec
import pandas as pd

OUT = os.path.join("data", "bundesliga")
os.makedirs(OUT, exist_ok=True)
MATCHES = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
test = "--test" in sys.argv
if test:
    MATCHES = MATCHES[:1]

for mid in MATCHES:
    print("Downloading", mid, "...", flush=True)
    ds = sportec.load_open_tracking_data(match_id=mid, sample_rate=1 / 5, limit=500 if test else None)  # 25 Hz -> 5 Hz
    df = ds.to_df()
    df.to_csv(os.path.join(OUT, f"{mid}_tracking.csv"), index=False)
    rows = []
    for team in ds.metadata.teams:
        for p in team.players:
            g = lambda o, a: getattr(o, a, None)
            rows.append(dict(team_name=g(team, "name"), ground=str(g(team, "ground")), player_id=g(p, "player_id"),
                             jersey=g(p, "jersey_no"), name=g(p, "name"), starting=g(p, "starting"),
                             position=str(g(p, "starting_position"))))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{mid}_players.csv"), index=False)
    print("  saved", len(df), "frames,", len(rows), "players ->", OUT, flush=True)
print("Done.")
