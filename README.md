# Tactical Formation & Defensive Block Analyst
Streamlit multipage app: unsupervised shape clustering (K-Means / DBSCAN + convex hulls) and a supervised block classifier.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Data (not included - too large)
Create a `data/` folder next to `app.py` and copy in:
* Metrica sample files (`Sample_Game_*`)
* `opendata-master.zip` (SkillCorner repo zip, read directly)
* For each SkillCorner match: `*_tracking_extrapolated.jsonl` (~90 MB, real file - not a Git-LFS pointer), `*_match.json`, `*_phases_of_play.csv`, `*_dynamic_events.csv`

Matches are discovered automatically. In the sidebar, set the data folder if it is not `data/`.

## Pages
Home · Timeline · Phase Shapes · Window Explorer · SkillCorner Phases · Team Comparison · Data Export · ML Lab · Formation Validator · Phase Dynamics · Data Quality · League Comparison · Tactical Board · Heat & Pass Maps · Phase Gallery · Opponent & Context · Transitions · Roles · Pitch Control · Team Report · Shape vs Outcomes (needs `*_dynamic_events.csv` too; first open ~2 min)
(Team Comparison analyses every match on first open - about 2 minutes, then cached.)

`analytics.py` holds the opponent / ball / game-state, loss-recovery and pitch-control code. First open of the heavier pages takes 1-3 min (then cached).

## Bundesliga (DFL / IDSSE) data - optional
```
pip install kloppy
python download_bundesliga.py          # saves data/bundesliga/<id>_tracking.csv + <id>_players.csv for 7 matches
```
The dashboard lists these as "Bundesliga …" in the match selector (Home, Timeline, Phase Shapes, Window Explorer, Team Comparison, Data Export).
Coordinates are kloppy-normalised and scaled to a 105 x 68 m pitch; dead-ball frames are dropped. The event-based pages (Opponent & Context, Transitions, Roles, Pitch Control, ML Lab, Shape vs Outcomes) still use SkillCorner only.

## Changes in this version
* Shots are counted once from SkillCorner dynamic events (end_type == "shot"), not from phase flags.
* Formation Validator adds a role-aware, match-level formation fit (average position of each starter, with / without the ball).
* Data Quality page: all vs detected-only positions (SkillCorner extrapolates players outside the camera view).
* Team Report adds 95% bootstrap intervals and only writes a scouting note when the difference is clear.
