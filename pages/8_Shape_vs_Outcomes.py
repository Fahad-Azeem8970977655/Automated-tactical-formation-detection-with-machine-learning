import re
from core import *

ctx = get_context("Shape vs outcomes", cluster=False)
ddir = ctx["ddir"]
st.title("🎯 Which shapes create or prevent chances?")
st.caption("Tracking windows from all SkillCorner matches are clustered into shapes, then linked to SkillCorner dynamic events "
           "(pressing engagements, EPV gain) and phases_of_play (shots for / against) in the same 10-second window.")
games = {g: v for g, v in available_games(ddir).items() if v[0] == "sc"}
win = 10
FEATS = ["hull_area", "length", "width", "def_line", "centroid_x", "poss"]


def count_in(frames, lo, hi, w=None):
    """Sum of w (or count) for events with lo <= frame < hi, via cumulative sums."""
    o = np.argsort(frames); f = frames[o]
    c = np.concatenate([[0], np.cumsum(np.ones(len(f)) if w is None else np.nan_to_num(w[o]))])
    return c[np.searchsorted(f, hi)] - c[np.searchsorted(f, lo)]


@st.cache_data(show_spinner="Linking shapes to events across all matches (first run ~1-2 min)…")
def linked(ddir):
    out = []
    for g, (_, tp, mp) in games.items():
        mid = re.search(r"(\d+)_tracking_extrapolated", os.path.basename(tp)).group(1)
        pf = glob.glob(os.path.join(os.path.dirname(tp), f"*{mid}_phases_of_play.csv"))
        ef = glob.glob(os.path.join(os.path.dirname(tp), f"*{mid}_dynamic_events.csv"))
        if not pf or not ef:
            continue
        mj, ph, d = json.load(open(mp, encoding="utf-8")), pd.read_csv(pf[0]), load_sc_game(tp, mp)
        ev = pd.read_csv(ef[0], usecols=["frame_start", "frame_end", "team_id", "event_type", "end_type", "possession_epv_delta_for"], low_memory=False)
        start = {p["period"]: p["start_frame"] for p in mj["match_periods"]}
        for tm, tid in (("home", mj["home_team"]["id"]), ("away", mj["away_team"]["id"])):
            w, _, _ = build_windows((g, ddir), d, tm, win)
            if w.empty:
                continue
            lo = np.array([start[p] + (m - 45 * (p - 1)) * 600 for p, m in zip(w.period, w.minute)])
            hi = lo + win * 10
            e = ev[ev.team_id == tid]
            eng = e[e.event_type == "on_ball_engagement"].frame_start.to_numpy(float)
            pp = e[e.event_type == "player_possession"]
            w = w.copy()
            w["pressures"] = count_in(eng, lo, hi)
            w["epv_gain"] = count_in(pp.frame_start.to_numpy(float), lo, hi, pp.possession_epv_delta_for.to_numpy(float))
            for col, own in (("shots_for", True), ("shots_against", False)):
                sh = ev[((ev.team_id == tid) == own) & (ev.end_type == "shot")].frame_end.to_numpy(float)
                w[col] = count_in(sh, lo, hi)
            w["match"] = mid
            out.append(w)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


df = linked(ddir)
if df.empty:
    st.warning("Need SkillCorner tracking + phases_of_play + dynamic_events files in the data folder."); st.stop()

k = st.slider("Number of shape clusters", 2, 8, 4, key="so_k")
X = StandardScaler().fit_transform(df[FEATS])
df["cluster"] = KMeans(k, n_init=10, random_state=0).fit_predict(X)
names = name_clusters(df, df.cluster.to_numpy())
df["phase"] = df.cluster.map(names)
per10 = 600 / win                                  # windows per 10 minutes

def rates(g):
    return g.agg(windows=("match", "size"), pressures=("pressures", "mean"), epv_gain=("epv_gain", "mean"),
                 shots_for=("shots_for", "mean"), shots_against=("shots_against", "mean"),
                 def_line=("def_line", "mean"), hull_area=("hull_area", "mean")).assign(
        **{c: lambda x, c=c: x[c] * per10 for c in ["pressures", "epv_gain", "shots_for", "shots_against"]}).round(2)

a, b, c = st.columns(3)
a.metric("Windows", f"{len(df):,}"); b.metric("Matches", df.match.nunique())
c.metric("Shots", int(df.shots_for.sum()))
st.subheader("Outcomes per 10 minutes, by shape phase")
t = rates(df.groupby("phase"))
st.dataframe(t, width="stretch")
L, R = st.columns(2)
L.markdown("**Shots: for vs against**"); L.bar_chart(t[["shots_for", "shots_against"]])
R.markdown("**Pressing engagements**"); R.bar_chart(t[["pressures"]])

st.subheader("Outcomes per 10 minutes, by detected formation")
top = df.formation.value_counts().loc[lambda s: s >= 40].index
f = rates(df[df.formation.isin(top)].groupby("formation")).sort_values("windows", ascending=False)
st.dataframe(f, width="stretch")
st.caption("Shots = possession events that end in a shot (SkillCorner dynamic events), counted once each. EPV gain = sum of expected-possession-value "
           "change on the team's own possessions. Rates are descriptive, not causal - shape also reflects score, opponent and game state.")
