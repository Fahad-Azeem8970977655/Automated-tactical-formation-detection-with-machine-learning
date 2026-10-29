from analytics import *

ctx = get_context("League comparison", cluster=False)
ddir = ctx["ddir"]
st.title("🌍 League comparison - A-League vs Bundesliga")
st.caption("Same shape measures, computed the same way, for every SkillCorner (A-League) and Bundesliga (DFL) match found in the data folder. "
           "Bundesliga tracking is stadium-based and tracks all players; SkillCorner is broadcast-based with extrapolated positions.")
games = {g: v for g, v in available_games(ddir).items() if v[0] in ("sc", "dfl")}
LOAD = {"sc": load_sc_game, "dfl": load_dfl_game}


@st.cache_data(show_spinner="Computing shapes for every match (first run a few minutes)…")
def league_table(ddir):
    rows = []
    for g, (kind, p1, p2) in games.items():
        if kind == "dfl" and len(pd.read_csv(p1, usecols=["period_id"])) < 5000:
            continue                                            # skip the 100-second test sample
        d = LOAD[kind](p1, p2)
        for tm in ("home", "away"):
            w, _, _ = build_windows((g, ddir), d, tm, 10)
            if len(w) < 20:
                continue
            w = w.assign(league="A-League" if kind == "sc" else "Bundesliga", match=g, side=tm)
            rows.append(w)
    return pd.concat(rows, ignore_index=True)


df = league_table(ddir)
if df.league.nunique() < 2:
    st.warning("Need both SkillCorner (A-League) and Bundesliga matches in the data folder."); st.stop()
state = st.radio("Phase", ["Out of possession", "In possession", "All windows"], horizontal=True, key="lc_state")
d = df if state == "All windows" else df[df.poss < .45] if state.startswith("Out") else df[df.poss > .55]
M = ["def_line", "hull_area", "width", "length", "compactness"]
t = d.groupby("league").agg(matches=("match", "nunique"), windows=("match", "size"), **{m: (m, "mean") for m in M}).round(1)
st.dataframe(t, width="stretch")
a, b = d[d.league == "A-League"], d[d.league == "Bundesliga"]
r = np.random.default_rng(0)
rows = []
for m in M:
    x, y = a[m].to_numpy(), b[m].to_numpy()
    diffs = [r.choice(y, len(y)).mean() - r.choice(x, len(x)).mean() for _ in range(500)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rows.append(dict(measure=m, bundesliga_minus_aleague=y.mean() - x.mean(), ci_low=lo, ci_high=hi, clearly_different=bool(lo > 0 or hi < 0)))
st.subheader("Bundesliga minus A-League (95% bootstrap interval)")
st.dataframe(pd.DataFrame(rows).set_index("measure").round(1), width="stretch")
st.bar_chart(t[["def_line", "width", "length"]].T)
st.subheader("Per match")
st.dataframe(d.groupby(["league", "match"]).agg(windows=("match", "size"), def_line=("def_line", "mean"), hull_area=("hull_area", "mean"), width=("width", "mean")).round(1), width="stretch")
st.caption("Differences can come from the leagues, the tracking technology, or the pitch scaling (Bundesliga pitches are assumed 105 x 68 m). "
           "Windows from the same match are correlated, so the real uncertainty is wider than the intervals shown.")
