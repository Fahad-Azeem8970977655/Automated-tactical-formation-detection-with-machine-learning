from core import *

ctx = get_context("Team comparison", cluster=False)
ddir = ctx["ddir"]
st.title("Team comparison - how compact is each team in & out of possession?")
games = available_games(ddir)
win = st.slider("Window length (s)", 5, 30, 10, 5, key="cmp_win")

@st.cache_data(show_spinner="Analysing every match…")
def summarise(ddir, win):
    rows = []
    for g, (kind, p1, p2) in games.items():
        d = {"csv": load_csv_game, "epts": load_epts_game, "sc": load_sc_game, "dfl": load_dfl_game}[kind](p1, p2)
        for tm in ("home", "away"):
            w, _, _ = build_windows((g, ddir), d, tm, win)
            if w.empty:
                continue
            for state, m in (("In possession", w.poss > 0.55), ("Out of possession", w.poss < 0.45)):
                x = w[m]
                if len(x) < 5:
                    continue
                rows.append(dict(match=g, team=tm, state=state, windows=len(x), def_line=x.def_line.mean(),
                                 hull_area=x.hull_area.mean(), width=x.width.mean(), length=x.length.mean(),
                                 compactness=x.compactness.mean(), top_formation=x.formation.mode()[0]))
    return pd.DataFrame(rows)

df = summarise(ddir, win)
if df.empty:
    st.warning("No data found."); st.stop()
df["label"] = df.match.str.replace("Metrica ", "M").str.replace(r"SkillCorner (\d+).*", r"SC \1", regex=True) + " " + df.team
state = st.radio("Phase", ["Out of possession", "In possession"], horizontal=True)
s = df[df.state == state].set_index("label")
metric = st.selectbox("Metric", ["def_line", "hull_area", "width", "length", "compactness"])
st.bar_chart(s[metric].round(1))
st.caption("def_line = height of the back four (m from own goal); lower = deeper block. compactness = mean distance of outfielders to team centre.")
st.dataframe(s.drop(columns="state").round(1), width="stretch")
