from analytics import *

ctx = get_context("Opponent & context", cluster=False)
ddir = ctx["ddir"]
st.title("🆚 Opponent, game state and ball-relative shape")
st.caption("Fixes three blind spots of single-team analysis: the opposition's shape, the scoreline, and where the ball is.")
df = context_all(ddir)
if df.empty:
    st.warning("Need SkillCorner tracking + phases_of_play + dynamic_events + match files."); st.stop()
per10 = 60.0
FE = ["hull_area", "length", "width", "def_line", "centroid_x", "poss"]
k = st.slider("Shape clusters", 2, 6, 4, key="oc_k")
df["cluster"] = KMeans(k, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(df[FE]))
nm = name_clusters(df, df.cluster.to_numpy()); df["phase"] = df.cluster.map(nm)
out_ = df[df.poss < .45]
c1, c2, c3 = st.columns(3)
c1.metric("Windows", f"{len(df):,}"); c2.metric("Out-of-possession windows", f"{len(out_):,}"); c3.metric("Matches", df.match.nunique())

st.header("1 · Does the opponent's shape change how a team defends?")
oc = ["opp_def_line", "opp_width", "opp_hull", "centroid_gap", "width_diff"]
corr = out_[["def_line", "hull_area", "width"] + oc].corr().loc[["def_line", "hull_area", "width"], oc].round(2)
st.dataframe(corr, width="stretch")
out_ = out_.assign(opp_depth=pd.qcut(out_.opp_def_line, 3, labels=["opp low", "opp mid", "opp high"]))
st.bar_chart(out_.groupby("opp_depth", observed=True)[["def_line", "hull_area"]].mean().round(1))
st.caption("Correlations of the defending team's shape with the opposition's. 'opp low/mid/high' = depth of the opposition's own back line. "
           "A strong negative def_line correlation is largely mechanical: when the opponent has the ball and pushes up, the defending team drops - "
           "it shows the two teams move together, not that one causes the other.")

st.header("2 · Game state: is the low block really worse - or just protecting a lead?")
order = ["winning", "drawing", "losing"]
g = df.assign(is_out=df.poss < .45).groupby(["is_out", "state"]).agg(windows=("match", "size"), def_line=("def_line", "mean"), hull=("hull_area", "mean"),
              pressures=("pressures", "mean"), shots_against=("shots_against", "mean"), shots_for=("shots_for", "mean"))
g[["pressures", "shots_against", "shots_for"]] *= per10
st.dataframe(g.round(2), width="stretch")
dblocks = [n for n in nm.values() if "Defensive block" in n]
pv = df[df.phase.isin(dblocks)].pivot_table(index="phase", columns="state", values="shots_against", aggfunc="mean")[order] * per10
st.subheader("Shots conceded per 10 min - same block, different scoreline")
st.dataframe(pv.round(2), width="stretch"); st.bar_chart(pv.T)
st.caption("If a low block concedes more than a mid block even when the score is held constant, depth itself matters; "
           "if the gap closes within a row, game state explains part of it.")

st.header("3 · Shape relative to the ball")
b = out_.assign(ball_zone=pd.cut(out_.ball_x, [0, 25, 40, 55, 70, 105], labels=["0-25", "25-40", "40-55", "55-70", "70+"]))
t = b.groupby("ball_zone", observed=True).agg(windows=("match", "size"), def_line=("def_line", "mean"), near5=("near5", "mean"), far5=("far5", "mean"), ball_dist=("ball_dist", "mean")).round(1)
st.dataframe(t, width="stretch")
st.bar_chart(t[["near5", "far5"]])
st.caption("near5 / far5 = average spread of the 5 defenders closest to / furthest from the ball (smaller = tighter). "
           "Ball zone is the ball's distance from the defending team's own goal.")
