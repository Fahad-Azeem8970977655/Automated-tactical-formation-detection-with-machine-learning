from analytics import *

ctx = get_context("Team report", cluster=False)
ddir = ctx["ddir"]
st.title("📋 Team scouting report")
st.caption("One-page summary of a team's defensive and transition tendencies versus the league (the 10 matches loaded), with plain-language notes.")
df, L = context_all(ddir), losses_all(ddir)
if df.empty:
    st.warning("Need SkillCorner tracking + phases_of_play + dynamic_events files."); st.stop()
team = st.selectbox("Team", sorted(df.team.unique()), key="rep_team")
per10 = 60.0

def profile(g):
    o, i = g[g.poss < .45], g[g.poss > .55]
    return pd.Series(dict(
        block_depth_m=o.def_line.mean(), defensive_hull_m2=o.hull_area.mean(), defensive_width_m=o.width.mean(),
        pressing_per10=o.pressures.mean() * per10, conceded_per10=o.shots_against.mean() * per10,
        created_per10=i.shots_for.mean() * per10, possession_hull_m2=i.hull_area.mean(), attacking_width_m=i.width.mean()))

P = df.groupby("team").apply(profile)
Lt = L.groupby("team").agg(recovery_s=("recover_s", "median"), regain_10s=("regain_10s", "mean"))
P = P.join(Lt)
row, lg = P.loc[team], P.mean()
pct = P.rank(pct=True).loc[team] * 100
n_m = df[df.team == team].match.nunique()
a, b, c, d_ = st.columns(4)
a.metric("Matches", n_m); b.metric("Block depth", f"{row.block_depth_m:.1f} m", f"{row.block_depth_m - lg.block_depth_m:+.1f} vs league")
c.metric("Pressing / 10 min", f"{row.pressing_per10:.0f}", f"{row.pressing_per10 - lg.pressing_per10:+.0f}")
d_.metric("Recovery after loss", f"{row.recovery_s:.1f} s", f"{row.recovery_s - lg.recovery_s:+.1f}", delta_color="inverse")
tbl = pd.DataFrame({"team": row, "league mean": lg, "percentile": pct}).round(1)
st.dataframe(tbl, width="stretch")
st.bar_chart(tbl["percentile"])

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan, np.nan
    r = np.random.default_rng(0)
    m = [r.choice(x, len(x)).mean() for _ in range(500)]
    return np.percentile(m, [2.5, 97.5])

T, o_t, i_t, o_all, i_all = df[df.team == team], df[(df.team == team) & (df.poss < .45)], df[(df.team == team) & (df.poss > .55)], df[df.poss < .45], df[df.poss > .55]
spec = [("block_depth_m", o_t.def_line, 1, o_all.def_line.mean()), ("defensive_width_m", o_t.width, 1, o_all.width.mean()),
        ("pressing_per10", o_t.pressures, per10, o_all.pressures.mean() * per10), ("conceded_per10", o_t.shots_against, per10, o_all.shots_against.mean() * per10),
        ("created_per10", i_t.shots_for, per10, i_all.shots_for.mean() * per10)]
cr = []
for nm, x, sc, lgv in spec:
    lo_, hi_ = ci(x); lo_, hi_ = lo_ * sc, hi_ * sc
    cr.append(dict(metric=nm, team=x.mean() * sc, ci_low=lo_, ci_high=hi_, league=lgv, clearly_different=bool(lgv < lo_ or lgv > hi_)))
cr = pd.DataFrame(cr).set_index("metric")
st.subheader("How certain are these? (95% bootstrap intervals)")
st.dataframe(cr.round(2), width="stretch")
st.caption("Resampled over this team's windows. Windows from the same match are correlated, so the real uncertainty is wider than shown - "
           "a team with only 1-2 matches should be read very cautiously.")
sig = cr.clearly_different.to_dict()
notes = []
def say(cond, text):
    if cond: notes.append(text)
say(pct.block_depth_m <= 25 and sig['block_depth_m'], f"Defends deep: block depth {row.block_depth_m:.0f} m vs league {lg.block_depth_m:.0f} m (bottom quartile).")
say(pct.block_depth_m >= 75 and sig['block_depth_m'], f"Defends high: block depth {row.block_depth_m:.0f} m vs league {lg.block_depth_m:.0f} m (top quartile).")
say(pct.pressing_per10 >= 75 and sig['pressing_per10'], "Presses intensely - among the most pressing engagements per 10 minutes in the league.")
say(pct.pressing_per10 <= 25 and sig['pressing_per10'], "Presses little - lets opponents build with few engagements.")
say(pct.recovery_s <= 25, "Recovers its defensive shape quickly after losing the ball.")
say(pct.recovery_s >= 75, "Slow to recover shape after losing the ball - transition vulnerability.")
say(pct.conceded_per10 >= 75 and sig['conceded_per10'], "Concedes more shots than most teams when defending.")
say(pct.created_per10 >= 75 and sig['created_per10'], "Creates many shots in possession.")
say(pct.attacking_width_m >= 75, "Stretches the pitch in possession - wide attacking shape.")
g = df[(df.team == team) & (df.poss < .45)].groupby("state").def_line.mean()
if {"winning", "losing"} <= set(g.index):
    say(g["winning"] - g["losing"] <= -4, f"Drops {g['losing'] - g['winning']:.0f} m deeper when winning ({g['winning']:.0f} m) than when losing ({g['losing']:.0f} m) - protects leads.")
if not notes:
    notes = ["No extreme tendencies versus the league; profile is close to average."]
st.subheader("Scouting notes")
for n in notes:
    st.markdown(f"- {n}")
st.subheader("Block depth by game state")
st.bar_chart(g.reindex(["winning", "drawing", "losing"]))
st.caption(f"Based on {n_m} match(es). With so few matches per team, treat percentiles as indications, not rankings.")
md = f"# Scouting report - {team}\n\nMatches analysed: {n_m}\n\n" + tbl.to_markdown() + "\n\n## Notes\n" + "\n".join(f"- {n}" for n in notes)
st.download_button("Download report (Markdown)", md, f"{team.replace(' ', '_')}_report.md")
