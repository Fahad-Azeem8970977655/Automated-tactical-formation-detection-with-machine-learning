from analytics import *

ctx = get_context("Transitions", cluster=False)
ddir = ctx["ddir"]
st.title("⚡ Transitions - what happens after a team loses the ball?")
st.caption("From every possession loss (SkillCorner phases_of_play) we track the team's front-to-back depth for 15 s and measure how long it takes to "
           "return to its typical defensive depth, whether it wins the ball back within 10 s, and how many pressing engagements it makes in the first 5 s.")
L = losses_all(ddir)
if L.empty:
    st.warning("Need SkillCorner tracking + phases_of_play + dynamic_events files."); st.stop()
a, b, c, d_ = st.columns(4)
a.metric("Possession losses", f"{len(L):,}"); b.metric("Median recovery", f"{L.recover_s.median():.1f} s")
c.metric("Regain within 10 s", f"{L.regain_10s.mean():.0%}"); d_.metric("Presses in first 5 s", f"{L.presses_5s.mean():.1f}")
agg = lambda g: g.agg(losses=("match", "size"), recovery_s=("recover_s", "median"), recovered_15s=("recovered", "mean"), regain_10s=("regain_10s", "mean"), presses_5s=("presses_5s", "mean")).round(2)
st.subheader("By where the ball was lost"); st.dataframe(agg(L.groupby("third")), width="stretch")
st.subheader("By phase before the loss"); st.dataframe(agg(L.groupby("phase_before")), width="stretch")
st.subheader("Does counter-pressing work?")
L["press_level"] = pd.cut(L.presses_5s, [-1, 0, 2, 100], labels=["0", "1-2", "3+"])
cp = agg(L.groupby("press_level", observed=True)); st.dataframe(cp, width="stretch"); st.bar_chart(cp[["regain_10s", "recovered_15s"]])
st.subheader("Teams ranked by recovery speed (≥ 20 losses)")
tm = agg(L.groupby("team")); st.dataframe(tm[tm.losses >= 20].sort_values("recovery_s"), width="stretch")
st.caption("Recovery = first moment the team's depth is at or below its own median out-of-possession depth. Depth = max minus min outfield x.")
