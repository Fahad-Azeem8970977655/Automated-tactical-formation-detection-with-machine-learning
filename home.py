"""Tactical Analyst - home page.  Run:  streamlit run app.py"""
from core import *

ctx = get_context("Tactical Analyst")
globals().update(ctx)
st.title("⚽ Tactical Formation & Defensive Block Analyst")
st.caption("Unsupervised clustering of team shape from tracking data (K-Means / DBSCAN + convex hulls).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Match", game)
c2.metric("Team", team)
c3.metric("Windows analysed", len(wdf))
c4.metric("Phases found", len([u for u in uniq if u != -1]))

left, right = st.columns([3, 2])
with left:
    st.subheader("Phase profile")
    st.dataframe(wdf.groupby("phase").agg(windows=("cluster", "size"), hull_area=("hull_area", "mean"),
                 def_line=("def_line", "mean"), width=("width", "mean"), length=("length", "mean"),
                 poss=("poss", "mean"), formation=("formation", lambda s: s.mode()[0])).round(1), width="stretch")
with right:
    st.subheader("Formation share")
    st.bar_chart(wdf.formation.value_counts().head(8))

st.subheader("Datasets found")
st.dataframe(pd.DataFrame({"match": list(games), "format": [v[0] for v in games.values()]}), width="stretch", hide_index=True)
st.info("Use the pages in the left sidebar: Timeline → Phase Shapes → Window Explorer → SkillCorner Phases → Team Comparison → Data Export. "
        "Sidebar settings (match, team, window, algorithm) are shared by all pages.")
