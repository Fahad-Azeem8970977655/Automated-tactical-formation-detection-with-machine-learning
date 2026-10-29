from core import *

ctx = get_context('Phase timeline')
globals().update(ctx)
st.title('Phase timeline')
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw=dict(height_ratios=[4, 1]))
a1.plot(wdf.minute, wdf.def_line, c="#555", lw=0.8, zorder=1)
a1.scatter(wdf.minute, wdf.def_line, c=[col[l] for l in labels], s=28, zorder=2)
a1.set_ylabel("Defensive line height (m from own goal)")
a1.axvline(45, c="#999", ls=":"); a1.grid(alpha=.2)
a2.scatter(wdf.minute, np.zeros(len(wdf)), c=[col[l] for l in labels], marker="s", s=60)
a2.set_yticks([]); a2.set_xlabel("Match minute")
a1.legend(handles=[plt.Line2D([], [], marker="o", ls="", c=col[l], label=names[l]) for l in uniq],
          fontsize=8, loc="upper left", ncol=2)
st.pyplot(fig, clear_figure=True)

cA, cB = st.columns([3, 2])
with cA:
    st.subheader("Phase profile")
    prof = wdf.groupby("phase").agg(windows=("cluster", "size"), hull_area=("hull_area", "mean"),
                                   length=("length", "mean"), width=("width", "mean"),
                                   def_line=("def_line", "mean"), centroid_x=("centroid_x", "mean"),
                                   poss=("poss", "mean"),
                                   top_formation=("formation", lambda s: s.mode()[0])).round(1)
    st.dataframe(prof, width="stretch")
with cB:
    st.subheader("Formation share")
    st.bar_chart(wdf.formation.value_counts().head(8))
if sil_tbl:
    st.caption("Silhouette by k: " + ", ".join(f"k={a}: {b:.2f}" for a, b in sil_tbl.items()))

st.subheader("Phase changes")
min_run = st.slider("Ignore blips shorter than N windows", 1, 6, 2)
run_id = (wdf.cluster != wdf.cluster.shift()).cumsum()
runs = wdf.groupby(run_id).agg(start=("minute", "first"), cluster=("cluster", "first"), n=("minute", "size"),
                               formation=("formation", lambda s: s.mode()[0]))
runs = runs[runs.n >= min_run].reset_index(drop=True)
runs["phase"] = runs.cluster.map(names)
runs["from"] = runs.phase.shift().fillna("-")
runs["formation_from"] = runs.formation.shift().fillna("-")
ch = runs[runs.phase != runs["from"]][["start", "from", "phase", "formation_from", "formation"]]
ch.columns = ["minute", "from phase", "to phase", "from formation", "to formation"]
st.dataframe(ch.round({"minute": 1}), width="stretch", hide_index=True)
