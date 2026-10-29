from core import *

ctx = get_context('Window explorer')
globals().update(ctx)
st.title('Window explorer')
i = st.slider("Window", 0, len(wdf) - 1, 0, format="%d")
r = wdf.iloc[i]
cL, cR = st.columns([2, 1])
with cL:
    fig, ax = plt.subplots(figsize=(7, 4.8), facecolor="#111")
    pitch_shape(ax, shapes[i], lines[i], col[r.cluster], f"Minute {r.minute:.1f} · {r.formation}")
    for lv in range(lines[i].max() + 1):
        ax.axvline(shapes[i][lines[i] == lv, 0].mean(), c=LINE_COLS[lv % 4], ls="--", lw=.8, alpha=.7)
    st.pyplot(fig, clear_figure=True)
with cR:
    st.markdown(f"**Phase:** {r.phase}")
    st.markdown(f"**Formation:** `{r.formation}`")
    st.metric("Hull area", f"{r.hull_area:.0f} m²")
    st.metric("Width × length", f"{r.width:.0f} × {r.length:.0f} m")
    st.metric("Defensive line", f"{r.def_line:.1f} m")
    st.metric("Possession share in window", f"{r.poss:.0%}")
