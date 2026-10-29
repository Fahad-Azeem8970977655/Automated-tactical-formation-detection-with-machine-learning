from core import *

ctx = get_context('Phase shapes')
globals().update(ctx)
st.title('Phase shapes')
st.caption("Faint outlines = convex hulls of every window in the phase. Solid shape = most representative window "
           "(closest to the cluster centre). Dot colour = detected line (red back → blue front). Team attacks left → right.")
show = [u for u in uniq]
ncol = min(3, len(show))
nrow = int(np.ceil(len(show) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow), facecolor="#111")
axs = np.atleast_1d(axs).ravel()
for ax, l in zip(axs, show):
    m = np.where(labels == l)[0]
    draw_pitch(ax)
    for i in m:
        ax.add_patch(hull_poly(shapes[i], fc=col[l], ec=col[l], alpha=0.06, lw=0.6))
    if l != -1:
        ctr = X[m].mean(0)
        med = m[np.argmin(((X[m] - ctr) ** 2).sum(1))]
        ax.add_patch(hull_poly(shapes[med], fc=col[l], ec="white", alpha=0.3, lw=1.8))
        ax.scatter(*shapes[med].T, c=[LINE_COLS[v % 4] for v in lines[med]], s=110, ec="white", zorder=3)
        f = wdf.formation[m].mode()[0]
    else:
        f = "-"
    ax.set_title(f"{names[l]}\n{len(m)} windows · formation {f}", fontsize=9, color="#eee")
for ax in axs[len(show):]:
    ax.axis("off")
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

st.subheader("Formation × phase")
st.dataframe(pd.crosstab(wdf.phase, wdf.formation), width="stretch")
