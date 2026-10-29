from scipy.ndimage import gaussian_filter
from analytics import *

ctx = get_context("Phase gallery", cluster=False)
ddir = ctx["ddir"]
st.title("🖼️ Phase gallery - what each tactical phase looks like")
st.caption("All SkillCorner A-League windows are clustered into tactical phases. For each phase: where the outfield players spend time (glow), "
           "the average formation shape (dots and outline) and the numbers behind it. Teams attack left → right.")
games = {g: v for g, v in available_games(ddir).items() if v[0] == "sc"}
FEATS = ["hull_area", "length", "width", "def_line", "centroid_x", "poss"]


@st.cache_data(show_spinner="Pooling windows from every A-League match (first run ~1-2 min)…")
def pooled(ddir):
    W, S, L = [], [], []
    for g, (_, tp, mp) in games.items():
        d = load_sc_game(tp, mp)
        for tm in ("home", "away"):
            w, shp, ln = build_windows((g, ddir), d, tm, 10)
            if len(w):
                W.append(w); S.append(shp); L.append(ln)
    return pd.concat(W, ignore_index=True), np.concatenate(S), np.concatenate(L)


df, shapes, lines = pooled(ddir)
k = st.slider("Number of phases", 2, 6, 4, key="pg_k")
lab = KMeans(k, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(df[FEATS]))
df["cluster"] = lab
names = name_clusters(df, lab)
PAL = ["#4C9BE8", "#F2A541", "#5CC689", "#E8604C", "#B07CE8", "#4CE8D8"]
order = df.groupby("cluster").poss.mean().sort_values(ascending=False).index
for n, c in enumerate(order):
    m = np.where(lab == c)[0]
    sub = df.iloc[m]
    form = sub.formation.mode()[0]
    pts = shapes[m].reshape(-1, 2)
    fm = m[(sub.formation == form).to_numpy()]
    avg = np.mean([shapes[i][np.lexsort((shapes[i][:, 1], lines[i]))] for i in fm], axis=0)       # average shape, ordered by line then width
    ln_sorted = np.sort(lines[fm[0]])
    col = PAL[n % len(PAL)]
    st.markdown(f"### {names[c]}")
    L_, R_ = st.columns([2.3, 1])
    with L_:
        fig, ax = plt.subplots(figsize=(11, 7), facecolor="#0e1117")
        draw_pitch(ax, bg="#0e1117", ln="#555")
        h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[70, 45], range=[[0, PL], [0, PW]])
        ax.imshow(gaussian_filter(h, 1.8).T, extent=(0, PL, 0, PW), origin="lower", cmap="magma", alpha=.85, aspect="auto", zorder=.5)
        ax.add_patch(hull_poly(avg, fc="none", ec=col, lw=2.5, zorder=3))
        ax.scatter(avg[:, 0], avg[:, 1], s=260, c=[LINE_COLS[v % 4] for v in ln_sorted], ec="white", lw=1.5, zorder=4)
        ax.text(PL / 2, PW + 3.4, f"average shape: {form}", ha="center", color=col, fontsize=14, fontweight="bold")
        st.pyplot(fig, clear_figure=True)
    with R_:
        st.metric("Share of all windows", f"{len(m) / len(df):.0%}")
        st.metric("Possession share", f"{sub.poss.mean():.0%}")
        a, b = st.columns(2)
        a.metric("Back line", f"{sub.def_line.mean():.0f} m"); b.metric("Hull area", f"{sub.hull_area.mean():.0f} m²")
        a.metric("Width", f"{sub.width.mean():.0f} m"); b.metric("Length", f"{sub.length.mean():.0f} m")
        top = sub.formation.value_counts(normalize=True).head(4)
        st.markdown("**Most common formations**")
        for fo, v in top.items():
            st.progress(float(v), text=f"{fo} · {v:.0%}")
st.caption("Dots = average position of each slot (ordered by line, then width) in windows with the phase's most common formation. "
           "Glow = density of all outfield average positions in the phase. Formation names are approximate (see Formation Validator).")
