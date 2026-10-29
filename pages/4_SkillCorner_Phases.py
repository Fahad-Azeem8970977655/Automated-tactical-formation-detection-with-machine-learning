from core import *

ctx = get_context('SkillCorner phases (zip)', cluster=False)
globals().update(ctx)
st.title('SkillCorner phases (zip)')
zp = find_zip(ddir)
if not zp:
    st.info("Put `opendata-master.zip` in the data folder to enable this tab.")
else:
    pref, minfo, lfs = sc_index(zp)
    st.caption(f"Reading directly from `{os.path.basename(zp)}` - only phases_of_play + match.json are loaded.")
    if lfs and max(lfs.values()) < 1000:
        st.warning("The SkillCorner `*_tracking_extrapolated.jsonl` files in this zip are Git-LFS pointers (≈133 bytes, no coordinates). "
                   "So frame-level shape clustering uses Metrica (or a SkillCorner tracking file you download and put in the data folder); SkillCorner's own block labels + team width/length are used here as an "
                   "independent reference for what low / medium / high blocks look like.")
    lab = st.selectbox("SkillCorner match", [m["label"] for m in minfo])
    mid = next(m["id"] for m in minfo if m["label"] == lab)
    ph, tmap = sc_match(zp, pref, mid)
    tname = st.radio("Team", list(tmap.values()), horizontal=True)
    tid = next(k_ for k_, v in tmap.items() if v == tname)
    tf = sc_team_frame(ph, tid)
    state = st.radio("Phase state", ["Out of possession", "In possession"], horizontal=True)
    s = tf[tf.state == state]

    share = (s.groupby("phase").duration.sum() / s.duration.sum() * 100).sort_values(ascending=False)
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Share of time by phase type (%)")
        st.bar_chart(share.round(1))
    with g2:
        st.subheader("Width / length by phase type (m)")
        st.dataframe(s.groupby("phase").agg(phases=("width", "size"), width=("width", "mean"), length=("length", "mean")).round(1),
                     width="stretch")

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    order = list(share.index)
    for ax, v in zip(axs, ["width", "length"]):
        ax.boxplot([s[s.phase == o][v] for o in order], showfliers=False)
        ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order)
        ax.set_title(f"Team {v} (m)"); ax.tick_params(axis="x", rotation=40); ax.grid(alpha=.2)
    fig.tight_layout(); st.pyplot(fig, clear_figure=True)

    st.subheader("Does unsupervised clustering recover SkillCorner's phase labels?")
    kk = st.slider("k", 2, 8, 4, key="sck")
    sx = StandardScaler().fit_transform(s[["width", "length"]])
    s = s.assign(cluster=KMeans(kk, n_init=10, random_state=0).fit_predict(sx))
    st.dataframe(pd.crosstab(s.cluster, s.phase), width="stretch")

    b = s.assign(bucket=(s.minute // 15 * 15).astype(int)).pivot_table(index="bucket", columns="phase", values="duration", aggfunc="sum").fillna(0)
    st.subheader("Phase mix per 15-minute block (seconds)")
    st.bar_chart(b)
