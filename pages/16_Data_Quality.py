from analytics import *

ctx = get_context("Data quality", cluster=False)
ddir = ctx["ddir"]
st.title("🧪 Data quality - how much do estimated positions distort the shapes?")
st.caption("SkillCorner tracks from broadcast video, so players outside the camera view are extrapolated. Here every window is computed twice: "
           "with all positions, and with detected positions only (a player needs to be detected in 30% of a window - at the usual 70% almost no window survives). The gap tells you how much to trust the shape measures.")


@st.cache_data(show_spinner="Comparing all vs detected-only positions across matches (first run ~4 min)…")
def quality(ddir):
    rows = []
    for fl in sc_files(ddir):
        dA = load_sc_game(fl["tp"], fl["mp"]); dD = load_sc_game(fl["tp"], fl["mp"], True)
        for tm in ("home", "away"):
            wA, _, _ = build_windows((fl["game"], ddir, "all"), dA, tm, 10)
            wD, _, _ = build_windows((fl["game"], ddir, "det"), dD, tm, 10, 0.3)   # relaxed: players are seen ~58% of frames
            if wD.empty or wA.empty:
                continue
            m = wA.merge(wD, on=["period", "minute"], suffixes=("_all", "_det"))
            rows.append(dict(match=fl["mid"], team=tm, detected_share=np.isfinite(dD[tm][..., 0]).sum() / np.isfinite(dA[tm][..., 0]).sum(),
                             windows_all=len(wA), windows_detected=len(wD), compared=len(m),
                             hull_diff=(m.hull_area_det - m.hull_area_all).mean(), defline_diff=(m.def_line_det - m.def_line_all).mean(),
                             width_diff=(m.width_det - m.width_all).mean(), hull_corr=m.hull_area_all.corr(m.hull_area_det)))
    return pd.DataFrame(rows)


q = quality(ddir)
a, b, c, d_ = st.columns(4)
a.metric("Positions truly detected", f"{q.detected_share.mean():.0%}")
b.metric("Windows kept when detected-only", f"{q.windows_detected.sum() / q.windows_all.sum():.0%}")
c.metric("Hull-area correlation", f"{q.hull_corr.mean():.2f}")
d_.metric("Mean hull-area difference", f"{q.hull_diff.mean():+.0f} m²")
st.dataframe(q.round(2), width="stretch", hide_index=True)
st.bar_chart(q.set_index("match")[["detected_share"]])
st.caption("A high correlation with small mean differences means extrapolation changes the absolute numbers a little but not the patterns. "
           "'Windows kept' shows what is lost if you insist on detected-only data.")
