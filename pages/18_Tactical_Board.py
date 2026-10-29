import time
from analytics import *

ctx = get_context("Tactical board", cluster=False)
ddir = ctx["ddir"]
st.title("🎬 Tactical board - watch the match on a pitch")
st.caption("Both teams, the ball, team shapes (convex hulls), detected lines and pitch control for any moment of any match. Step with the slider or press Play.")
games = available_games(ddir)
if not games:
    st.warning("No matches found in the data folder."); st.stop()
LOAD = {"csv": load_csv_game, "epts": load_epts_game, "sc": load_sc_game, "dfl": load_dfl_game}
name = st.selectbox("Match", list(games), key="tb_match")
kind, p1, p2 = games[name]
d = LOAD[kind](p1, p2)
F = len(d["t"])
side = st.radio("Team shown with shape lines", ["home", "away"], horizontal=True, key="tb_side")
o1, o2, o3, o4 = st.columns(4)
hulls = o1.checkbox("Team shapes", True, key="tb_h"); lines_on = o2.checkbox("Formation lines", True, key="tb_l")
trail = o3.checkbox("Ball trail", True, key="tb_t"); ctrl = o4.checkbox("Pitch control", False, key="tb_c")
COL = {"home": "#4C9BE8", "away": "#E8604C"}


def outfield(xy, fl_i):
    """Valid players of one team in one frame, goalkeeper removed (deepest player in the team's own frame)."""
    v = xy[np.isfinite(xy[:, 0])]
    xo = PL - v[:, 0] if fl_i else v[:, 0]
    return np.delete(v, xo.argmin(), 0) if len(v) > 1 else v


def frame_fig(i):
    per = d["period"][i]
    fl = {s: flip_mask(d[s], d["period"])[i] for s in ("home", "away")}
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#111")
    draw_pitch(ax, bg="#143d26")
    if ctrl:
        g, share = voronoi(d["home"][i], d["away"][i])
        ax.imshow(g, extent=(0, PL, 0, PW), origin="lower", cmap="coolwarm_r", alpha=.16, vmin=0, vmax=1, aspect="auto", zorder=1)
    for s in ("home", "away"):
        o = outfield(d[s][i], fl[s])
        if hulls and len(o) >= 3:
            ax.add_patch(hull_poly(o, fc=COL[s], ec=COL[s], alpha=.12, lw=1.2, zorder=2))
        v = d[s][i][np.isfinite(d[s][i][:, 0])]
        ax.scatter(v[:, 0], v[:, 1], c=COL[s], s=150, ec="white", zorder=4)
    if lines_on:                                                  # lines from a +-5 s average (stable shape)
        lo, hi = max(0, i - 25), min(F, i + 26)
        seg = d[side][lo:hi]
        keep = np.isfinite(seg[..., 0]).mean(0) > .7
        if keep.sum() == 11:
            m = np.nanmean(seg[:, keep], 0)
            xo = PL - m[:, 0] if fl[side] else m[:, 0]
            m = np.delete(m, xo.argmin(), 0)
            form, ln = line_formation(PL - m[:, 0] if fl[side] else m[:, 0])
            for lv in range(ln.max() + 1):
                pts = m[ln == lv]
                ax.plot([pts[:, 0].mean()] * 2, [pts[:, 1].min() - 2, pts[:, 1].max() + 2], ls="--", c=LINE_COLS[lv % 4], lw=2, zorder=3)
            ax.text(PL / 2, -1.2, f"{side} shape (±5 s): {form}", ha="center", color="#eee", fontsize=10)
    if trail:
        lo = max(0, i - 25); tr = d["ball"][lo:i + 1]; tr = tr[np.isfinite(tr[:, 0])]
        if len(tr) > 1:
            ax.plot(tr[:, 0], tr[:, 1], c="white", lw=1.5, alpha=.7, zorder=5)
    if np.isfinite(d["ball"][i]).all():
        ax.scatter(*d["ball"][i], c="white", s=90, ec="black", zorder=6)
    for s, x in (("home", 12), ("away", PL - 12)):
        ax.text(x, PW + 3.2, f"{s} → {'left' if fl[s] else 'right'}", ha="center", color=COL[s], fontsize=9)
    ax.set_title(f"{name}  ·  minute {d['t'][i] / 60 + 45 * (per - 1):.1f}", color="#eee", fontsize=11)
    return fig


i = st.slider("Moment (5 Hz frames)", 0, F - 1, min(1500, F - 1), key="tb_i")
box = st.empty()
fig = frame_fig(i); box.pyplot(fig, clear_figure=True); plt.close(fig)
if st.button("▶ Play next 10 seconds"):
    for j in range(i, min(F, i + 50)):
        fig = frame_fig(j); box.pyplot(fig, clear_figure=True); plt.close(fig); time.sleep(.05)
