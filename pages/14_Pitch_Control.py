from analytics import *

ctx = get_context("Pitch control", cluster=False)
ddir = ctx["ddir"]
st.title("🗺️ Pitch control (Voronoi) - who owns the space?")
st.caption("Simple model: every point on the pitch belongs to the team whose nearest player is closer. It ignores speed and reaction time, "
           "and SkillCorner fills in off-camera players, so use it for patterns rather than single-frame truth.")
files = list(sc_files(ddir))
if not files:
    st.warning("No SkillCorner matches found."); st.stop()
lab = {}
for f in files:
    mj = json.load(open(f["mp"], encoding="utf-8")); lab[f"{mj['home_team']['short_name']} v {mj['away_team']['short_name']} ({f['mid']})"] = f
f = lab[st.selectbox("Match", list(lab), key="pc_match")]
mj, d = json.load(open(f["mp"], encoding="utf-8")), load_sc_game(f["tp"], f["mp"])
hn, an = mj["home_team"]["short_name"], mj["away_team"]["short_name"]

@st.cache_data(show_spinner="Computing pitch control over the match…")
def series(tp, mp):
    d = load_sc_game(tp, mp)
    fl = flip_mask(d["home"], d["period"])
    rows = []
    for i in range(0, len(d["t"]), 25):                    # every 5 s
        grid, s = voronoi(d["home"][i], d["away"][i])
        att = grid[:, :35].mean() if fl[i] else grid[:, 71:].mean()   # home's attacking third
        rows.append(dict(minute=d["t"][i] / 60 + 45 * (d["period"][i] - 1), home_control=s, home_attacking_third=att, poss=np.nan))
    return pd.DataFrame(rows)

s = series(f["tp"], f["mp"])
m1, m2, m3 = st.columns(3)
m1.metric(f"{hn} mean pitch control", f"{s.home_control.mean():.1%}")
m2.metric(f"{hn} control in its attacking third", f"{s.home_attacking_third.mean():.1%}")
m3.metric(f"{an} mean pitch control", f"{(1 - s.home_control).mean():.1%}")
st.line_chart(s.set_index("minute")[["home_control", "home_attacking_third"]].rolling(6, min_periods=1).mean())
i = st.slider("Frame (5 Hz)", 0, len(d["t"]) - 1, min(3000, len(d["t"]) - 1), key="pc_frame")
grid, share = voronoi(d["home"][i], d["away"][i])
fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#111")
draw_pitch(ax, bg="#143d26")
ax.imshow(grid, extent=(0, PL, 0, PW), origin="lower", cmap="coolwarm_r", alpha=.35, vmin=0, vmax=1, aspect="auto", zorder=1)
ax.scatter(*d["home"][i].T, c="#4C9BE8", s=130, ec="white", zorder=3, label=hn)
ax.scatter(*d["away"][i].T, c="#E8604C", s=130, ec="white", zorder=3, label=an)
if np.isfinite(d["ball"][i]).all():
    ax.scatter(*d["ball"][i], c="white", s=60, ec="black", zorder=4)
ax.legend(loc="upper left", fontsize=8); ax.set_title(f"Minute {d['t'][i] / 60 + 45 * (d['period'][i] - 1):.1f} · {hn} controls {share:.0%}", color="#eee")
st.pyplot(fig, clear_figure=True)
