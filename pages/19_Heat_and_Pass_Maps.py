from scipy.ndimage import gaussian_filter
from analytics import *

ctx = get_context("Heat & pass maps", cluster=False)
ddir = ctx["ddir"]
st.title("🔥 Heat maps & pass maps")
st.caption("Heat maps come from the tracking (where a player actually spent time); pass maps and the pass network come from SkillCorner dynamic events. "
           "Every team is shown attacking left → right.")
files = [f for f in sc_files(ddir) if f["ef"]]
if not files:
    st.warning("Need SkillCorner tracking + dynamic_events files."); st.stop()
lab = {}
for f in files:
    mj = json.load(open(f["mp"], encoding="utf-8")); lab[f"{mj['home_team']['short_name']} v {mj['away_team']['short_name']} ({f['mid']})"] = (f, mj)
f, mj = lab[st.selectbox("Match", list(lab), key="hp_match")]
side = st.radio("Team", ["home", "away"], horizontal=True, key="hp_side")
tid, tname = mj[f"{side}_team"]["id"], mj[f"{side}_team"]["short_name"]
starters = [p for p in mj["players"] if p["team_id"] == tid and p["player_role"]["acronym"] not in ("SUB",) and p["start_time"] == "00:00:00"]
pname = {p["id"]: f"{p['short_name']} ({p['player_role']['acronym']})" for p in mj["players"] if p["team_id"] == tid}
tab1, tab2, tab3 = st.tabs(["🔥 Heat map", "➡️ Player pass map", "🕸️ Pass network"])


def heat(ax, x, y, cmap="inferno"):
    h, _, _ = np.histogram2d(x, y, bins=[70, 45], range=[[0, PL], [0, PW]])
    h = gaussian_filter(h, 1.6)
    draw_pitch(ax, bg="#111")
    ax.imshow(h.T, extent=(0, PL, 0, PW), origin="lower", cmap=cmap, alpha=.9, aspect="auto", zorder=0.5)


with tab1:
    d = load_sc_game(f["tp"], f["mp"])
    slot = {}
    for p in mj["players"]:
        if p["team_id"] == tid:
            slot[p["id"]] = len(slot)
    pos = apply_flip(d[side], flip_mask(d[side], d["period"]))
    hp = home_possession(d); tp = hp if side == "home" else 1 - hp
    c1, c2 = st.columns(2)
    who = c1.selectbox("Player", ["Whole team (outfield)"] + [pname[p["id"]] for p in starters], key="hp_player")
    st_ = c2.radio("When", ["All", "With the ball", "Without the ball"], horizontal=True, key="hp_state")
    m = np.ones(len(tp), bool) if st_ == "All" else (tp == 1) if st_ == "With the ball" else (tp == 0)
    if who.startswith("Whole"):
        gk = [p["id"] for p in starters if p["player_role"]["acronym"] == "GK"]
        ids = [p["id"] for p in starters if p["id"] not in gk]
    else:
        ids = [p["id"] for p in starters if pname[p["id"]] == who]
    pts = np.concatenate([pos[m][:, slot[i]] for i in ids])
    pts = pts[np.isfinite(pts[:, 0])]
    fig, ax = plt.subplots(figsize=(10, 6.4), facecolor="#111")
    heat(ax, pts[:, 0], pts[:, 1]); ax.set_title(f"{tname} · {who} · {st_.lower()}", color="#eee")
    st.pyplot(fig, clear_figure=True)
    st.caption("Brighter = more time spent there. Based on 5 Hz tracking; SkillCorner extrapolates players outside the camera view.")


@st.cache_data(show_spinner="Reading pass events…")
def passes(ep, mp_, side):
    mj = json.load(open(mp_, encoding="utf-8")); tid = mj[f"{side}_team"]["id"]
    e = pd.read_csv(ep, low_memory=False, usecols=["period", "team_id", "player_id", "event_type", "end_type", "x_end", "y_end", "pass_outcome",
                                                  "player_targeted_id", "player_targeted_x_reception", "player_targeted_y_reception"])
    e = e[(e.event_type == "player_possession") & (e.end_type == "pass") & (e.team_id == tid)].copy()
    sides = mj["home_team_side"]
    def flip(p):
        dirn = sides[int(p) - 1] if int(p) - 1 < len(sides) else sides[-1]
        if side == "away":
            dirn = "right_to_left" if dirn == "left_to_right" else "left_to_right"
        return dirn == "right_to_left"
    fl = e.period.map(flip).to_numpy()
    s = np.where(fl, -1, 1)
    e["x0"], e["y0"] = e.x_end * s + PL / 2, e.y_end * s + PW / 2
    e["x1"], e["y1"] = e.player_targeted_x_reception * s + PL / 2, e.player_targeted_y_reception * s + PW / 2
    return e.dropna(subset=["x0", "y0", "x1", "y1"])


P = passes(f["ef"], f["mp"], side)
with tab2:
    pl_opts = {pname[i]: i for i in P.player_id.unique() if i in pname}
    who = st.selectbox("Player", sorted(pl_opts), key="pm_player")
    q = P[P.player_id == pl_opts[who]]
    fig, ax = plt.subplots(figsize=(10, 6.4), facecolor="#111"); draw_pitch(ax, bg="#143d26")
    for _, r in q.iterrows():
        ok = r.pass_outcome == "successful"
        ax.annotate("", xy=(r.x1, r.y1), xytext=(r.x0, r.y0), arrowprops=dict(arrowstyle="->", color="#5CC689" if ok else "#E8604C", lw=1.4, alpha=.85), zorder=3)
    ax.scatter(q.x0, q.y0, s=18, c="white", zorder=4)
    ax.set_title(f"{who} · {len(q)} passes · {(q.pass_outcome == 'successful').mean():.0%} completed", color="#eee")
    st.pyplot(fig, clear_figure=True)
    st.caption("Green = completed, red = not completed. Arrow starts where the pass was played and ends where it was received (or would have been).")

with tab3:
    mn = st.slider("Minimum passes between two players to draw a link", 1, 10, 3, key="pn_min")
    ok = P[(P.pass_outcome == "successful") & P.player_targeted_id.notna()]
    nodes = P.groupby("player_id").agg(x=("x0", "mean"), y=("y0", "mean"), n=("x0", "size"))
    pair = ok.assign(a=ok[["player_id", "player_targeted_id"]].min(1), b=ok[["player_id", "player_targeted_id"]].max(1)).groupby(["a", "b"]).size()
    fig, ax = plt.subplots(figsize=(10, 6.4), facecolor="#111"); draw_pitch(ax, bg="#143d26")
    for (a, b), n in pair[pair >= mn].items():
        if a in nodes.index and b in nodes.index:
            ax.plot([nodes.x[a], nodes.x[b]], [nodes.y[a], nodes.y[b]], c="white", lw=.6 + n * .35, alpha=.55, zorder=2)
    ax.scatter(nodes.x, nodes.y, s=80 + nodes.n * 9, c="#4C9BE8", ec="white", zorder=4)
    for i, r in nodes.iterrows():
        ax.text(r.x, r.y, pname.get(i, str(i)).split("(")[-1].rstrip(")"), ha="center", va="center", fontsize=7, color="white", zorder=5)
    ax.set_title(f"{tname} pass network · node size = passes played, link width = completed passes between the pair", color="#eee", fontsize=10)
    st.pyplot(fig, clear_figure=True)
    st.dataframe(P.groupby("player_id").agg(passes=("x0", "size"), completed=("pass_outcome", lambda s: (s == "successful").mean())).rename(index=pname).sort_values("passes", ascending=False).round(2), width="stretch")
