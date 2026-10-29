from analytics import *

ctx = get_context("Roles", cluster=False)
ddir = ctx["ddir"]
st.title("🎭 Player roles - how positions change with and without the ball")
files = [f for f in sc_files(ddir)]
if not files:
    st.warning("No SkillCorner matches found."); st.stop()
labels = {}
for f in files:
    mj = json.load(open(f["mp"], encoding="utf-8")); labels[f"{mj['home_team']['short_name']} v {mj['away_team']['short_name']} ({f['mid']})"] = (f, mj)
sel = st.selectbox("Match", list(labels), key="role_match"); f, mj = labels[sel]
side = st.radio("Team", ["home", "away"], horizontal=True, key="role_side")
tid = mj[f"{side}_team"]["id"]; tname = mj[f"{side}_team"]["short_name"]

@st.cache_data(show_spinner="Computing role positions…")
def role_table(tp, mp, side):
    mj, d = json.load(open(mp, encoding="utf-8")), load_sc_game(tp, mp)
    tid = mj[f"{side}_team"]["id"]
    team_of = {p["id"]: p["team_id"] for p in mj["players"]}
    slot = {}
    for pid, t in team_of.items():
        if t == tid:
            slot[pid] = len(slot)
    pos = apply_flip(d[side], flip_mask(d[side], d["period"]))
    hp = home_possession(d); tp = hp if side == "home" else 1 - hp
    rows = []
    for p in mj["players"]:
        if p["team_id"] != tid or p["player_role"]["acronym"] in ("SUB",) or p["start_time"] != "00:00:00":
            continue
        xy = pos[:, slot[p["id"]]]
        r = dict(role=p["player_role"]["acronym"], player=p["short_name"])
        for nm_, m in (("in", tp == 1), ("out", tp == 0)):
            v = xy[m]; v = v[np.isfinite(v[:, 0])]
            r[f"x_{nm_}"], r[f"y_{nm_}"] = (v[:, 0].mean(), v[:, 1].mean()) if len(v) > 50 else (np.nan, np.nan)
        rows.append(r)
    t = pd.DataFrame(rows)
    t["height_gain"] = t.x_in - t.x_out
    t["width_in"], t["width_out"] = (t.y_in - PW / 2).abs(), (t.y_out - PW / 2).abs()
    t["narrower_in_poss"] = t.width_in - t.width_out
    return t

t = role_table(f["tp"], f["mp"], side)
c1, c2 = st.columns([3, 2])
with c1:
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), facecolor="#111")
    for ax, s, ttl in zip(axs, ("in", "out"), ("In possession", "Out of possession")):
        draw_pitch(ax); ax.scatter(t[f"x_{s}"], t[f"y_{s}"], s=170, c="#4C9BE8", ec="white", zorder=3)
        for _, r in t.iterrows():
            ax.text(r[f"x_{s}"], r[f"y_{s}"], r.role, ha="center", va="center", fontsize=6.5, color="white", zorder=4)
        ax.set_title(f"{tname} · {ttl}", color="#eee", fontsize=10)
    fig.tight_layout(); st.pyplot(fig, clear_figure=True)
with c2:
    inv = t[t.role.isin(["LB", "RB", "LWB", "RWB"]) & (t.narrower_in_poss < -3)]
    st.metric("Full-backs moving inside in possession", len(inv))
    st.caption("A full-back counts as 'inverted' when his average position is ≥ 3 m closer to the centre in possession than out of possession.")
st.dataframe(t.round(1), width="stretch", hide_index=True)
st.caption("Mean positions of the starting XI (team attacks left → right). height_gain = how much further forward each role plays in possession; "
           "narrower_in_poss < 0 means the player tucks inside.")
