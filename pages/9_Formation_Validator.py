import re
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from analytics import *

ctx = get_context("Formation validator", cluster=False)
ddir = ctx["ddir"]
st.title("📐 Formation validator - detected shape vs real starting line-up")
st.caption("Two detectors are scored against the starting line-up in SkillCorner's match files: (1) line clustering (1-D K-means, used elsewhere) "
           "and (2) template matching - every window is fitted to each candidate formation with the Hungarian algorithm and the lowest-cost one wins.")
games = {g: v for g, v in available_games(ddir).items() if v[0] == "sc"}
win = 10

CANDS = [(d, 10 - d - f, f) for d in (3, 4, 5) for f in (1, 2, 3, 4) if 10 - d - f >= 2]


def _tpl(d, m, f):
    pts = []
    for xi, n in zip((0, .5, 1), (d, m, f)):
        ys = np.linspace(-1, 1, n) * min(1, .3 * n) if n > 1 else [0]
        pts += [(xi, y) for y in ys]
    return np.array(pts)


TPL = {c: _tpl(*c) for c in CANDS}


def template_formation(p):
    z = np.c_[(p[:, 0] - p[:, 0].min()) / max(np.ptp(p[:, 0]), 1e-6), (p[:, 1] - p[:, 1].mean()) / max(np.ptp(p[:, 1]) / 2, 1e-6)]
    best = min(((lambda C: (C[linear_sum_assignment(C)].mean(), c))(cdist(z, T)) for c, T in TPL.items()), key=lambda t: t[0])
    return "-".join(map(str, best[1]))


def lineup(mj, tid):
    st_ = [p for p in mj["players"] if p["team_id"] == tid and p["start_time"] == "00:00:00" and p["player_role"]["acronym"] not in ("SUB", "GK")]
    g = [p["player_role"]["position_group"] for p in st_]
    d = sum(x in ("Central Defender", "Full Back") for x in g); f = sum(x in ("Center Forward", "Wide Attacker") for x in g)
    return f"{d}-{len(st_) - d - f}-{f}" if len(st_) == 10 else None


@st.cache_data(show_spinner="Fitting formation templates to every window (first run ~2-3 min)…")
def run(ddir):
    rows = []
    for g, (_, tp, mp) in games.items():
        mj, d = json.load(open(mp, encoding="utf-8")), load_sc_game(tp, mp)
        for tm, tid, nm in (("home", mj["home_team"]["id"], mj["home_team"]["short_name"]), ("away", mj["away_team"]["id"], mj["away_team"]["short_name"])):
            lu = lineup(mj, tid)
            w, shp, _ = build_windows((g, ddir), d, tm, win)
            if lu is None or w.empty:
                continue
            w = w.assign(team=nm, match=str(mj["id"]), lineup=lu, template=[template_formation(s) for s in shp])
            rows.append(w)
    return pd.concat(rows, ignore_index=True)


df = run(ddir)
if df.empty:
    st.warning("Need SkillCorner tracking + match.json files."); st.stop()
first = st.checkbox("First half only (fewer substitutions / tactical changes)", True, key="fv_first")
if first:
    df = df[df.period == 1]
df["state"] = np.where(df.poss > .55, "In possession", np.where(df.poss < .45, "Out of possession", "Mixed"))
back = lambda s: s.str.split("-").str[0].astype(int)
df["ok_template"] = df.template == df.lineup
df["ok_cluster"] = df.formation == df.lineup
df["back_template"] = back(df.template) == back(df.lineup)
df["back_cluster"] = back(df.formation) == back(df.lineup)

tab = df[df.state != "Mixed"].groupby("state").agg(windows=("match", "size"), template_exact=("ok_template", "mean"), cluster_exact=("ok_cluster", "mean"),
                                                   template_backline=("back_template", "mean"), cluster_backline=("back_cluster", "mean"))
tab.loc["All windows"] = [len(df), df.ok_template.mean(), df.ok_cluster.mean(), df.back_template.mean(), df.back_cluster.mean()]
c1, c2, c3 = st.columns(3)
c1.metric("Windows scored", f"{len(df):,}")
c2.metric("Template exact match", f"{df.ok_template.mean():.0%}")
c3.metric("Line-clustering exact match", f"{df.ok_cluster.mean():.0%}")
st.subheader("Agreement with the starting line-up")
st.dataframe((tab.assign(**{c: tab[c] * 100 for c in tab.columns[1:]})).round(0), width="stretch")
st.bar_chart(tab[["template_exact", "cluster_exact", "template_backline", "cluster_backline"]])
st.caption("exact = full D-M-F string equals the line-up; backline = only the number of defenders agrees.")

st.subheader("Per team: line-up vs most common detected formation")
per = df[df.state != "Mixed"].groupby(["team", "match", "lineup"]).apply(lambda g: pd.Series(dict(
    template_exact=g.ok_template.mean(), in_poss=g[g.state == "In possession"].template.mode().get(0, "-"),
    out_poss=g[g.state == "Out of possession"].template.mode().get(0, "-"),
    out_exact=g[g.state == "Out of possession"].ok_template.mean()))).reset_index()
st.dataframe(per.round(2), width="stretch", hide_index=True)
st.subheader("Most common template formations")
st.bar_chart(df.template.value_counts().head(8))


st.header("Role-aware formation from average positions")
st.caption("Instead of fitting every 10-second window, each starter's average position (known by name and role) over the whole match is fitted to the "
           "templates - separately with and without the ball. Averaging over thousands of frames removes most of the noise.")


@st.cache_data(show_spinner="Fitting formations to each team's average positions…")
def match_level(ddir):
    rows = []
    for fl in sc_files(ddir):
        mj = json.load(open(fl["mp"], encoding="utf-8"))
        for side in ("home", "away"):
            t = role_positions(fl["tp"], fl["mp"], side)
            t = t[t.role != "GK"]
            lu = lineup(mj, mj[f"{side}_team"]["id"])
            r = dict(match=fl["mid"], team=mj[f"{side}_team"]["short_name"], lineup=lu)
            for s in ("in", "out"):
                pts = t[[f"x_{s}", f"y_{s}"]].dropna().to_numpy()
                r[s] = template_formation(pts) if len(pts) == 10 else None
            rows.append(r)
    return pd.DataFrame(rows).dropna()


ml = match_level(ddir)
for s in ("in", "out"):
    ml[f"exact_{s}"] = ml[s] == ml.lineup
    ml[f"back_{s}"] = back(ml[s]) == back(ml.lineup)
a, b, c, d_, e_ = st.columns(5)
a.metric("Team-matches", len(ml))
b.metric("Exact match - with ball", f"{ml.exact_in.mean():.0%}")
c.metric("Exact match - without ball", f"{ml.exact_out.mean():.0%}")
d_.metric("Defender count right - with ball", f"{ml.back_in.mean():.0%}")
e_.metric("Defender count right - without ball", f"{ml.back_out.mean():.0%}")
st.dataframe(ml[["team", "match", "lineup", "in", "out"]].rename(columns={"in": "with ball", "out": "without ball"}), width="stretch", hide_index=True)
st.caption("Compare with the window-level scores above. Remaining gaps are real tactical behaviour: a 4-3-3 line-up often defends as 4-5-1 or 4-4-2, and "
           "the line-up counts wingers as forwards.")
