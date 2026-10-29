import re
from core import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ctx = get_context("ML lab", cluster=False)
ddir = ctx["ddir"]
st.title("🤖 ML lab - can shape alone predict the defensive block?")
st.caption("Random Forest trained on tracking-derived shape features, labelled with SkillCorner's own low / medium / high block "
           "(phases_of_play). Validation is leave-one-match-out: the model is always tested on a match it never saw.")
BLOCKS = ["low_block", "medium_block", "high_block"]
FEATS = ["hull_area", "length", "width", "def_line", "centroid_x", "compactness"]
games = available_games(ddir)
win = st.slider("Window length (s)", 5, 20, 10, 5, key="ml_win")


@st.cache_data(show_spinner="Building labelled windows from every SkillCorner match…")
def labelled(ddir, win):
    out = []
    for g, (kind, tp, mp) in games.items():
        if kind != "sc":
            continue
        mid = re.search(r"(\d+)_tracking_extrapolated", os.path.basename(tp)).group(1)
        pf = glob.glob(os.path.join(os.path.dirname(tp), f"*{mid}_phases_of_play.csv"))
        if not pf:
            continue
        mj, ph, d = json.load(open(mp, encoding="utf-8")), pd.read_csv(pf[0]), load_sc_game(tp, mp)
        start = {p["period"]: p["start_frame"] for p in mj["match_periods"]}
        for tm, tid in (("home", mj["home_team"]["id"]), ("away", mj["away_team"]["id"])):
            w, _, _ = build_windows((g, ddir), d, tm, win)
            if w.empty:
                continue
            mf = np.array([start[p] + ((m - 45 * (p - 1)) * 60 + win / 2) * 10 for p, m in zip(w.period, w.minute)])
            o = ph[(ph.team_in_possession_id != tid) & ph.team_out_of_possession_phase_type.isin(BLOCKS)]
            hit = (mf[:, None] >= o.frame_start.to_numpy()) & (mf[:, None] < o.frame_end.to_numpy())
            ok = hit.any(1)
            x = w[ok].copy()
            x["label"] = o.team_out_of_possession_phase_type.to_numpy()[hit.argmax(1)[ok]]
            x["match"] = f"{mj['home_team']['short_name']} v {mj['away_team']['short_name']}"
            x["team"] = tm
            out.append(x)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


df = labelled(ddir, win)
if df.empty or df.match.nunique() < 2:
    st.warning("Need at least 2 SkillCorner matches (tracking + phases_of_play) in the data folder."); st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Labelled windows", len(df)); c2.metric("Matches", df.match.nunique())
c3.metric("Classes", ", ".join(sorted(df.label.str.replace("_block", "").unique())))

pred = pd.Series(index=df.index, dtype=object)
imp = np.zeros(len(FEATS))
rows = []
for m in df.match.unique():
    tr, te = df.match != m, df.match == m
    clf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0).fit(df.loc[tr, FEATS], df.label[tr])
    pred[te] = clf.predict(df.loc[te, FEATS]); imp += clf.feature_importances_ / df.match.nunique()
    rows.append(dict(held_out_match=m, windows=int(te.sum()), accuracy=accuracy_score(df.label[te], pred[te]),
                     baseline_majority=(df.label[te] == df.label[tr].mode()[0]).mean()))
res = pd.DataFrame(rows)
acc, f1 = accuracy_score(df.label, pred), f1_score(df.label, pred, average="macro")
a, b, c = st.columns(3)
a.metric("Leave-one-match-out accuracy", f"{acc:.0%}")
b.metric("Macro F1", f"{f1:.2f}")
c.metric("Majority-class baseline", f"{(df.label == df.label.mode()[0]).mean():.0%}")

L, R = st.columns(2)
with L:
    st.subheader("Per held-out match")
    st.dataframe(res.round(2), width="stretch", hide_index=True)
    st.subheader("Confusion matrix (rows = SkillCorner label)")
    labs = sorted(df.label.unique())
    st.dataframe(pd.DataFrame(confusion_matrix(df.label, pred, labels=labs), index=labs, columns=labs), width="stretch")
with R:
    st.subheader("Which shape features matter?")
    st.bar_chart(pd.Series(imp, index=FEATS).sort_values())
st.caption("If accuracy beats the baseline on unseen matches, team shape genuinely encodes the block. Small gaps mean the labels "
           "(which also use ball and event context) carry information the shape alone doesn't.")
