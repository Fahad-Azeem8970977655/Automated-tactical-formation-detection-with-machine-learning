from sklearn.mixture import GaussianMixture
from core import *

ctx = get_context("Phase dynamics", cluster=False)
ddir = ctx["ddir"]
st.title("🔄 Phase dynamics - how do teams move between shapes?")
st.caption("A Gaussian mixture model gives each 10-second window a probability of belonging to each phase (soft clustering). "
           "Consecutive windows then give a Markov transition matrix, dwell times and a check that uncertain windows sit at phase changes.")
games = {g: v for g, v in available_games(ddir).items() if v[0] == "sc"}
FEATS = ["hull_area", "length", "width", "def_line", "centroid_x", "poss"]


@st.cache_data(show_spinner="Building windows for every SkillCorner match (first run ~1-2 min)…")
def pooled(ddir):
    out = []
    for g, (_, tp, mp) in games.items():
        d, mj = load_sc_game(tp, mp), json.load(open(mp, encoding="utf-8"))
        for tm in ("home", "away"):
            w, _, _ = build_windows((g, ddir), d, tm, 10)
            if not w.empty:
                out.append(w.assign(match=str(mj["id"]), team=tm))
    return pd.concat(out, ignore_index=True)


df = pooled(ddir)
if df.empty:
    st.warning("Need SkillCorner tracking files in the data folder."); st.stop()
X = StandardScaler().fit_transform(df[FEATS])

bic = {k_: GaussianMixture(k_, covariance_type="full", n_init=2, random_state=0).fit(X).bic(X) for k_ in range(2, 9)}
c1, c2 = st.columns([1, 2])
with c1:
    best = min(bic, key=bic.get)
    k = st.slider("Mixture components (phases)", 2, 8, 4, key="pd_k")
    st.caption(f"Lowest BIC at k = {best}. Lower BIC = better fit after penalising complexity.")
with c2:
    st.line_chart(pd.Series(bic, name="BIC"))

gm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=0).fit(X)
P = gm.predict_proba(X)
df["cluster"] = P.argmax(1)
df["conf"] = P.max(1)
names = name_clusters(df, df.cluster.to_numpy())
df["phase"] = df.cluster.map(names)

# consecutive-window transitions within the same match / team / period
grp = df.groupby(["match", "team", "period"])
df["next"] = grp.cluster.shift(-1)
df["prev"] = grp.cluster.shift(1)
tr = df.dropna(subset=["next"])
M = pd.crosstab(tr.cluster.map(names), tr.next.astype(int).map(names), normalize="index")
M = M.reindex(index=[names[i] for i in range(k)], columns=[names[i] for i in range(k)]).fillna(0)

a, b, c = st.columns(3)
a.metric("Windows", f"{len(df):,}")
b.metric("Mean membership confidence", f"{df.conf.mean():.0%}")
c.metric("Uncertain windows (<60%)", f"{(df.conf < .6).mean():.0%}")

st.subheader("Transition matrix - where does the team go next? (rows = from, 10 s later)")
fig, ax = plt.subplots(figsize=(7.5, 4))
im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(k)); ax.set_yticks(range(k))
ax.set_xticklabels([n.split(":")[0] for n in M.columns]); ax.set_yticklabels(list(M.index), fontsize=8)
for i in range(k):
    for j in range(k):
        ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", color="white" if M.values[i, j] > .5 else "black", fontsize=9)
ax.set_xlabel("next phase")
st.pyplot(fig, clear_figure=True)

L, R = st.columns(2)
with L:
    st.subheader("Dwell time per phase")
    run = (df.cluster != df.groupby(["match", "team", "period"]).cluster.shift()).cumsum()
    dw = df.groupby(run).agg(phase=("phase", "first"), n=("phase", "size")).groupby("phase").n.agg(mean_windows="mean", longest="max")
    dw["mean_seconds"] = dw.mean_windows * 10
    st.dataframe(dw.round(1), width="stretch")
with R:
    st.subheader("Most common phase changes")
    sw = tr[tr.cluster != tr.next]
    top = sw.groupby([sw.cluster.map(names), sw.next.astype(int).map(names)]).size().sort_values(ascending=False).head(8)
    top.index = [f"{a_.split(':')[0]} → {b_.split(':')[0]}" for a_, b_ in top.index]
    st.dataframe(top.rename("count"), width="stretch")

st.subheader("Do uncertain windows sit at phase changes?")
edge = (df.next.notna() & (df.next != df.cluster)) | (df.prev.notna() & (df.prev != df.cluster))
cmp_ = pd.DataFrame({"mean confidence": [df.conf[edge].mean(), df.conf[~edge].mean()], "windows": [int(edge.sum()), int((~edge).sum())]},
                    index=["next to a phase change", "stable windows"])
st.dataframe(cmp_.round(3), width="stretch")
st.caption("If confidence is lower next to changes, the soft memberships are capturing real transitions rather than noise. "
           "Phase names are heuristics from cluster means (possession share and depth).")
