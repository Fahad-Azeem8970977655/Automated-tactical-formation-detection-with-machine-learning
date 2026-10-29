"""
Automated Tactical Formation & Defensive Block Clustering  -  Streamlit dashboard

Data
  * Metrica Sports sample games 1-3  (real optical tracking, 25 Hz)  -> shape clustering
  * SkillCorner opendata-master.zip  (read straight from the zip)    -> block-type validation
    (only matches.json, *_match.json and *_phases_of_play.csv are read; nothing is extracted)

Run:  streamlit run app.py
"""
import glob
import io
import re
import json
import os
import warnings
import xml.etree.ElementTree as ET
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PL, PW = 105.0, 68.0          # pitch length / width (m)
FPS = 25
STEP = 5                      # keep every 5th frame -> 5 Hz
HZ = FPS // STEP
PALETTE = ["#4C9BE8", "#F2A541", "#5CC689", "#E8604C", "#B07CE8", "#E8D44C", "#4CE8D8", "#E84C9B"]
NOISE_COL = "#8a8a8a"
LINE_COLS = ["#E8604C", "#F2A541", "#4C9BE8", "#5CC689"]   # back -> front


# ───────────────────────────── data discovery ─────────────────────────────
def find_data_dir():
    for d in ["data", ".", "/mnt/user-data/uploads"]:
        if glob.glob(os.path.join(d, "Sample_Game_*")) or glob.glob(os.path.join(d, "*.zip")):
            return d
    return "data"


def available_games(d):
    games = {}
    for g in (1, 2):
        h = os.path.join(d, f"Sample_Game_{g}_RawTrackingData_Home_Team.csv")
        a = os.path.join(d, f"Sample_Game_{g}_RawTrackingData_Away_Team.csv")
        if os.path.exists(h) and os.path.exists(a):
            games[f"Metrica Game {g}"] = ("csv", h, a)
    t = os.path.join(d, "Sample_Game_3_tracking.txt")
    m = os.path.join(d, "Sample_Game_3_metadata.xml")
    if os.path.exists(t) and os.path.exists(m):
        games["Metrica Game 3"] = ("epts", t, m)
    import re
    for tp in sorted(glob.glob(os.path.join(d, "**", "*_tracking_extrapolated.jsonl"), recursive=True)):
        if os.path.getsize(tp) < 10_000:      # Git-LFS pointer, not real data
            continue
        mid = re.search(r"(\d+)_tracking_extrapolated", os.path.basename(tp)).group(1)
        mp = glob.glob(os.path.join(os.path.dirname(tp), f"*{mid}_match.json"))
        if mp:
            mj = json.load(open(mp[0], encoding="utf-8"))
            games[f"SkillCorner {mid} ({mj['home_team']['short_name']} v {mj['away_team']['short_name']})"] = ("sc", tp, mp[0])
    for tp in sorted(glob.glob(os.path.join(d, "**", "*_tracking.csv"), recursive=True)):      # Bundesliga (DFL / IDSSE) via kloppy
        m = re.search(r"([A-Z0-9]{6})_tracking\.csv$", os.path.basename(tp))
        pp = glob.glob(os.path.join(os.path.dirname(tp), f"*{m.group(1)}_players.csv")) if m else []
        if pp:
            t = pd.read_csv(pp[0], encoding="utf-8").drop_duplicates("ground").set_index("ground").team_name
            games[f"Bundesliga {m.group(1)} ({t.get('home', '?')} v {t.get('away', '?')})"] = ("dfl", tp, pp[0])
    return games


# ───────────────────────────── Metrica loaders ─────────────────────────────
def _finish(period, time_s, home, away, ball):
    """Scale to metres and make time relative to the start of each period."""
    sc = np.array([PL, PW])
    t = time_s.copy()
    for p in np.unique(period):
        m = period == p
        t[m] = time_s[m] - np.nanmin(time_s[m])
    return dict(period=period.astype(int), t=t, home=home * sc, away=away * sc, ball=ball * sc)


def _read_side(path):
    df = pd.read_csv(path, skiprows=3, header=None, low_memory=False).iloc[::STEP].reset_index(drop=True)
    n = (df.shape[1] - 3) // 2 - 1
    xy = df.iloc[:, 3:3 + 2 * n].to_numpy(float).reshape(len(df), n, 2)
    ball = df.iloc[:, 3 + 2 * n:5 + 2 * n].to_numpy(float)
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 2].to_numpy(float), xy, ball


@st.cache_data(show_spinner="Loading Metrica CSV tracking…")
def load_csv_game(home_path, away_path):
    p, t, h, b = _read_side(home_path)
    _, _, a, _ = _read_side(away_path)
    n = min(len(h), len(a))
    return _finish(p[:n], t[:n], h[:n], a[:n], b[:n])


@st.cache_data(show_spinner="Loading Metrica EPTS tracking (Game 3)…")
def load_epts_game(txt_path, xml_path):
    root = ET.parse(xml_path).getroot()
    pl_team = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    ch_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    home_id = next(root.iter("Score")).get("idLocalTeam")
    prm = {n.text: v.text for pp in root.iter("ProviderParameter")
           for n, v in [(pp.find("Name"), pp.find("Value"))] if n is not None and v is not None}
    h1_end = int(prm.get("first_half_end") or 10**9)

    specs = []
    for s in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in s.iter("PlayerChannelRef")]
        pids = [ch_player[r] for r in refs if r in ch_player and r.endswith("_x")]
        teams = [pl_team[p] for p in pids]
        specs.append((int(s.get("startFrame")), int(s.get("endFrame")),
                      [i for i, tm in enumerate(teams) if tm == home_id],
                      [i for i, tm in enumerate(teams) if tm != home_id]))

    def xy_list(s):
        return [[float(v) for v in c.split(",")] for c in s.split(";")]

    fr, hs, as_, bl = [], [], [], []
    with open(txt_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % STEP:
                continue
            a, b, c = line.rstrip().split(":")
            frame = int(a)
            sp = next((s for s in specs if s[0] <= frame <= s[1]), specs[-1])
            pts = np.array(xy_list(b))
            fr.append(frame)
            hs.append(pts[sp[2]])
            as_.append(pts[sp[3]])
            bl.append([float(v) for v in c.split(",")])
    N = max(max(len(x) for x in hs), max(len(x) for x in as_))

    def pad(lst):
        out = np.full((len(lst), N, 2), np.nan)
        for i, x in enumerate(lst):
            out[i, :len(x)] = x
        return out

    fr = np.array(fr)
    period = np.where(fr <= h1_end, 1, 2)
    return _finish(period, fr / FPS, pad(hs), pad(as_), np.array(bl))


@st.cache_data(show_spinner="Loading SkillCorner tracking (10 Hz → 5 Hz)…")
def load_sc_game(track_path, match_path, detected_only=False):
    mj = json.load(open(match_path, encoding="utf-8"))
    hid, aid = mj["home_team"]["id"], mj["away_team"]["id"]
    team_of = {p["id"]: p["team_id"] for p in mj["players"]}
    slot = {hid: {}, aid: {}}
    for pid, tid in team_of.items():
        if tid in slot:
            slot[tid][pid] = len(slot[tid])
    NH, NA = len(slot[hid]), len(slot[aid])
    N = max(NH, NA)
    per, tt, H, A, B, PH = [], [], [], [], [], []
    with open(track_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 2:                                   # 10 Hz -> 5 Hz
                continue
            r = json.loads(line)
            if r["period"] is None or not r["player_data"]:
                continue
            h, a = np.full((N, 2), np.nan), np.full((N, 2), np.nan)
            for p in r["player_data"]:
                if detected_only and not p.get("is_detected"):
                    continue                                    # skip extrapolated (off-camera) positions
                tid = team_of.get(p["player_id"])
                if tid == hid:
                    h[slot[hid][p["player_id"]]] = (p["x"], p["y"])
                elif tid == aid:
                    a[slot[aid][p["player_id"]]] = (p["x"], p["y"])
            b = r["ball_data"]
            per.append(r["period"]); tt.append(r["frame"] / 10)
            H.append(h); A.append(a)
            B.append((np.nan, np.nan) if b["x"] is None else (b["x"], b["y"]))
            g = (r["possession"] or {}).get("group")
            PH.append(1.0 if g == "home team" else 0.0 if g == "away team" else np.nan)
    per, tt = np.array(per), np.array(tt)
    for p in np.unique(per):
        tt[per == p] -= tt[per == p].min()
    off = np.array([PL / 2, PW / 2])                    # SkillCorner origin = pitch centre
    return dict(period=per, t=tt, home=np.array(H) + off, away=np.array(A) + off,
                ball=np.array(B) + off, poss_home=np.array(PH))


@st.cache_data(show_spinner="Loading Bundesliga (DFL) tracking…")
def load_dfl_game(track_path, players_path):
    """kloppy-exported IDSSE match (see download_bundesliga.py) -> same dict structure as the other loaders."""
    df = pd.read_csv(track_path)
    df = df[df.ball_state == "alive"].reset_index(drop=True)          # drop dead-ball frames
    pl = pd.read_csv(players_path)
    ids = {g: [p for p in pl[pl.ground == g].player_id if f"{p}_x" in df.columns] for g in ("home", "away")}
    N = max(len(ids["home"]), len(ids["away"]))

    def arr(g):
        a = np.full((len(df), N, 2), np.nan)
        for i, p in enumerate(ids[g]):
            a[:, i, 0], a[:, i, 1] = df[f"{p}_x"].to_numpy(float), df[f"{p}_y"].to_numpy(float)
        return a * np.array([PL, PW])                                  # kloppy normalised 0-1 -> metres (assumes 105 x 68)

    t = pd.to_timedelta(df.timestamp).dt.total_seconds().to_numpy().copy()
    per = df.period_id.to_numpy().astype(int)
    for p in np.unique(per):
        t[per == p] -= t[per == p].min()
    ball = df[["ball_x", "ball_y"]].to_numpy(float) * np.array([PL, PW])
    return dict(period=per, t=t, home=arr("home"), away=arr("away"), ball=ball)


# ───────────────────────────── core analytics ─────────────────────────────
def orient(xy, period):
    """Flip each period so the team always attacks left -> right (own goal at x=0)."""
    out = xy.copy()
    for p in np.unique(period):
        m = period == p
        mx = np.nanmean(xy[m][..., 0], axis=0)
        j = np.nanargmax(np.abs(mx - PL / 2))      # deepest player = goalkeeper
        if mx[j] > PL / 2:
            out[m, :, 0] = PL - xy[m, :, 0]
            out[m, :, 1] = PW - xy[m, :, 1]
    return out


def home_possession(d):
    """1.0 when nearest player to the ball is a home player, else 0.0 (NaN if ball missing)."""
    bd = lambda xy: np.where(np.isfinite(xy[..., 0]),
                             np.hypot(xy[..., 0] - d["ball"][:, None, 0], xy[..., 1] - d["ball"][:, None, 1]),
                             np.inf).min(1)
    dh, da = bd(d["home"]), bd(d["away"])
    poss = (dh < da).astype(float)
    poss[~np.isfinite(d["ball"][:, 0])] = np.nan
    if "poss_home" in d:                               # provider's own possession label wins
        poss = np.where(np.isfinite(d["poss_home"]), d["poss_home"], poss)
    return poss


def line_formation(xs):
    """1-D K-means on outfield x-positions (k=3 or 4, best silhouette). Returns (string, line index per player)."""
    x = np.asarray(xs).reshape(-1, 1)
    best = (-2, None)
    for k in (3, 4):
        km = KMeans(k, n_init=5, random_state=0).fit(x)
        s = silhouette_score(x, km.labels_)
        if s > best[0]:
            best = (s, km)
    km = best[1]
    order = np.argsort(km.cluster_centers_.ravel())
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    lines = rank[km.labels_]
    return "-".join(str(int((lines == i).sum())) for i in range(len(order))), lines


@st.cache_data(show_spinner="Building tactical windows…")
def build_windows(data_key, _d, team, win_s, min_cover=0.7):
    d = _d
    pos = orient(d[team], d["period"])
    hp = home_possession(d)
    tp = hp if team == "home" else 1 - hp
    rows, shapes, lines_all = [], [], []
    for p in np.unique(d["period"]):
        idx = np.where(d["period"] == p)[0]
        wid = (d["t"][idx] // win_s).astype(int)
        for w in np.unique(wid):
            ii = idx[wid == w]
            if len(ii) < 0.8 * win_s * HZ:
                continue
            seg = pos[ii]
            keep = np.isfinite(seg[..., 0]).mean(0) >= min_cover
            if keep.sum() != 11:
                continue
            mp = np.nanmean(seg[:, keep], axis=0)
            out = np.delete(mp, np.argmin(mp[:, 0]), axis=0)          # drop GK
            hull = ConvexHull(out)
            form, lines = line_formation(out[:, 0])
            c = out.mean(0)
            rows.append(dict(
                period=int(p), minute=d["t"][ii[0]] / 60 + 45 * (p - 1),
                hull_area=hull.volume, length=np.ptp(out[:, 0]), width=np.ptp(out[:, 1]),
                def_line=np.sort(out[:, 0])[:4].mean(), centroid_x=c[0], centroid_y=c[1],
                compactness=np.hypot(*(out - c).T).mean(),
                poss=float(np.nan_to_num(np.nanmean(tp[ii]), nan=0.5)), formation=form))
            shapes.append(out)
            lines_all.append(lines)
    return pd.DataFrame(rows), np.array(shapes), np.array(lines_all)


def name_clusters(df, labels):
    names, depth = {}, lambda x: "low" if x < 40 else ("mid" if x < 52 else "high")
    for c in sorted(set(labels)):
        if c == -1:
            names[c] = "Transition / noise"
            continue
        m = df[labels == c]
        phase = "Possession / build-up" if m.poss.mean() > 0.55 else ("Defensive block" if m.poss.mean() < 0.45 else "Mixed / transition")
        names[c] = f"C{c}: {phase} ({depth(m.centroid_x.mean())})"
    return names


# ───────────────────────────── plotting ─────────────────────────────
def draw_pitch(ax, bg="#1e5a36", ln="#e8e8e8"):
    ax.set_facecolor(bg)
    ax.add_patch(Rectangle((-2, -2), PL + 4, PW + 8, fc=bg, ec="none", zorder=0))     # visible even when the axes are switched off
    kw = dict(fill=False, ec=ln, lw=1)
    ax.add_patch(Rectangle((0, 0), PL, PW, **kw))
    ax.plot([PL / 2] * 2, [0, PW], c=ln, lw=1)
    ax.add_patch(Circle((PL / 2, PW / 2), 9.15, **kw))
    for x0, s in ((0, 1), (PL, -1)):
        ax.add_patch(Rectangle((x0 if s > 0 else PL - 16.5, PW / 2 - 20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((x0 if s > 0 else PL - 5.5, PW / 2 - 9.16), 5.5, 18.32, **kw))
    ax.annotate("", xy=(PL * 0.62, PW + 3), xytext=(PL * 0.38, PW + 3),
                arrowprops=dict(arrowstyle="->", color="#999"))
    ax.set_xlim(-2, PL + 2); ax.set_ylim(-2, PW + 6); ax.set_aspect("equal"); ax.axis("off")


def hull_poly(pts, **kw):
    return Polygon(pts[ConvexHull(pts).vertices], closed=True, **kw)


def pitch_shape(ax, pts, lines=None, color="#4C9BE8", title=None):
    draw_pitch(ax)
    ax.add_patch(hull_poly(pts, fc=color, ec="white", alpha=0.25, lw=1.5))
    cols = [LINE_COLS[l % 4] for l in lines] if lines is not None else color
    ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=120, ec="white", zorder=3)
    if title:
        ax.set_title(title, fontsize=9, color="#ddd")


# ───────────────────────────── SkillCorner (from zip) ─────────────────────────────
def find_zip(d):
    z = glob.glob(os.path.join(d, "opendata*.zip")) or glob.glob(os.path.join(d, "*.zip"))
    return z[0] if z else None


@st.cache_data(show_spinner="Reading SkillCorner phases from zip…")
def sc_index(zpath):
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        pref = next(n for n in names if n.endswith("data/matches.json"))[:-len("data/matches.json")]
        ms = json.loads(z.read(pref + "data/matches.json"))
        ok = {n for n in names if n.endswith("_phases_of_play.csv")}
        info = []
        for m in ms:
            mid = m["id"]
            if f"{pref}data/matches/{mid}/{mid}_phases_of_play.csv" in ok:
                info.append(dict(id=mid, label=f"{m['home_team']['short_name']} vs {m['away_team']['short_name']} ({m['date_time'][:10]})"))
        lfs = {n: z.getinfo(n).file_size for n in names if n.endswith("_tracking_extrapolated.jsonl")}
    return pref, info, lfs


@st.cache_data(show_spinner=False)
def sc_match(zpath, pref, mid):
    with zipfile.ZipFile(zpath) as z:
        ph = pd.read_csv(io.BytesIO(z.read(f"{pref}data/matches/{mid}/{mid}_phases_of_play.csv")))
        mj = json.loads(z.read(f"{pref}data/matches/{mid}/{mid}_match.json"))
    return ph, {mj["home_team"]["id"]: mj["home_team"]["short_name"], mj["away_team"]["id"]: mj["away_team"]["short_name"]}


def sc_team_frame(ph, tid):
    def mk(df, state, pre):
        return pd.DataFrame(dict(
            state=state, phase=df[f"{pre}_phase_type"],
            width=(df[f"{pre}_width_start"] + df[f"{pre}_width_end"]) / 2,
            length=(df[f"{pre}_length_start"] + df[f"{pre}_length_end"]) / 2,
            duration=df.duration, minute=(df.period - 1) * 45 + df.minute_start))
    i = ph[ph.team_in_possession_id == tid]
    o = ph[ph.team_in_possession_id != tid]
    return pd.concat([mk(i, "In possession", "team_in_possession"), mk(o, "Out of possession", "team_out_of_possession")]).dropna()




def get_context(title="Tactical Analyst", cluster=True):
    """Shared sidebar + windowing + clustering. Every page calls this so settings persist across pages."""
    try:
        st.set_page_config(page_title=title, page_icon="⚽", layout="wide")
    except Exception:
        pass                                     # already set by app.py (st.navigation)
    if not cluster:
        with st.sidebar:
            ddir = st.text_input("Data folder", find_data_dir(), key="ddir")
        return dict(ddir=ddir)
    with st.sidebar:
        st.header("Data")
        ddir = st.text_input("Data folder", find_data_dir(), key="ddir", help="Folder holding the Metrica files and opendata-master.zip")
        games = available_games(ddir)
        if not games:
            st.error("No Metrica files found in this folder.")
            st.stop()
        game = st.selectbox("Match (tracking)", list(games), key="game")
        team = st.radio("Team", ["home", "away"], horizontal=True, key="team")
        st.header("Windows")
        win_s = st.slider("Window length (s)", 5, 30, 10, 5, key="win")
        pfilter = st.selectbox("Possession filter", ["All", "In possession", "Out of possession"],
                               help="Shapes differ a lot between phases - clustering them separately is often cleaner.", key="pf")
        st.header("Clustering")
        method = st.radio("Algorithm", ["K-Means", "DBSCAN"], horizontal=True, key="algo")
        feats = st.multiselect("Features", ["hull_area", "length", "width", "def_line", "centroid_x", "compactness", "poss"],
                               default=["hull_area", "length", "width", "def_line", "centroid_x", "poss"], key="feats")
        if method == "K-Means":
            auto_k = st.checkbox("Auto-pick k (silhouette)", False, key="auto")
            k = st.slider("k (phases)", 2, 8, 4, disabled=auto_k, key="k")
        else:
            eps = st.slider("eps", 0.2, 3.0, 0.9, 0.05, key="eps")
            min_s = st.slider("min_samples", 3, 30, 8, key="mins")

    kind, p1, p2 = games[game]
    data = {"csv": load_csv_game, "epts": load_epts_game, "sc": load_sc_game, "dfl": load_dfl_game}[kind](p1, p2)
    wdf, shapes, lines = build_windows((game, ddir), data, team, win_s)
    if wdf.empty:
        st.warning("No valid windows (need exactly 11 tracked players of the team).")
        st.stop()

    # possession filter + clustering
    mask = np.ones(len(wdf), bool)
    if pfilter == "In possession":
        mask = (wdf.poss > 0.5).to_numpy()
    elif pfilter == "Out of possession":
        mask = (wdf.poss <= 0.5).to_numpy()
    wdf, shapes, lines = wdf[mask].reset_index(drop=True), shapes[mask], lines[mask]
    if len(wdf) < 12 or len(feats) < 2:
        st.warning("Not enough windows / features to cluster.")
        st.stop()

    X = StandardScaler().fit_transform(wdf[feats])
    sil_tbl = None
    if method == "K-Means":
        if auto_k:
            sil_tbl = {kk: silhouette_score(X, KMeans(kk, n_init=10, random_state=0).fit_predict(X)) for kk in range(2, 9)}
            k = max(sil_tbl, key=sil_tbl.get)
        labels = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)

    names = name_clusters(wdf, labels)
    wdf["cluster"] = labels
    wdf["phase"] = [names[l] for l in labels]
    uniq = sorted(set(labels))
    col = {l: (NOISE_COL if l == -1 else PALETTE[i % len(PALETTE)]) for i, l in enumerate(uniq)}


    return {k_: v_ for k_, v_ in locals().items() if not k_.startswith("_")}
