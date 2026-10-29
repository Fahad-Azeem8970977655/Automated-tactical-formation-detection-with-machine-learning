"""Advanced analytics on SkillCorner matches: opponent / ball / game-state context, loss-recovery, roles, pitch control."""
import re
from itertools import combinations
from scipy.spatial.distance import cdist, pdist
from core import *

FRAME_HZ = 10


def sc_files(ddir):
    """Yield dict(game, tp, mp, pf, ef, mid) for every SkillCorner match that has tracking + match json."""
    for g, (kind, tp, mp) in available_games(ddir).items():
        if kind != "sc":
            continue
        mid = re.search(r"(\d+)_tracking_extrapolated", os.path.basename(tp)).group(1)
        base = os.path.dirname(tp)
        pf = glob.glob(os.path.join(base, f"*{mid}_phases_of_play.csv"))
        ef = glob.glob(os.path.join(base, f"*{mid}_dynamic_events.csv"))
        yield dict(game=g, tp=tp, mp=mp, mid=mid, pf=pf[0] if pf else None, ef=ef[0] if ef else None)


def flip_mask(xy, period):
    """True for frames where the team's own goal is on the right (needs flipping so it attacks left -> right)."""
    fl = np.zeros(len(period), bool)
    for p in np.unique(period):
        m = period == p
        mx = np.nanmean(xy[m][..., 0], axis=0)
        j = np.nanargmax(np.abs(mx - PL / 2))
        fl[m] = mx[j] > PL / 2
    return fl


def apply_flip(a, fl):
    out = a.copy()
    out[fl, ..., 0] = PL - a[fl, ..., 0]
    out[fl, ..., 1] = PW - a[fl, ..., 1]
    return out


def count_in(frames, lo, hi, w=None):
    o = np.argsort(frames); f = frames[o]
    c = np.concatenate([[0], np.cumsum(np.ones(len(f)) if w is None else np.nan_to_num(w[o]))])
    return c[np.searchsorted(f, hi)] - c[np.searchsorted(f, lo)]


def _shape(out):
    h = ConvexHull(out)
    return dict(hull_area=h.volume, length=np.ptp(out[:, 0]), width=np.ptp(out[:, 1]), def_line=np.sort(out[:, 0])[:4].mean(),
                centroid_x=out[:, 0].mean(), compactness=np.hypot(*(out - out.mean(0)).T).mean())


def _team_windows(d, mj, ev, ph, team, win=10):
    ev = ev.sort_values("frame_start")
    other = "away" if team == "home" else "home"
    tid = mj["home_team"]["id"] if team == "home" else mj["away_team"]["id"]
    start = {p["period"]: p["start_frame"] for p in mj["match_periods"]}
    fl_o, fl_t = flip_mask(d[team], d["period"]), None
    own, opp = apply_flip(d[team], fl_o), d[other]
    fl_p = flip_mask(opp, d["period"])
    opp = apply_flip(opp, fl_p)                       # opponent in ITS OWN attacking frame
    ball = apply_flip(d["ball"][:, None, :], fl_o)[:, 0]   # ball in the team's frame
    hp = home_possession(d)
    tp = hp if team == "home" else 1 - hp
    e = ev[ev.team_id == tid].sort_values("frame_start") if ev is not None else None
    rows = []
    for p in np.unique(d["period"]):
        idx = np.where(d["period"] == p)[0]
        wid = (d["t"][idx] // win).astype(int)
        for w in np.unique(wid):
            ii = idx[wid == w]
            if len(ii) < 0.8 * win * HZ:
                continue
            ko, kp = np.isfinite(own[ii][..., 0]).mean(0) >= .7, np.isfinite(opp[ii][..., 0]).mean(0) >= .7
            if ko.sum() != 11 or kp.sum() != 11:
                continue
            mo, mp_ = np.nanmean(own[ii][:, ko], 0), np.nanmean(opp[ii][:, kp], 0)
            o_out, p_out = np.delete(mo, mo[:, 0].argmin(), 0), np.delete(mp_, mp_[:, 0].argmin(), 0)
            r = _shape(o_out)
            po = _shape(p_out)
            b = np.nanmean(ball[ii], 0)
            dist = np.hypot(*(o_out - b).T) if np.isfinite(b).all() else np.full(10, np.nan)
            s = np.sort(dist)
            lo = start[p] + (d["t"][ii[0]]) * FRAME_HZ
            rows.append(dict(
                period=int(p), minute=d["t"][ii[0]] / 60 + 45 * (p - 1), lo=lo, hi=lo + win * FRAME_HZ, **r,
                poss=float(np.nan_to_num(np.nanmean(tp[ii]), nan=.5)),
                opp_hull=po["hull_area"], opp_width=po["width"], opp_length=po["length"], opp_def_line=po["def_line"],
                centroid_gap=(PL - p_out[:, 0].mean()) - o_out[:, 0].mean(), width_diff=r["width"] - po["width"],
                ball_x=b[0], ball_dist=np.nanmean(dist),
                near5=pdist(o_out[np.argsort(dist)[:5]]).mean() if np.isfinite(dist).all() else np.nan,
                far5=pdist(o_out[np.argsort(dist)[5:]]).mean() if np.isfinite(dist).all() else np.nan))
    w = pd.DataFrame(rows)
    if w.empty:
        return w
    mid_f = ((w.lo + w.hi) / 2).to_numpy()
    # game state from the team's last event at/before the window
    ef = e.frame_start.to_numpy(float); k = np.clip(np.searchsorted(ef, mid_f) - 1, 0, len(ef) - 1)
    w["state"] = e.game_state.to_numpy()[k]
    w["score_diff"] = (e.team_score - e.opponent_team_score).to_numpy()[k]
    eng = e[e.event_type == "on_ball_engagement"].frame_start.to_numpy(float)
    w["pressures"] = count_in(eng, w.lo.to_numpy(), w.hi.to_numpy())
    sh = ev[ev.end_type == "shot"]                       # real shots: possession events that end in a shot
    w["shots_for"] = count_in(sh[sh.team_id == tid].frame_end.to_numpy(float), w.lo.to_numpy(), w.hi.to_numpy())
    w["shot_phases_for"] = count_in(ph[(ph.team_in_possession_id == tid) & ph.team_possession_lead_to_shot.astype(bool)].frame_start.to_numpy(float), w.lo.to_numpy(), w.hi.to_numpy())
    w["shots_against"] = count_in(sh[sh.team_id != tid].frame_end.to_numpy(float), w.lo.to_numpy(), w.hi.to_numpy())
    w["shot_phases_against"] = count_in(ph[(ph.team_in_possession_id != tid) & ph.team_possession_lead_to_shot.astype(bool)].frame_start.to_numpy(float), w.lo.to_numpy(), w.hi.to_numpy())
    return w


@st.cache_data(show_spinner="Building opponent / ball / game-state windows for all matches (first run ~3 min)…")
def context_all(ddir):
    out = []
    for f in sc_files(ddir):
        if not (f["pf"] and f["ef"]):
            continue
        mj, d = json.load(open(f["mp"], encoding="utf-8")), load_sc_game(f["tp"], f["mp"])
        ph = pd.read_csv(f["pf"])
        ev = pd.read_csv(f["ef"], usecols=["frame_start", "frame_end", "team_id", "event_type", "end_type", "game_state", "team_score", "opponent_team_score"], low_memory=False)
        for tm in ("home", "away"):
            w = _team_windows(d, mj, ev, ph, tm)
            if w.empty:
                continue
            side = mj["home_team"] if tm == "home" else mj["away_team"]
            opp = mj["away_team"] if tm == "home" else mj["home_team"]
            out.append(w.assign(match=f["mid"], team=side["short_name"], opponent=opp["short_name"], side=tm))
    return pd.concat(out, ignore_index=True)


# ───────────── loss -> recovery ─────────────
def _frame_length(pos):
    ok = np.isfinite(pos[..., 0])
    x = np.sort(np.where(ok, pos[..., 0], np.inf), axis=1)
    n = ok.sum(1)
    L = x[np.arange(len(x)), np.clip(n - 1, 0, None)] - x[:, 1]
    L[n < 10] = np.nan
    return L


@st.cache_data(show_spinner="Analysing possession losses and shape recovery (first run ~2 min)…")
def losses_all(ddir, horizon=15):
    rows = []
    for f in sc_files(ddir):
        if not (f["pf"] and f["ef"]):
            continue
        mj, d, ph = json.load(open(f["mp"], encoding="utf-8")), load_sc_game(f["tp"], f["mp"]), pd.read_csv(f["pf"])
        ev = pd.read_csv(f["ef"], usecols=["frame_start", "team_id", "event_type"], low_memory=False)
        start = {p["period"]: p["start_frame"] for p in mj["match_periods"]}
        frame = np.array([start[p] + t * FRAME_HZ for p, t in zip(d["period"], d["t"])])
        hp = home_possession(d)
        for tm in ("home", "away"):
            side = mj["home_team"] if tm == "home" else mj["away_team"]
            tid = side["id"]
            L = _frame_length(apply_flip(d[tm], flip_mask(d[tm], d["period"])))
            tp = hp if tm == "home" else 1 - hp
            base = np.nanmedian(L[tp == 0])                       # typical out-of-possession depth
            p_own = ph[ph.team_in_possession_id == tid]
            gains = p_own.frame_start.to_numpy(float)
            eng = ev[(ev.team_id == tid) & (ev.event_type == "on_ball_engagement")].frame_start.to_numpy(float)
            for _, r in p_own[p_own.team_possession_loss_in_phase.astype(bool)].iterrows():
                f0 = r.frame_end
                m = (frame >= f0) & (frame <= f0 + horizon * FRAME_HZ)
                if m.sum() < 5:
                    continue
                Lw, tw = L[m], (frame[m] - f0) / FRAME_HZ
                ok = np.where(Lw <= base)[0]
                nxt = gains[gains > f0]
                rows.append(dict(match=f["mid"], team=side["short_name"], third=r.third_end, phase_before=r.team_in_possession_phase_type,
                                 length_at_loss=np.nanmean(L[(frame >= f0 - 5) & (frame <= f0 + 5)][:3]), baseline=base,
                                 recover_s=tw[ok[0]] if len(ok) else np.nan, recovered=len(ok) > 0,
                                 regain_10s=bool(len(nxt) and (nxt[0] - f0) <= 10 * FRAME_HZ),
                                 presses_5s=count_in(eng, np.array([f0]), np.array([f0 + 5 * FRAME_HZ]))[0]))
    return pd.DataFrame(rows)


# ───────────── pitch control (Voronoi) ─────────────
GX, GY = np.meshgrid(np.linspace(0, PL, 106), np.linspace(0, PW, 69))
GRID = np.c_[GX.ravel(), GY.ravel()]


def voronoi(home, away):
    """Share of pitch closer to each team's nearest player. Returns (control grid of 0/1 for home, home share)."""
    h, a = home[np.isfinite(home[:, 0])], away[np.isfinite(away[:, 0])]
    dh, da = cdist(GRID, h).min(1), cdist(GRID, a).min(1)
    c = (dh < da).astype(float)
    return c.reshape(GX.shape), c.mean()


@st.cache_data(show_spinner="Computing role positions…")
def role_positions(tp, mp, side):
    """Mean position of every starter with / without the ball (team attacks left -> right)."""
    mj, d = json.load(open(mp, encoding="utf-8")), load_sc_game(tp, mp)
    tid = mj[f"{side}_team"]["id"]
    slot = {}
    for p in mj["players"]:
        if p["team_id"] == tid:
            slot[p["id"]] = len(slot)
    pos = apply_flip(d[side], flip_mask(d[side], d["period"]))
    hp = home_possession(d); tpos = hp if side == "home" else 1 - hp
    rows = []
    for p in mj["players"]:
        if p["team_id"] != tid or p["player_role"]["acronym"] == "SUB" or p["start_time"] != "00:00:00":
            continue
        xy = pos[:, slot[p["id"]]]
        r = dict(role=p["player_role"]["acronym"], player=p["short_name"])
        for nm_, m in (("in", tpos == 1), ("out", tpos == 0)):
            v = xy[m]; v = v[np.isfinite(v[:, 0])]
            r[f"x_{nm_}"], r[f"y_{nm_}"] = (v[:, 0].mean(), v[:, 1].mean()) if len(v) > 50 else (np.nan, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)
