from core import *

ctx = get_context('Data & export')
globals().update(ctx)
st.title('Data & export')
st.dataframe(wdf.round(2), width="stretch")
st.download_button("Download window features (CSV)", wdf.to_csv(index=False), f"{game.replace(' ', '_')}_{team}_windows.csv")
with st.expander("Method notes"):
    st.markdown("""
* Tracking is down-sampled to 5 Hz and cut into fixed windows; each window uses the **mean position** of the 10 outfield players (GK = deepest player, dropped).
* Coordinates are flipped per half so the team always attacks left → right. Windows are used only if exactly 11 players are tracked ≥70 % of the window.
* **Formation** = 1-D K-means on outfield x-positions (k = 3 or 4, best silhouette) → player count per line. It is a *shape* description, not the official line-up.
* **Possession** is approximated as "team of the player nearest the ball" averaged over the window (no event data needed). Windows with share > 55 % = possession, < 45 % = defending.
* **Phases** = K-Means/DBSCAN on hull area, length, width, defensive-line height, centroid depth (+ possession). Names are heuristics from cluster means; always sanity-check against video.
""")
