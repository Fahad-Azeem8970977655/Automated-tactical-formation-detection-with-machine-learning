"""Tactical Analyst - entry point.  Run:  python -m streamlit run app.py
Page names + emoji icons are set here (st.navigation), so the page FILE names stay plain ASCII
and survive zipping / unzipping on Windows."""
import streamlit as st

st.set_page_config(page_title="Tactical Analyst", page_icon="⚽", layout="wide")


def P(path, title, icon, **kw):
    return st.Page(path, title=title, icon=icon, **kw)


pages = {
    "Overview": [P("home.py", "Home", "🏠", default=True)],
    "Shape & phases": [
        P("pages/1_Timeline.py", "Timeline", "📈"), P("pages/2_Phase_Shapes.py", "Phase Shapes", "🧩"),
        P("pages/3_Window_Explorer.py", "Window Explorer", "🔍"), P("pages/20_Phase_Gallery.py", "Phase Gallery", "🖼️"),
        P("pages/10_Phase_Dynamics.py", "Phase Dynamics", "🔄"), P("pages/4_SkillCorner_Phases.py", "SkillCorner Phases", "🛡️"),
    ],
    "Context & outcomes": [
        P("pages/8_Shape_vs_Outcomes.py", "Shape vs Outcomes", "🎯"), P("pages/11_Opponent_and_Context.py", "Opponent & Context", "🆚"),
        P("pages/12_Transitions.py", "Transitions", "⚡"), P("pages/5_Team_Comparison.py", "Team Comparison", "⚖️"),
        P("pages/17_League_Comparison.py", "League Comparison", "🌍"), P("pages/15_Team_Report.py", "Team Report", "📋"),
    ],
    "Players & space": [
        P("pages/18_Tactical_Board.py", "Tactical Board", "🎬"), P("pages/19_Heat_and_Pass_Maps.py", "Heat & Pass Maps", "🔥"),
        P("pages/13_Roles.py", "Roles", "🎭"), P("pages/14_Pitch_Control.py", "Pitch Control", "🗺️"),
    ],
    "Models & data": [
        P("pages/7_ML_Lab.py", "ML Lab", "🤖"), P("pages/9_Formation_Validator.py", "Formation Validator", "📐"),
        P("pages/16_Data_Quality.py", "Data Quality", "🧪"), P("pages/6_Data_Export.py", "Data Export", "📄"),
    ],
}
st.navigation(pages).run()
