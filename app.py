import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# =========================================================
# 1. PAGE CONFIG & UI STYLING
# =========================================================
st.set_page_config(
    page_title="FIFA AI Scout Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {direction: ltr;}
    h1, h2, h3, h4, h5, h6, p, div {text-align: left;}
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton > button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# 2. DATA LOADING & STATE MANAGEMENT
# =========================================================
@st.cache_resource
def load_all_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model_path = os.path.join(current_dir, 'fifa_model_pipeline.pkl')
        features_path = os.path.join(current_dir, 'model_features.pkl')
        db_path = os.path.join(current_dir, 'fifa_players_lite.pkl')

        model = joblib.load(model_path)
        features = joblib.load(features_path)
        db = joblib.load(db_path)
        return model, features, db
    except FileNotFoundError:
        st.error("❌ Error: Essential .pkl files are missing. Please ensure they are uploaded.")
        st.stop()


model_pipeline, model_features, players_db = load_all_data()

# Initialize session states
if 'player1_data' not in st.session_state:
    st.session_state['player1_data'] = None
if 'player1_name' not in st.session_state:
    st.session_state['player1_name'] = ""
if 'run' not in st.session_state:
    st.session_state['run'] = False


def full_reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# =========================================================
# 3. ANALYTICS & RADAR HELPERS
# =========================================================
def get_radar_values(d):
    """Maps raw attributes to the 6 radar categories"""
    pace = d.get('SprintSpeed', 50)
    sho = (d.get('Finishing', 50) + d.get('ShotPower', 50)) / 2
    pas = (d.get('ShortPassing', 50) + d.get('Vision', 50)) / 2
    dri = d.get('Dribbling', 50)
    def_ = (d.get('StandingTackle', 40) + d.get('Interceptions', 40)) / 2
    phy = (d.get('Strength', 50) + d.get('Stamina', 50)) / 2
    return [pace, sho, pas, dri, def_, phy]


def get_db_player_data(name):
    row = players_db[players_db['Name'] == name].head(1)
    if row.empty: return None
    return get_radar_values(row.to_dict('records')[0])


def plot_radar_comparison(labels, v1, v2=None, n1="Current", n2="Comparison"):
    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Player 1
    v1_plot = v1 + v1[:1]
    ax.plot(angles, v1_plot, linewidth=2, color='#1f77b4', label=n1)
    ax.fill(angles, v1_plot, '#1f77b4', alpha=0.1)

    # Player 2 / Comparison
    if v2:
        v2_plot = v2 + v2[:1]
        ax.plot(angles, v2_plot, linewidth=2, color='#d62728', label=n2)
        ax.fill(angles, v2_plot, '#d62728', alpha=0.1)

    plt.xticks(angles[:-1], labels, color='grey', size=11)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    return fig


# =========================================================
# 4. SIDEBAR - SCOUTING FORM (Expanders)
# =========================================================
st.sidebar.title("🛠️ Scouting Form")
if st.sidebar.button("🔄 Reset App"): full_reset()

st.sidebar.markdown("---")
star_name = st.sidebar.selectbox("Benchmark vs Star:", ["None"] + sorted(players_db['Name'].unique().tolist()))

# --- SECTION: PROFILE ---
st.sidebar.subheader("👤 Player Profile")
name_input = st.sidebar.text_input("Full Name", "New Prospect")
age_input = st.sidebar.slider("Age", 15, 45, 22)
pot_input = st.sidebar.slider("Potential", 40, 99, 85)
pos_input = st.sidebar.selectbox("Best Position", ['ST', 'LW', 'RW', 'CAM', 'CM', 'CDM', 'CB', 'LB', 'RB', 'GK'])
foot_input = st.sidebar.radio("Preferred Foot", ["Right", "Left"], horizontal=True)

# --- SECTION: ATTACKING ---
with st.sidebar.expander("⚽ ATTACKING SKILLS", expanded=True):
    in_ball = st.sidebar.slider("Ball Control", 10, 99, 75)
    in_drib = st.sidebar.slider("Dribbling", 10, 99, 75)
    in_fin = st.sidebar.slider("Finishing", 10, 99, 70)
    in_shot = st.sidebar.slider("Shot Power", 10, 99, 70)

# --- SECTION: PHYSICAL ---
with st.sidebar.expander("🏃 PHYSICAL & PACE"):
    in_sprint = st.sidebar.slider("Sprint Speed", 10, 99, 80)
    in_stam = st.sidebar.slider("Stamina", 10, 99, 70)
    in_str = st.sidebar.slider("Strength", 10, 99, 70)
    in_reac = st.sidebar.slider("Reactions", 10, 99, 72)

# --- SECTION: DEFENSE & PASSING ---
with st.sidebar.expander("🛡️ DEFENSE & PASSING"):
    in_pass = st.sidebar.slider("Short Passing", 10, 99, 70)
    in_vis = st.sidebar.slider("Vision", 10, 99, 68)
    in_tack = st.sidebar.slider("Standing Tackle", 10, 99, 50)
    in_int = st.sidebar.slider("Interceptions", 10, 99, 50)

# --- SECTION: ADVANCED ---
with st.sidebar.expander("💎 ADVANCED PARAMETERS"):
    in_comp = st.sidebar.slider("Composure", 10, 99, 75)
    in_long_p = st.sidebar.slider("Long Passing", 10, 99, 65)
    in_long_s = st.sidebar.slider("Long Shots", 10, 99, 60)
    in_agg = st.sidebar.slider("Aggression", 10, 99, 60)
    in_pos_att = st.sidebar.slider("Positioning", 10, 99, 72)

# Consolidate Dictionary
input_dict = {
    'Age': age_input, 'Potential': pot_input, 'Best Position': pos_input, 'Preferred Foot': foot_input,
    'BallControl': in_ball, 'Dribbling': in_drib, 'Finishing': in_fin, 'ShotPower': in_shot,
    'SprintSpeed': in_sprint, 'Stamina': in_stam, 'Strength': in_str, 'Reactions': in_reac,
    'ShortPassing': in_pass, 'Vision': in_vis, 'StandingTackle': in_tack, 'Interceptions': in_int,
    'Composure': in_comp, 'LongPassing': in_long_p, 'LongShots': in_long_s, 'Aggression': in_agg,
    'Positioning': in_pos_att, 'SlidingTackle': in_tack - 5
}

# =========================================================
# 5. MAIN PAGE - LOGIC & DASHBOARD
# =========================================================
st.title("⚽ FIFA AI Scout Pro: Advanced Scouting System")
st.write("Professional Decision Support System for Talent Identification.")

# Comparison Controls
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("💾 SAVE AS PLAYER 1", use_container_width=True):
        st.session_state['player1_data'] = input_dict.copy()
        st.session_state['player1_name'] = name_input
        st.success(f"Player '{name_input}' stored in slot 1.")

with col_btn2:
    if st.button("🚀 EXECUTE SCOUTING ENGINE", use_container_width=True):
        st.session_state['run'] = True

if st.session_state.get('run'):
    st.divider()
    l_col, r_col = st.columns([1, 2])

    with l_col:
        # Valuation Model
        df_x = pd.DataFrame(columns=model_features)
        df_x.loc[0] = 0
        for k, v in input_dict.items():
            if k in df_x.columns: df_x.at[0, k] = v

        raw_pred = model_pipeline.predict(df_x)[0]
        st.metric("Estimated Market Value", f"€{raw_pred / 1e6:.2f}M")

        # AI Insights Calculation
        impact_feats = ['SprintSpeed', 'Finishing', 'Potential', 'ShortPassing', 'Dribbling', 'Reactions']
        boosters = []
        recommendations = []

        for f in impact_feats:
            # Strength check
            val = input_dict[f]
            if val > 75:
                boosters.append((f, f"High proficiency in {f} drives current value."))

            # Improvement potential
            t_up = df_x.copy()
            t_up.at[0, f] += 10
            potential_gain = (model_pipeline.predict(t_up)[0] - raw_pred) / 1e6
            recommendations.append((f, potential_gain))

        st.subheader("💡 AI Scouting Insights")
        top_recs = sorted(recommendations, key=lambda x: x[1], reverse=True)[:3]

        with st.container(border=True):
            st.markdown("**Top Strengths:**")
            for b_name, b_text in boosters[:3]:
                st.write(f"✅ {b_text}")

            st.markdown("**Growth Opportunities:**")
            for r_name, r_gain in top_recs:
                st.write(f"🚀 +10 in **{r_name}** ➔ **+€{r_gain:.1f}M** value")

        # Radar Display
        st.subheader("📊 Tactical Analysis")
        radar_labels = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defense', 'Physical']
        v1 = get_radar_values(input_dict)

        # Comparison logic
        if st.session_state['player1_data']:
            v2 = get_radar_values(st.session_state['player1_data'])
            fig = plot_radar_comparison(radar_labels, v1, v2, name_input, st.session_state['player1_name'])
        elif star_name != "None":
            v2 = get_db_player_data(star_name)
            fig = plot_radar_comparison(radar_labels, v1, v2, name_input, star_name)
        else:
            fig = plot_radar_comparison(radar_labels, v1, None, name_input)

        st.pyplot(fig)

    with r_col:
        st.subheader(f"🔍 Talent Discovery: {name_input}")

        # Content-Based Filtering (CBF) Engine
        pos_pool = players_db[players_db['Best Position'] == pos_input].copy()
        pos_pool['sim_score'] = 100 - (
                abs(pos_pool['Potential'] - pot_input) +
                abs(pos_pool.get('SprintSpeed', 70) - in_sprint) * 0.5 +
                abs(pos_pool.get('ShortPassing', 70) - in_pass) * 0.5
        )

        # Table 1: Younger Prospects (Wonderkids)
        st.markdown("### 🎣 1. Next-Gen Talents")
        st.caption("Players younger than the subject with high tactical similarity.")
        wonderkids = pos_pool[pos_pool['Age'] < age_input].sort_values('sim_score', ascending=False).head(5)
        st.dataframe(wonderkids[['Name', 'Age', 'Value_EUR', 'Potential', 'sim_score']], use_container_width=True)

        # Table 2: General Tactical Matches
        st.markdown("### 🧬 2. Tactical Soulmates")
        st.caption("Closest matches in the global database regardless of age.")
        matches = pos_pool.sort_values('sim_score', ascending=False).head(5)
        st.dataframe(matches[['Name', 'Age', 'Value_EUR', 'Potential', 'sim_score']], use_container_width=True)

        # =========================================================
        # 6. PROFESSIONAL CSV EXPORT (Strict Horizontal Layout)
        # =========================================================
        st.divider()
        st.subheader("📥 Export Comprehensive Report")

        try:
            # A. TARGET PLAYER (Horizontal Row)
            exp_target = pd.DataFrame([input_dict])
            exp_target.insert(0, 'Player Name', name_input)

            # B. MARKET VALUE (Single Cell Block)
            exp_val = pd.DataFrame([{'Metric': 'AI Predicted Market Value', 'Result': f"€{raw_pred:,.0f}"}])

            # C. INSIGHTS SUMMARY
            exp_insights = pd.DataFrame([
                {'Type': 'Strength', 'Observation': boosters[0][1] if len(boosters) > 0 else 'N/A'},
                {'Type': 'Strength', 'Observation': boosters[1][1] if len(boosters) > 1 else 'N/A'},
                {'Type': 'Improvement', 'Observation': f"Improve {top_recs[0][0]} for +€{top_recs[0][1]}M impact"},
                {'Type': 'Improvement', 'Observation': f"Improve {top_recs[1][0]} for +€{top_recs[1][1]}M impact"}
            ])

            # Building the unified CSV string
            csv_final = "--- SECTION 1: TARGET PLAYER ATTRIBUTES ---\n"
            csv_final += exp_target.to_csv(index=False)
            csv_final += "\n--- SECTION 2: FINANCIAL VALUATION ---\n"
            csv_final += exp_val.to_csv(index=False)
            csv_final += "\n--- SECTION 3: AI SCOUTING RECOMMENDATIONS ---\n"
            csv_final += exp_insights.to_csv(index=False)

            # Discovery Tables
            safe_cols = ['Name', 'Age', 'Value_EUR', 'Potential', 'sim_score']
            if 'Club' in players_db.columns: safe_cols.insert(2, 'Club')

            csv_final += "\n--- SECTION 4: NEXT-GEN TALENT DISCOVERY ---\n"
            csv_final += wonderkids[safe_cols].to_csv(index=False)
            csv_final += "\n--- SECTION 5: TACTICAL SOULMATES ---\n"
            csv_final += matches[safe_cols].to_csv(index=False)

            st.download_button(
                label="Download Full Scouting Dossier (CSV)",
                data=csv_final.encode('utf-8-sig'),
                file_name=f"Dossier_{name_input}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Export Generation Failed: {e}")

else:
    # Landing View
    st.write("---")
    st.markdown("### Welcome, Head Scout")
    st.write("Construct a player profile using the panel on the left to begin deep-dive analysis.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Football_in_Bloomington%2C_Indiana%2C_1996.jpg",
             caption="FIFA Analytics Engine v2.0", use_container_width=True)