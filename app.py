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

    /* Styling for the Action Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }

    /* Highlight the Download Button */
    .stDownloadButton > button {
        background-color: #28a745 !important;
        color: white !important;
        border: none;
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
if 'selected_comparison' not in st.session_state:
    st.session_state['selected_comparison'] = None


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
    """
    Draws a radar chart with numbers at the vertices (corners)
    """
    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Offset rotation to put the first axis at the top
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Player 1 (Input)
    v1_plot = v1 + v1[:1]
    ax.plot(angles, v1_plot, linewidth=2, color='#1f77b4', label=n1)
    ax.fill(angles, v1_plot, '#1f77b4', alpha=0.1)

    # Add numbers at vertices for Player 1
    for angle, val in zip(angles[:-1], v1):
        ax.text(angle, val + 8, str(int(val)), ha='center', va='center', fontsize=10, fontweight='bold',
                color='#1f77b4')

    # Player 2 (Comparison)
    if v2:
        v2_plot = v2 + v2[:1]
        ax.plot(angles, v2_plot, linewidth=2, color='#d62728', label=n2)
        ax.fill(angles, v2_plot, '#d62728', alpha=0.1)
        # Optional: Add numbers for player 2 (can be cluttered, usually p1 is enough)
        # for angle, val in zip(angles[:-1], v2):
        #     ax.text(angle, val - 8, str(int(val)), ha='center', va='center', fontsize=9, color='#d62728')

    # Labels for axes
    plt.xticks(angles[:-1], labels, color='black', size=11)

    # Remove radial labels (the 20, 40, 60 circles) to clean up view since we have vertex numbers
    ax.set_yticklabels([])

    # Legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    return fig


# =========================================================
# 4. SIDEBAR - SCOUTING FORM
# =========================================================
st.sidebar.title("🛠️ Scouting Form")

# --- RESTORED COMPARISON DROPDOWN ---
st.sidebar.markdown("### 🏆 Comparison Benchmark")
star_options = ["None"] + sorted(players_db['Name'].unique().tolist())
star_name = st.sidebar.selectbox("Compare with Real Player:", star_options, index=0)
st.sidebar.markdown("---")

# Basic Info
st.sidebar.subheader("👤 Player Profile")
name_input = st.sidebar.text_input("Full Name", "New Prospect")
age_input = st.sidebar.slider("Age", 15, 45, 22)
pot_input = st.sidebar.slider("Potential", 40, 99, 85)
pos_input = st.sidebar.selectbox("Best Position", ['ST', 'LW', 'RW', 'CAM', 'CM', 'CDM', 'CB', 'LB', 'RB', 'GK'])
foot_input = st.sidebar.radio("Preferred Foot", ["Right", "Left"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.caption("👇 Open categories to edit attributes")

# --- SECTION: ATTACKING ---
with st.sidebar.expander("⚽ ATTACKING SKILLS", expanded=False):
    in_ball = st.slider("Ball Control", 10, 99, 75)
    in_drib = st.slider("Dribbling", 10, 99, 75)
    in_fin = st.slider("Finishing", 10, 99, 70)
    in_shot = st.slider("Shot Power", 10, 99, 70)

# --- SECTION: PHYSICAL ---
with st.sidebar.expander("🏃 PHYSICAL & PACE", expanded=False):
    in_sprint = st.slider("Sprint Speed", 10, 99, 80)
    in_stam = st.slider("Stamina", 10, 99, 70)
    in_str = st.slider("Strength", 10, 99, 70)
    in_reac = st.slider("Reactions", 10, 99, 72)

# --- SECTION: DEFENSE & PASSING ---
with st.sidebar.expander("🛡️ DEFENSE & PASSING", expanded=False):
    in_pass = st.slider("Short Passing", 10, 99, 70)
    in_vis = st.slider("Vision", 10, 99, 68)
    in_tack = st.slider("Standing Tackle", 10, 99, 50)
    in_int = st.slider("Interceptions", 10, 99, 50)

# --- SECTION: ADVANCED ---
with st.sidebar.expander("💎 ADVANCED PARAMETERS", expanded=False):
    in_comp = st.slider("Composure", 10, 99, 75)
    in_long_p = st.slider("Long Passing", 10, 99, 65)
    in_long_s = st.slider("Long Shots", 10, 99, 60)
    in_agg = st.slider("Aggression", 10, 99, 60)
    in_pos_att = st.slider("Positioning", 10, 99, 72)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset App"): full_reset()

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
st.title("⚽ FIFA AI Scout Pro")

# --- ACTION AREA ---
action_container = st.container()
with action_container:
    col_act1, col_act2 = st.columns([1, 1])
    with col_act1:
        if st.button("💾 SAVE THIS PLAYER (For Comparison)"):
            st.session_state['player1_data'] = input_dict.copy()
            st.session_state['player1_name'] = name_input
            st.success(f"Player '{name_input}' saved! You can now edit stats to create a second player.")

    with col_act2:
        if st.button("🚀 LAUNCH ANALYSIS", type="primary"):
            st.session_state['run'] = True
            st.session_state['selected_comparison'] = None

if st.session_state.get('run'):
    st.divider()
    l_col, r_col = st.columns([1, 1.5])

    with l_col:
        # --- PREDICTION ---
        df_x = pd.DataFrame(columns=model_features)
        df_x.loc[0] = 0
        for k, v in input_dict.items():
            if k in df_x.columns: df_x.at[0, k] = v

        raw_pred = model_pipeline.predict(df_x)[0]
        st.markdown(f"### 💰 Value: €{raw_pred / 1e6:.2f}M")

        # --- AI INSIGHTS CALCULATION ---
        # 1. Top Contributors (What makes him valuable NOW)
        # Strategy: Reduce each attribute by 20 and see how much value drops
        current_drivers = []
        check_features = ['SprintSpeed', 'Finishing', 'Potential', 'ShortPassing', 'Dribbling', 'Reactions',
                          'BallControl', 'ShotPower']

        for f in check_features:
            t_down = df_x.copy()
            t_down.at[0, f] -= 20  # Simulate removing this skill
            drop = raw_pred - model_pipeline.predict(t_down)[0]
            if drop > 0:
                current_drivers.append((f, drop / 1e6))  # Convert to Millions

        # Sort by impact
        top_drivers = sorted(current_drivers, key=lambda x: x[1], reverse=True)[:3]

        # 2. Future Improvements (What adds value if Improved)
        recommendations = []
        for f in check_features:
            t_up = df_x.copy()
            t_up.at[0, f] += 10  # Simulate improving
            gain = (model_pipeline.predict(t_up)[0] - raw_pred) / 1e6
            if gain > 0:
                recommendations.append((f, gain))

        top_recs = sorted(recommendations, key=lambda x: x[1], reverse=True)[:3]

        # --- DISPLAY INSIGHTS ---
        st.subheader("💡 AI Scout Insights")

        with st.container(border=True):
            st.markdown("**✅ Top Value Contributors (Why he is worth this):**")
            for dr_name, dr_val in top_drivers:
                st.write(f"- **{dr_name}**: Contributes approx **€{dr_val:.1f}M** to current value.")

            st.markdown("---")
            st.markdown("**🚀 Improvement Recommendations (+10 pts):**")
            for r_name, r_gain in top_recs:
                st.write(f"- Improving **{r_name}** adds **€{r_gain:.1f}M** value.")

        # --- RADAR CHART LOGIC ---
        st.subheader("📊 Comparison Radar")
        radar_labels = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defense', 'Physical']
        v1 = get_radar_values(input_dict)

        comp_vals = None
        comp_name = "None"

        # Priority Logic for Comparison
        if st.session_state['selected_comparison']:
            comp_name = st.session_state['selected_comparison']
            comp_vals = get_db_player_data(comp_name)
            st.info(f"Comparing against selected: {comp_name}")

        elif st.session_state['player1_data']:
            comp_name = st.session_state['player1_name']
            comp_vals = get_radar_values(st.session_state['player1_data'])
            st.info(f"Comparing against saved: {comp_name}")

        elif star_name != "None":
            comp_name = star_name
            comp_vals = get_db_player_data(star_name)
            st.info(f"Comparing against star: {comp_name}")

        fig = plot_radar_comparison(radar_labels, v1, comp_vals, name_input, comp_name)
        st.pyplot(fig)

    with r_col:
        st.subheader(f"🔍 Discovery Engine")

        # CBF Engine
        pos_pool = players_db[players_db['Best Position'] == pos_input].copy()
        pos_pool['sim_score'] = 100 - (
                abs(pos_pool['Potential'] - pot_input) +
                abs(pos_pool.get('SprintSpeed', 70) - in_sprint) * 0.5 +
                abs(pos_pool.get('ShortPassing', 70) - in_pass) * 0.5
        )

        display_cols = ['Name', 'Age', 'Value_EUR', 'sim_score']

        # --- TABLE 1: WONDERKIDS ---
        st.markdown("### 🎣 1. Next-Gen Talents")
        wonderkids = pos_pool[pos_pool['Age'] < age_input].sort_values('sim_score', ascending=False).head(5)

        event_w = st.dataframe(
            wonderkids[display_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="wk_table"
        )

        if len(event_w.selection.rows) > 0:
            row_idx = event_w.selection.rows[0]
            st.session_state['selected_comparison'] = wonderkids.iloc[row_idx]['Name']

        # --- TABLE 2: SOULMATES ---
        st.markdown("### 🧬 2. Tactical Soulmates")
        matches = pos_pool.sort_values('sim_score', ascending=False).head(5)

        event_m = st.dataframe(
            matches[display_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="sm_table"
        )

        if len(event_m.selection.rows) > 0:
            row_idx = event_m.selection.rows[0]
            st.session_state['selected_comparison'] = matches.iloc[row_idx]['Name']

        # =========================================================
        # 6. CSV EXPORT
        # =========================================================
        st.divider()
        st.subheader("📥 Export Report")

        try:
            # 1. Tech Cols
            tech_cols_to_export = [
                'SprintSpeed', 'Finishing', 'ShotPower', 'ShortPassing', 'Dribbling',
                'StandingTackle', 'Interceptions', 'Stamina', 'Strength', 'Vision',
                'BallControl', 'LongPassing', 'Aggression', 'Composure'
            ]
            valid_tech = [c for c in tech_cols_to_export if c in players_db.columns]
            full_cols = ['Name', 'Age', 'Value_EUR', 'sim_score'] + valid_tech
            if 'Club' in players_db.columns: full_cols.insert(2, 'Club')

            # 2. Build Insights DataFrame for CSV
            insights_data = []
            for d_name, d_val in top_drivers:
                insights_data.append(
                    {'Type': 'Current Contributor', 'Feature': d_name, 'Impact': f"Contributes €{d_val:.1f}M"})
            for r_name, r_val in top_recs:
                insights_data.append(
                    {'Type': 'Improvement Opportunity', 'Feature': r_name, 'Impact': f"Gain €{r_val:.1f}M if improved"})

            df_insights = pd.DataFrame(insights_data)

            # 3. Construct CSV
            csv_final = "--- SECTION 1: TARGET PLAYER INPUT ---\n"
            target_df = pd.DataFrame([input_dict])
            target_df.insert(0, 'Player Name', name_input)
            csv_final += target_df.to_csv(index=False)

            csv_final += "\n--- SECTION 2: AI PREDICTION ---\n"
            csv_final += f"Predicted Value,€{raw_pred:.0f}\n"

            csv_final += "\n--- SECTION 3: AI DRIVERS & RECOMMENDATIONS ---\n"
            csv_final += df_insights.to_csv(index=False)

            csv_final += "\n--- SECTION 4: WONDERKIDS (FULL STATS) ---\n"
            csv_final += wonderkids[full_cols].to_csv(index=False)

            csv_final += "\n--- SECTION 5: SOULMATES (FULL STATS) ---\n"
            csv_final += matches[full_cols].to_csv(index=False)

            st.download_button(
                label="Download Full CSV (With Insights & Stats)",
                data=csv_final.encode('utf-8-sig'),
                file_name=f"Report_{name_input}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Export Error: {e}")

else:
    st.write("---")
    st.markdown("### Ready to Scout?")
    st.info("👈 Use the Scouting Form on the left to build a player profile.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Football_in_Bloomington%2C_Indiana%2C_1996.jpg",
             use_container_width=True)