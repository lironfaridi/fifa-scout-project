import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# 1. PAGE CONFIG & UI STYLING
# =========================================================
st.set_page_config(
    page_title="FIFA AI Scout Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS THEME UPGRADE ---
# הוספנו עיצוב לכרטיסיות המדדים (Metrics) כדי שייראו כמו דאשבורד מקצועי
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

    /* Custom Styling for Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #f4f6f9;
        border: 1px solid #e1e4e8;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 4px 4px 12px rgba(0,0,0,0.1);
    }

    /* Dark Mode Support for Metrics */
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background-color: #1e1e1e;
            border: 1px solid #333;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }
    }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# 2. DATA LOADING & STATE MANAGEMENT (USING SMART LITE)
# =========================================================
@st.cache_resource
def load_all_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model_path = os.path.join(current_dir, 'fifa_model_pipeline.pkl')
        features_path = os.path.join(current_dir, 'model_features.pkl')
        # LITE DB
        db_path = os.path.join(current_dir, 'fifa_players_lite.pkl')
        scaler_path = os.path.join(current_dir, 'cbf_scaler.pkl')

        model = joblib.load(model_path)
        features = joblib.load(features_path)
        db = joblib.load(db_path)
        cbf_scaler = joblib.load(scaler_path)

        # --- DATA ENGINEERING TRICK: Build the matrix in memory from the Lite DB ---
        meta_cols = ['Name', 'Age', 'Value_EUR', 'Best Position', 'Potential', 'Name_norm', 'Club', 'sim_score']
        cbf_features = [c for c in db.columns if c not in meta_cols]

        raw_matrix = db[cbf_features].values
        cbf_matrix_scaled_memory = cbf_scaler.transform(raw_matrix)
        # ----------------------------------------------------------

        return model, features, db, cbf_matrix_scaled_memory, cbf_scaler, cbf_features
    except FileNotFoundError as e:
        st.error(f"❌ Error: Essential .pkl files are missing. Please ensure they are uploaded. {e}")
        st.stop()


model_pipeline, model_features, players_db, cbf_matrix_scaled, cbf_scaler, cbf_features_list = load_all_data()

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


# Callback to safely clear table selection when dropdown changes
def on_dropdown_change():
    if 'wk_table' in st.session_state:
        del st.session_state['wk_table']
    if 'sm_table' in st.session_state:
        del st.session_state['sm_table']
    st.session_state['selected_comparison'] = None


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

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    v1_plot = v1 + v1[:1]
    ax.plot(angles, v1_plot, linewidth=2, color='#1f77b4', label=n1)
    ax.fill(angles, v1_plot, '#1f77b4', alpha=0.1)

    for angle, val in zip(angles[:-1], v1):
        ax.text(angle, val + 12, str(int(val)), ha='center', va='center', fontsize=10, fontweight='bold',
                color='#1f77b4')

    if v2:
        v2_plot = v2 + v2[:1]
        ax.plot(angles, v2_plot, linewidth=2, color='#d62728', label=n2)
        ax.fill(angles, v2_plot, '#d62728', alpha=0.1)

    plt.xticks(angles[:-1], labels, color='black', size=11)
    ax.set_yticklabels([])
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    return fig


# =========================================================
# 4. SIDEBAR - SCOUTING FORM
# =========================================================
st.sidebar.title("🛠️ Scouting Form")

# --- COMPARISON DROPDOWN ---
st.sidebar.markdown("### 🏆 Comparison Benchmark")
star_options = ["None"] + sorted(players_db['Name'].unique().tolist())
star_name = st.sidebar.selectbox("Compare with Real Player:", star_options, index=0, on_change=on_dropdown_change,
                                 help="Select a player from the database to compare against your custom profile on the radar chart.")
st.sidebar.markdown("---")

# Basic Info
st.sidebar.subheader("👤 Player Profile")
name_input = st.sidebar.text_input("Full Name", "New Prospect")
age_input = st.sidebar.slider("Age", 15, 45, 22)
wk_max_age = st.sidebar.slider("Max Age for Wonderkids", 16, 24, 22,
                               help="Set the maximum age limit for the High-Potential Prospects discovery table.")
pot_input = st.sidebar.slider("Potential", 40, 99, 85,
                              help="The maximum overall rating the player is projected to reach.")
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
    in_comp = st.slider("Composure", 10, 99, 75, help="How well the player performs under pressure.")
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
            with st.spinner('Running AI Scouting Engine & Valuations...'):
                time.sleep(0.5)
            st.toast('Analysis Complete! 📊')

            st.session_state['run'] = True
            st.session_state['selected_comparison'] = None
            if 'wk_table' in st.session_state: del st.session_state['wk_table']
            if 'sm_table' in st.session_state: del st.session_state['sm_table']

if st.session_state.get('run'):
    run_start_time = time.time()  # <-- NEW: Start the execution timer
    st.divider()

    # -------------------------------------------------------------
    # REAL AI PRE-CALCULATION LOGIC (Cosine Similarity)
    # -------------------------------------------------------------

    # Build the custom player vector precisely matching CBF features
    user_cbf_df = pd.DataFrame(columns=cbf_features_list)

    # *** CRITICAL FIX: Fill missing attributes with DB Mean ***
    user_cbf_df.loc[0] = players_db[cbf_features_list].mean()

    for k, v in input_dict.items():
        if k in user_cbf_df.columns:
            user_cbf_df.at[0, k] = v

    # Scale and Calculate Similarity
    try:
        user_vec_scaled = cbf_scaler.transform(user_cbf_df.values)
        sims = cosine_similarity(user_vec_scaled, cbf_matrix_scaled)[0]
        players_db['sim_score'] = sims * 100
    except Exception as e:
        st.error(f"⚠️ CBF Engine Error: {e}")
        players_db['sim_score'] = 0

    # Filter by position
    pos_pool = players_db[players_db['Best Position'] == pos_input].copy()

    # Generate Tables
    wonderkids = pos_pool[pos_pool['Age'] <= wk_max_age].sort_values('sim_score', ascending=False).head(5)
    matches = pos_pool.sort_values('sim_score', ascending=False).head(5)

    comp_vals = None
    comp_name = "None"
    comp_source_msg = ""
    sel_player_name = None

    if 'wk_table' in st.session_state and st.session_state['wk_table'].get('selection') and \
            st.session_state['wk_table']['selection'].get('rows'):
        idx = st.session_state['wk_table']['selection']['rows'][0]
        sel_player_name = wonderkids.iloc[idx]['Name']

    elif 'sm_table' in st.session_state and st.session_state['sm_table'].get('selection') and \
            st.session_state['sm_table']['selection'].get('rows'):
        idx = st.session_state['sm_table']['selection']['rows'][0]
        sel_player_name = matches.iloc[idx]['Name']

    if sel_player_name:
        comp_name = sel_player_name
        comp_vals = get_db_player_data(sel_player_name)
        comp_source_msg = f"Comparing against selected: {comp_name}"

    elif st.session_state['player1_data']:
        comp_name = st.session_state['player1_name']
        comp_vals = get_radar_values(st.session_state['player1_data'])
        comp_source_msg = f"Comparing against saved: {comp_name}"

    elif star_name != "None":
        comp_name = star_name
        comp_vals = get_db_player_data(star_name)
        comp_source_msg = f"Comparing against star: {comp_name}"

    # -------------------------------------------------------------
    # UI LAYOUT
    # -------------------------------------------------------------
    l_col, r_col = st.columns([1, 1.5])

    with l_col:
        timer_placeholder = st.empty()  # <-- NEW: Placeholder for the timer at the top

        # --- PREDICTION (UPGRADED TO METRIC CARD) ---
        df_x = pd.DataFrame(columns=model_features)
        df_x.loc[0] = 0
        for k, v in input_dict.items():
            if k in df_x.columns: df_x.at[0, k] = v

        raw_pred = model_pipeline.predict(df_x)[0]
        raw_pred = max(0.0, float(raw_pred))  # FIX: Ensure predicted value is never negative

        st.subheader("🎯 Valuation Overview")
        st.metric(label=f"Predicted Market Value for {name_input}", value=f"€{raw_pred / 1e6:.2f}M")

        # --- AI INSIGHTS ---
        current_drivers = []
        check_features = ['SprintSpeed', 'Finishing', 'Potential', 'ShortPassing', 'Dribbling', 'Reactions',
                          'BallControl', 'ShotPower']

        for f in check_features:
            t_down = df_x.copy()
            t_down.at[0, f] -= 20
            pred_down = max(0.0, float(model_pipeline.predict(t_down)[0]))  # FIX: Clamp simulated down-value
            drop = raw_pred - pred_down
            if drop > 0: current_drivers.append((f, drop / 1e6))

        top_drivers = sorted(current_drivers, key=lambda x: x[1], reverse=True)[:3]

        recommendations = []
        for f in check_features:
            t_up = df_x.copy()
            t_up.at[0, f] += 10
            pred_up = max(0.0, float(model_pipeline.predict(t_up)[0]))  # FIX: Clamp simulated up-value
            gain = (pred_up - raw_pred) / 1e6
            if gain > 0: recommendations.append((f, gain))

        top_recs = sorted(recommendations, key=lambda x: x[1], reverse=True)[:3]

        st.subheader("💡 AI Scout Insights")
        with st.container(border=True):
            st.markdown("**✅ Top Value Contributors:**")
            col_d1, col_d2, col_d3 = st.columns(3)
            if len(top_drivers) > 0: col_d1.metric(top_drivers[0][0], f"~€{top_drivers[0][1]:.1f}M")
            if len(top_drivers) > 1: col_d2.metric(top_drivers[1][0], f"~€{top_drivers[1][1]:.1f}M")
            if len(top_drivers) > 2: col_d3.metric(top_drivers[2][0], f"~€{top_drivers[2][1]:.1f}M")

            st.markdown("---")
            st.markdown("**🚀 Growth Opportunities (+10 pts):**")
            col_r1, col_r2, col_r3 = st.columns(3)
            if len(top_recs) > 0: col_r1.metric(top_recs[0][0], "", delta=f"€{top_recs[0][1]:.1f}M")
            if len(top_recs) > 1: col_r2.metric(top_recs[1][0], "", delta=f"€{top_recs[1][1]:.1f}M")
            if len(top_recs) > 2: col_r3.metric(top_recs[2][0], "", delta=f"€{top_recs[2][1]:.1f}M")

        # --- WHAT-IF SIMULATOR (NEW FEATURE) ---
        st.markdown("---")
        st.subheader("🎛️ Interactive What-If Simulator")
        st.caption("Test how changing a specific attribute impacts the player's market value in real-time.")

        sim_col1, sim_col2, sim_col3 = st.columns([1.2, 1.2, 1])
        with sim_col1:
            sim_feature = st.selectbox("Select Attribute to Modify", check_features)
        with sim_col2:
            sim_change = st.slider(f"Change in {sim_feature}", min_value=-20, max_value=20, value=5, step=1)

        # Calculate simulation dynamically
        sim_df = df_x.copy()
        # Keep the simulated value bounded between 10 and 99
        sim_df.at[0, sim_feature] = max(10, min(99, sim_df.at[0, sim_feature] + sim_change))
        sim_pred = max(0.0, float(model_pipeline.predict(sim_df)[0]))
        sim_diff = sim_pred - raw_pred

        with sim_col3:
            st.metric(label="Simulated Value",
                      value=f"€{sim_pred / 1e6:.2f}M",
                      delta=f"€{sim_diff / 1e6:.2f}M" if sim_diff != 0 else "No Change")

        # --- RADAR CHART ---
        st.markdown("---")
        st.subheader("📊 Comparison Radar")
        if comp_source_msg:
            if comp_name in players_db['Name'].values:
                real_val = players_db[players_db['Name'] == comp_name]['Value_EUR'].values[0]
                st.info(f"{comp_source_msg} | **Real Market Value: €{real_val / 1e6:.1f}M**")
            else:
                st.info(comp_source_msg)

        radar_labels = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defense', 'Physical']
        v1 = get_radar_values(input_dict)

        fig = plot_radar_comparison(radar_labels, v1, comp_vals, name_input, comp_name)
        st.pyplot(fig)

    with r_col:
        st.subheader(f"🔍 Discovery Engine")

        display_cols = ['Name', 'Age', 'Value_EUR', 'sim_score']

        # --- TABLE 1: WONDERKIDS ---
        st.markdown(f"### 🎣 1. High-Potential Prospects (U{wk_max_age})")

        if not wonderkids.empty:
            wk_styled = wonderkids[display_cols].style.background_gradient(
                subset=['sim_score'], cmap='RdYlGn', vmin=70, vmax=100
            ).format({'Value_EUR': '€{:,.0f}', 'sim_score': '{:.1f}'})
        else:
            wk_styled = wonderkids[display_cols]

        st.dataframe(
            wk_styled,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="wk_table"
        )

        # --- TABLE 2: SOULMATES ---
        st.markdown("### 🧬 2. Closest Tactical Profiles")

        if not matches.empty:
            sm_styled = matches[display_cols].style.background_gradient(
                subset=['sim_score'], cmap='RdYlGn', vmin=70, vmax=100
            ).format({'Value_EUR': '€{:,.0f}', 'sim_score': '{:.1f}'})
        else:
            sm_styled = matches[display_cols]

        st.dataframe(
            sm_styled,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="sm_table"
        )

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

            # 2. Build Insights DataFrame
            insights_data = []
            for d_name, d_val in top_drivers:
                insights_data.append({'Type': 'Current Driver', 'Feature': d_name, 'Impact': f"Adds €{d_val:.1f}M"})
            for r_name, r_val in top_recs:
                insights_data.append({'Type': 'Recommendation', 'Feature': r_name, 'Impact': f"Gain €{r_val:.1f}M"})

            df_insights = pd.DataFrame(insights_data)

            # 3. Construct CSV
            csv_final = "--- SECTION 1: TARGET PLAYER INPUT ---\n"
            target_df = pd.DataFrame([input_dict])
            target_df.insert(0, 'Player Name', name_input)
            csv_final += target_df.to_csv(index=False)

            csv_final += "\n--- SECTION 2: AI PREDICTION ---\n"
            csv_final += f"Predicted Value,€{raw_pred:.0f}\n"

            csv_final += "\n--- SECTION 3: AI INSIGHTS & RECOMMENDATIONS ---\n"
            csv_final += df_insights.to_csv(index=False)

            csv_final += "\n--- SECTION 4: HIGH-POTENTIAL PROSPECTS (FULL STATS) ---\n"
            csv_final += wonderkids[full_cols].to_csv(index=False)

            csv_final += "\n--- SECTION 5: CLOSEST TACTICAL PROFILES (FULL STATS) ---\n"
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

    # --- END OF RUN: STOP TIMER AND DISPLAY ---
    run_duration = time.time() - run_start_time
    timer_placeholder.markdown(
        f"<div style='text-align: left; color: #6c757d; font-size: 13px; margin-bottom: -15px;'>"
        f"⏱️ <b>Latency:</b> {run_duration:.3f}s</div>",
        unsafe_allow_html=True
    )

else:
    st.write("---")
    st.markdown("### Ready to Scout?")
    st.info("👈 Use the Scouting Form on the left to build a player profile.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Football_in_Bloomington%2C_Indiana%2C_1996.jpg",
             use_container_width=True)