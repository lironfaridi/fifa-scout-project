import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# =========================================================
# 1. PAGE CONFIG & UI STYLING (English)
# =========================================================
st.set_page_config(
    page_title="FIFA AI Scout Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard LTR styling for English
st.markdown("""
    <style>
    .main {direction: ltr;}
    h1, h2, h3, h4, h5, h6, p, div {text-align: left;}
    /* Reset Button Styling */
    div.stButton > button:first-child {
        background-color: #f0f2f6;
        color: black;
        border: 1px solid #d0d0d0;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# 2. DATA LOADING & RESET
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
        st.error("❌ Error: Model files (.pkl) are missing in the repository.")
        st.stop()


model_pipeline, model_features, players_db = load_all_data()


def reset_values():
    for key in list(st.session_state.keys()):
        if key.startswith('key_'):
            del st.session_state[key]
    st.session_state['run'] = False


# =========================================================
# 3. HELPER FUNCTIONS (Radar & Stats)
# =========================================================
def get_player_stats_from_db(player_name):
    row = players_db[players_db['Name'] == player_name].head(1)
    if row.empty: return None
    try:
        pace = row.get('SprintSpeed', pd.Series([50])).iloc[0]
        sho = (row.get('Finishing', pd.Series([50])).iloc[0] + row.get('ShotPower', pd.Series([50])).iloc[0]) / 2
        pas = (row.get('ShortPassing', pd.Series([50])).iloc[0] + row.get('Vision', pd.Series([50])).iloc[0]) / 2
        dri = row.get('Dribbling', pd.Series([50])).iloc[0]
        def_ = (row.get('StandingTackle', pd.Series([50])).iloc[0] + row.get('Interceptions', pd.Series([50])).iloc[
            0]) / 2
        phy = (row.get('Strength', pd.Series([50])).iloc[0] + row.get('Stamina', pd.Series([50])).iloc[0]) / 2
        return [pace, sho, pas, dri, def_, phy]
    except:
        return [50, 50, 50, 50, 50, 50]


def create_radar_comparison(categories, user_vals, comp_vals=None, user_name="My Player", comp_name="Comparison"):
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
    plt.ylim(0, 110)
    # User Player
    vals = user_vals + user_vals[:1]
    ax.plot(angles, vals, linewidth=2, linestyle='solid', label=user_name, color='#1f77b4')
    ax.fill(angles, vals, '#1f77b4', alpha=0.1)
    for ang, val in zip(angles[:-1], user_vals):
        ax.text(ang, val + 12, str(int(val)), ha='center', va='center', fontsize=9, fontweight='bold', color='#1f77b4')
    # Comparison Player
    if comp_vals:
        c_vals = comp_vals + comp_vals[:1]
        ax.plot(angles, c_vals, linewidth=2, linestyle='solid', label=comp_name, color='#d62728')
        ax.fill(angles, c_vals, '#d62728', alpha=0.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Player Comparison Analysis", size=14, y=1.1)
    return fig


# =========================================================
# 4. SIDEBAR - ORGANIZED INPUTS
# =========================================================
st.sidebar.title("🛠️ Player Attributes")

if st.sidebar.button("🔄 Reset All Data", on_click=reset_values):
    st.rerun()

st.sidebar.markdown("### 🔍 Reference Star")
all_names = ["None"] + sorted(players_db['Name'].unique().tolist())
ref_player = st.sidebar.selectbox("Compare with existing star:", all_names, key='key_ref')

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 General Info")
input_name = st.sidebar.text_input("Player Name", "New Prospect")
input_age = st.sidebar.slider("Age", 15, 45, 24, key='key_age')
input_potential = st.sidebar.slider("Potential", 40, 99, 85, key='key_potential')
input_position = st.sidebar.selectbox("Best Position",
                                      ['ST', 'CF', 'LW', 'RW', 'CAM', 'CM', 'CDM', 'CB', 'LB', 'RB', 'GK'],
                                      key='key_pos')
input_foot = st.sidebar.radio("Preferred Foot", ["Right", "Left"], horizontal=True)

st.sidebar.markdown("### ⚽ Core Skills")
in_ball = st.sidebar.slider("Ball Control", 10, 99, 75, key='key_ball')
in_drib = st.sidebar.slider("Dribbling", 10, 99, 75, key='key_drib')
in_sprint = st.sidebar.slider("Sprint Speed", 10, 99, 80, key='key_sprint')
in_fin = st.sidebar.slider("Finishing", 10, 99, 70, key='key_fin')

# ADVANCED STATS EXPANDER - To add more depth without clutter
with st.sidebar.expander("🚀 Advanced Technical Stats"):
    in_pass = st.sidebar.slider("Short Passing", 10, 99, 72, key='key_pass')
    in_vis = st.sidebar.slider("Vision", 10, 99, 68, key='key_vis')
    in_shot = st.sidebar.slider("Shot Power", 10, 99, 75, key='key_shot')
    in_long = st.sidebar.slider("Long Shots", 10, 99, 65, key='key_long')
    in_comp = st.sidebar.slider("Composure", 10, 99, 70, key='key_comp')
    in_reac = st.sidebar.slider("Reactions", 10, 99, 70, key='key_reac')
    in_pos_att = st.sidebar.slider("Att. Positioning", 10, 99, 70, key='key_pos_att')
    in_stam = st.sidebar.slider("Stamina", 10, 99, 75, key='key_stam')
    in_str = st.sidebar.slider("Strength", 10, 99, 70, key='key_str')
    in_agg = st.sidebar.slider("Aggression", 10, 99, 65, key='key_agg')
    in_tack = st.sidebar.slider("Standing Tackle", 10, 99, 40, key='key_tack')
    in_int = st.sidebar.slider("Interceptions", 10, 99, 40, key='key_int')

user_input_dict = {
    'Age': input_age, 'Potential': input_potential, 'Best Position': input_position, 'Preferred Foot': input_foot,
    'BallControl': in_ball, 'Dribbling': in_drib, 'Composure': in_comp, 'Reactions': in_reac,
    'Finishing': in_fin, 'ShotPower': in_shot, 'LongShots': in_long, 'ShortPassing': in_pass,
    'Vision': in_vis, 'SprintSpeed': in_sprint, 'Stamina': in_stam, 'Strength': in_str,
    'Aggression': in_agg, 'StandingTackle': in_tack, 'Interceptions': in_int, 'Positioning': in_pos_att,
    'SlidingTackle': in_tack - 5, 'LongPassing': in_pass - 5
}

# =========================================================
# 5. MAIN DASHBOARD
# =========================================================
st.title("⚽ FIFA AI Scout Pro: Interactive Dashboard")
st.markdown("##### Final Project - Industrial Engineering & Management")
st.write("---")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🚀 Calculate Value & Run Scouting", use_container_width=True):
        st.session_state['run'] = True

    if st.session_state.get('run'):
        input_df = pd.DataFrame(columns=model_features)
        input_df.loc[0] = 0
        for col, val in user_input_dict.items():
            if col in input_df.columns: input_df.at[0, col] = val

        try:
            base_pred = model_pipeline.predict(input_df)[0]
            val_m = max(0, base_pred / 1_000_000)

            st.success("Estimated Market Value:")
            st.metric(label="Market Value", value=f"€{val_m:.2f} M")

            st.markdown("### 📊 AI Insights (Feature Impact)")
            check_list = ['SprintSpeed', 'Finishing', 'ShotPower', 'Dribbling', 'StandingTackle', 'Potential',
                          'BallControl', 'ShortPassing']
            impacts = {}
            improvements = {}
            IMPROVE_STEP = 10

            for attr in check_list:
                temp = input_df.copy()
                temp.at[0, attr] -= 10
                loss = base_pred - model_pipeline.predict(temp)[0]
                if loss > 0: impacts[attr] = loss

                temp2 = input_df.copy()
                if temp2.at[0, attr] <= 89:
                    temp2.at[0, attr] += IMPROVE_STEP
                    gain = model_pipeline.predict(temp2)[0] - base_pred
                    if gain > 0: improvements[attr] = gain

            # INSIGHTS FOR CSV & UI
            top_boosters = sorted(impacts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_suggests = sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:3]

            with st.expander("✅ Main Value Drivers", expanded=True):
                for attr, val in top_boosters:
                    st.write(f"**{attr}**: adds approx €{val / 1e6:.1f}M to value")
                    st.progress(min(100, int(user_input_dict[attr])))

            with st.expander("🚀 Recommendations to Improve"):
                for attr, val in top_suggests:
                    st.info(f"💡 Improving **{attr}** by {IMPROVE_STEP} points increases value by €{val / 1e6:.1f}M")

            st.write("---")
            u_vals = [in_sprint, (in_fin + in_shot) / 2, (in_pass + in_vis) / 2, in_drib, (in_tack + in_int) / 2,
                      (in_str + in_stam) / 2]
            cats = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defense', 'Physical']
            c_vals = get_player_stats_from_db(ref_player) if ref_player != "None" else None
            fig = create_radar_comparison(cats, u_vals, c_vals, input_name, ref_player)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

with col2:
    if st.session_state.get('run'):
        st.subheader(f"🔎 Scouting Report: {input_name}")
        st.info("💡 Select a row to compare stats instantly!")

        try:
            pool = players_db[players_db['Best Position'] == input_position].copy()
            if pool.empty:
                st.warning("No players found in this position.")
            else:
                # Similarity Engine (CBF)
                pool['sim_score'] = 100 - (
                        abs(pool['Potential'] - input_potential) * 1.0 +
                        abs(pool.get('SprintSpeed', 70) - in_sprint) * 0.5 +
                        abs(pool.get('Finishing', 60) - in_fin) * 0.5 +
                        abs(pool.get('Dribbling', 70) - in_drib) * 0.5 +
                        abs(pool.get('StandingTackle', 40) - in_tack) * 0.5
                ) / 3.0

                # TABLE 1: Wonderkids (Young & High Potential)
                st.markdown("### 🎣 1. Next-Gen Talents (Younger Targets)")
                wonderkids = pool[pool['Age'] < input_age].sort_values(['sim_score', 'Potential'],
                                                                       ascending=False).head(10)
                selected_wonder = st.dataframe(
                    wonderkids[['Name', 'Age', 'Value_EUR', 'Potential', 'sim_score']],
                    column_config={
                        "Value_EUR": st.column_config.NumberColumn("Value", format="€%d"),
                        "sim_score": st.column_config.ProgressColumn("Similarity", format="%d%%", min_value=0,
                                                                     max_value=100)
                    },
                    on_select="rerun", selection_mode="single-row", use_container_width=True, key="table_wonder"
                )
                if len(selected_wonder.selection.rows) > 0:
                    p_name = wonderkids.iloc[selected_wonder.selection.rows[0]]['Name']
                    st.pyplot(
                        create_radar_comparison(cats, u_vals, get_player_stats_from_db(p_name), input_name, p_name))

                st.write("---")

                # TABLE 2: Pure General Similarity (Regardless of Age/Value)
                st.markdown("### 🧬 2. Tactical Soulmates (General Similarity)")
                general_matches = pool.sort_values('sim_score', ascending=False).head(10)
                selected_gen = st.dataframe(
                    general_matches[['Name', 'Age', 'Value_EUR', 'Potential', 'sim_score']],
                    column_config={
                        "Value_EUR": st.column_config.NumberColumn("Value", format="€%d"),
                        "sim_score": st.column_config.ProgressColumn("Similarity", format="%d%%", min_value=0,
                                                                     max_value=100)
                    },
                    on_select="rerun", selection_mode="single-row", use_container_width=True, key="table_gen"
                )
                if len(selected_gen.selection.rows) > 0:
                    p_name = general_matches.iloc[selected_gen.selection.rows[0]]['Name']
                    st.pyplot(
                        create_radar_comparison(cats, u_vals, get_player_stats_from_db(p_name), input_name, p_name))

                # =========================================================
                # 6. CSV EXPORT - STRUCTURED TABLES
                # =========================================================
                st.write("---")
                st.subheader("📥 Export Final Scouting Report")
                try:
                    # BLOCK 1: Horizontal Target Player Info
                    input_row = pd.DataFrame([user_input_dict])
                    input_row.insert(0, 'Player Name', input_name)

                    # BLOCK 2: Predicted Value (Single line)
                    value_df = pd.DataFrame([{'Category': 'Predicted Market Value', 'Amount': f"€{base_pred:,.0f}"}])

                    # BLOCK 3: AI Insights & Recommendations
                    insight_list = []
                    for attr, val in top_boosters: insight_list.append(
                        {'Type': 'Strength', 'Feature': attr, 'Impact': f"Adds €{val / 1e6:.1f}M"})
                    for attr, val in top_suggests: insight_list.append(
                        {'Type': 'Improvement', 'Feature': attr, 'Impact': f"Gain €{val / 1e6:.1f}M"})
                    insight_df = pd.DataFrame(insight_list)

                    # Build CSV with logical breaks
                    csv_header = "--- TARGET PLAYER ATTRIBUTES ---\n"
                    csv_input = input_row.to_csv(index=False)
                    csv_val = "\n--- FINANCIAL PREDICTION ---\n" + value_df.to_csv(index=False)
                    csv_ins = "\n--- AI INSIGHTS & RECOMMENDATIONS ---\n" + insight_df.to_csv(index=False)
                    csv_w = "\n--- 1. NEXT-GEN TALENTS REPORT ---\n" + wonderkids[
                        ['Name', 'Age', 'Club', 'Value_EUR', 'Potential', 'sim_score']].to_csv(index=False)
                    csv_g = "\n--- 2. TACTICAL SOULMATES REPORT ---\n" + general_matches[
                        ['Name', 'Age', 'Club', 'Value_EUR', 'Potential', 'sim_score']].to_csv(index=False)

                    full_csv = csv_header + csv_input + csv_val + csv_ins + csv_w + csv_g

                    st.download_button(
                        label="📄 Download Comprehensive CSV Report",
                        data=full_csv.encode('utf-8-sig'),
                        file_name=f"FIFA_Scout_Report_{input_name}.csv",
                        mime="text/csv", use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating export: {e}")

        except Exception as e:
            st.error(f"Scouting Engine Error: {e}")
    else:
        st.write("### Welcome to FIFA AI Scout Pro")
        st.write("Input player data on the left and click the launch button to start the analysis.")
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Football_in_Bloomington%2C_Indiana%2C_1996.jpg",
                 use_container_width=True)