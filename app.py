import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# =========================================================
# 1. הגדרות עמוד ועיצוב
# =========================================================
st.set_page_config(
    page_title="FIFA AI Scout Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {direction: rtl;}
    h1, h2, h3, h4, h5, h6, p, div {text-align: right;}
    .stTextInput > label {float: right;}
    .stSelectbox > label {float: right;}
    .stNumberInput > label {float: right;}
    .stSlider > label {float: right;}
    .stDataFrame {direction: ltr;} 
    /* עיצוב כפתור איפוס */
    div.stButton > button:first-child {
        background-color: #f0f2f6;
        color: black;
        border: 1px solid #d0d0d0;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# 2. פונקציות טעינה ואיפוס
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
        st.error("❌ שגיאה: קבצי המודל (.pkl) חסרים בתיקייה.")
        st.stop()


model_pipeline, model_features, players_db = load_all_data()


def reset_values():
    for key in list(st.session_state.keys()):
        if key.startswith('key_'):
            del st.session_state[key]
    st.session_state['run'] = False


# =========================================================
# 3. פונקציות עזר לגרפים ונתונים
# =========================================================

def get_player_stats_from_db(player_name):
    """שולף נתונים של שחקן מהמאגר וממיר אותם ל-6 קטגוריות הרדאר"""
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
    """מצייר גרף רדאר עם ערכים מספריים"""
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

    # --- שחקן המשתמש (כחול) ---
    vals = user_vals + user_vals[:1]
    ax.plot(angles, vals, linewidth=2, linestyle='solid', label=user_name, color='#1f77b4')
    ax.fill(angles, vals, '#1f77b4', alpha=0.1)

    for ang, val in zip(angles[:-1], user_vals):
        ax.text(ang, val + 12, str(int(val)), ha='center', va='center', fontsize=9, fontweight='bold', color='#1f77b4')

    # --- שחקן להשוואה (אדום) ---
    if comp_vals:
        c_vals = comp_vals + comp_vals[:1]
        ax.plot(angles, c_vals, linewidth=2, linestyle='solid', label=comp_name, color='#d62728')
        ax.fill(angles, c_vals, '#d62728', alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Player Comparison Analysis", size=14, y=1.1)
    return fig


# =========================================================
# 4. סרגל צד - קלט מלא
# =========================================================
st.sidebar.title("🛠️ נתוני שחקן")

if st.sidebar.button("🔄 אפס נתונים (Reset)", on_click=reset_values):
    st.rerun()

st.sidebar.markdown("### 🔍 השוואה לשחקן קיים")
all_names = ["ללא"] + sorted(players_db['Name'].unique().tolist())
ref_player = st.sidebar.selectbox("בחר כוכב להשוואה:", all_names, key='key_ref')

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 פרטים כלליים")
input_name = st.sidebar.text_input("שם השחקן", "My Player")

input_age = st.sidebar.slider("גיל", 15, 45, 24, key='key_age')
input_potential = st.sidebar.slider("פוטנציאל", 40, 99, 85, key='key_potential')
input_position = st.sidebar.selectbox("עמדה", ['ST', 'CF', 'LW', 'RW', 'CAM', 'CM', 'CDM', 'CB', 'LB', 'RB', 'GK'],
                                      key='key_pos')
input_foot = st.sidebar.radio("רגל מועדפת", ["Right", "Left"], horizontal=True)

st.sidebar.markdown("### ⚽ שליטה וטכניקה")
in_ball = st.sidebar.slider("שליטה בכדור", 10, 99, 75, key='key_ball')
in_drib = st.sidebar.slider("כדרור", 10, 99, 75, key='key_drib')
in_comp = st.sidebar.slider("קור רוח", 10, 99, 70, key='key_comp')
in_reac = st.sidebar.slider("תגובות", 10, 99, 70, key='key_reac')

st.sidebar.markdown("### 🎯 התקפה ובעיטה")
in_fin = st.sidebar.slider("סיומת", 10, 99, 70, key='key_fin')
in_shot = st.sidebar.slider("עוצמת בעיטה", 10, 99, 75, key='key_shot')
in_long = st.sidebar.slider("בעיטות מרחוק", 10, 99, 65, key='key_long')
in_pos_att = st.sidebar.slider("מיקום התקפי", 10, 99, 70, key='key_pos_att')

st.sidebar.markdown("### 🔭 מסירות")
in_pass = st.sidebar.slider("מסירות קצרות", 10, 99, 72, key='key_pass')
in_vis = st.sidebar.slider("ראיית משחק", 10, 99, 68, key='key_vis')

st.sidebar.markdown("### 💪 פיזי והגנה")
in_sprint = st.sidebar.slider("מהירות", 10, 99, 80, key='key_sprint')
in_stam = st.sidebar.slider("סיבולת", 10, 99, 75, key='key_stam')
in_str = st.sidebar.slider("חוזק", 10, 99, 70, key='key_str')
in_agg = st.sidebar.slider("אגרסיביות", 10, 99, 65, key='key_agg')

st.sidebar.markdown("### 🛡️ הגנה")
in_tack = st.sidebar.slider("תיקול עומד", 10, 99, 40, key='key_tack')
in_int = st.sidebar.slider("חטיפות", 10, 99, 40, key='key_int')

user_input_dict = {
    'Age': input_age, 'Potential': input_potential, 'Best Position': input_position, 'Preferred Foot': input_foot,
    'BallControl': in_ball, 'Dribbling': in_drib, 'Composure': in_comp, 'Reactions': in_reac,
    'Finishing': in_fin, 'ShotPower': in_shot, 'LongShots': in_long, 'ShortPassing': in_pass,
    'Vision': in_vis, 'SprintSpeed': in_sprint, 'Stamina': in_stam, 'Strength': in_str,
    'Aggression': in_agg, 'StandingTackle': in_tack, 'Interceptions': in_int, 'Positioning': in_pos_att,
    'SlidingTackle': in_tack - 5, 'LongPassing': in_pass - 5
}

# =========================================================
# 5. מסך ראשי
# =========================================================
st.title("⚽ FIFA AI Scout Pro: Interactive Dashboard")
st.markdown("### פרויקט גמר - הנדסת תעשייה וניהול")
st.write("---")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🚀 חשב שווי והרץ סקאוטינג", use_container_width=True):
        st.session_state['run'] = True

    if st.session_state.get('run'):
        input_df = pd.DataFrame(columns=model_features)
        input_df.loc[0] = 0
        for col, val in user_input_dict.items():
            if col in input_df.columns: input_df.at[0, col] = val

        try:
            base_pred = model_pipeline.predict(input_df)[0]
            val_m = max(0, base_pred / 1_000_000)

            st.success("שווי שוק מוערך:")
            st.metric(label="Market Value", value=f"€{val_m:.2f} M")

            st.markdown("### 📊 ניתוח גורמים (AI Insights)")
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

            with st.expander("✅ מהם הגורמים העיקריים לשווי הגבוה?", expanded=True):
                sorted_impacts = sorted(impacts.items(), key=lambda x: x[1], reverse=True)[:3]
                for attr, val in sorted_impacts:
                    st.write(f"**{attr}**: תורם כ-€{val / 1e6:.1f}M לשווי")
                    st.progress(min(100, int((user_input_dict[attr] / 100) * 100)))

            with st.expander("🚀 איפה כדאי להשתפר? (המלצות המערכת)"):
                sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:3]
                for attr, val in sorted_improvements:
                    st.info(f"💡 שיפור של **{IMPROVE_STEP} נקודות** ב-**{attr}** יעלה את השווי ב-€{val / 1e6:.1f}M")

            st.write("---")
            u_vals = [in_sprint, (in_fin + in_shot) / 2, (in_pass + in_vis) / 2, in_drib, (in_tack + in_int) / 2,
                      (in_str + in_stam) / 2]
            cats = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defense', 'Physical']

            c_vals = None
            c_name = ""
            if ref_player != "ללא":
                c_vals = get_player_stats_from_db(ref_player)
                c_name = ref_player

            fig = create_radar_comparison(cats, u_vals, c_vals, input_name, c_name)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"שגיאה בחיזוי: {e}")

with col2:
    if st.session_state.get('run'):
        st.subheader(f"🔎 דוח סקאוטינג: {input_name}")
        st.info("💡 לחץ על שורה בטבלה כדי לבצע השוואה גרפית מיידית!")

        try:
            pool = players_db[players_db['Best Position'] == input_position].copy()
            if pool.empty:
                st.warning("לא נמצאו שחקנים בעמדה זו.")
            else:
                pool['sim_score'] = 100 - (
                        abs(pool['Potential'] - input_potential) * 1.0 +
                        abs(pool.get('SprintSpeed', 70) - in_sprint) * 0.5 +
                        abs(pool.get('Finishing', 60) - in_fin) * 0.5 +
                        abs(pool.get('Dribbling', 70) - in_drib) * 0.5 +
                        abs(pool.get('StandingTackle', 40) - in_tack) * 0.5
                ) / 3.0

                st.markdown("### 🎣 1. איתור כישרונות (Wonderkids)")
                similar_pool = pool[pool['sim_score'] > 70].copy()
                wonderkids = similar_pool[similar_pool['Age'] <= input_age].sort_values(['Age', 'Potential'],
                                                                                        ascending=[True, False]).head(
                    10)

                selected_wonder = st.dataframe(
                    wonderkids[['Name', 'Age', 'Value_EUR', 'Potential', 'sim_score']],
                    column_config={
                        "Value_EUR": st.column_config.NumberColumn("שווי", format="€%d"),
                        "sim_score": st.column_config.ProgressColumn("דמיון", format="%d%%", min_value=0, max_value=100)
                    },
                    on_select="rerun",
                    selection_mode="single-row",
                    use_container_width=True,
                    key="table_wonder"
                )

                if len(selected_wonder.selection.rows) > 0:
                    idx = selected_wonder.selection.rows[0]
                    p_name = wonderkids.iloc[idx]['Name']
                    st.markdown(f"#### 📊 השוואה מול: {p_name}")
                    t_vals = get_player_stats_from_db(p_name)
                    fig_comp = create_radar_comparison(cats, u_vals, t_vals, input_name, p_name)
                    st.pyplot(fig_comp)

                st.write("---")
                st.markdown("### 💰 2. הזדמנויות בשוק (Bargains)")
                bargains = similar_pool[similar_pool['Value_EUR'] < base_pred * 0.9].sort_values('Potential',
                                                                                                 ascending=False).head(
                    5)

                selected_bargain = st.dataframe(
                    bargains[['Name', 'Age', 'Value_EUR', 'Potential', 'sim_score']],
                    column_config={
                        "Value_EUR": st.column_config.NumberColumn("מחיר שוק", format="€%d"),
                        "sim_score": st.column_config.ProgressColumn("דמיון", format="%d%%", min_value=0, max_value=100)
                    },
                    on_select="rerun",
                    selection_mode="single-row",
                    use_container_width=True,
                    key="table_bargain"
                )

                if len(selected_bargain.selection.rows) > 0:
                    idx = selected_bargain.selection.rows[0]
                    p_name = bargains.iloc[idx]['Name']
                    st.markdown(f"#### 📊 השוואה מול המציאה: {p_name}")
                    t_vals = get_player_stats_from_db(p_name)
                    fig_comp2 = create_radar_comparison(cats, u_vals, t_vals, input_name, p_name)
                    st.pyplot(fig_comp2)

                # =========================================================
                # 6. הוספת כפתור ייצוא דו"ח (Export)
                # =========================================================
                st.write("---")
                st.subheader("📥 ייצוא דו"
                ח
                סקאוטינג
                ")

                try:
                    if not wonderkids.empty or not bargains.empty:
                        export_df = pd.concat([wonderkids, bargains]).drop_duplicates(subset=['Name'])
                        csv_data = export_df.to_csv(index=False).encode('utf-8-sig')

                        st.download_button(
                            label="📄 הורד דו"
                        ח
                        שחקנים(CSV)
                        ",
                        data = csv_data,
                               file_name = f"Scouting_Report_{input_name}_{input_position}.csv",
                                           mime = "text/csv",
                                                  use_container_width = True
                        )
                        except:
                        pass

            except Exception as e:
            st.error(f"שגיאה במנוע הסקאוטינג: {e}")

    else:
        st.write("### ברוכים הבאים")
        st.write("הזן נתונים ולחץ על הכפתור כדי להתחיל.")
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/ad/Football_in_Bloomington%2C_Indiana%2C_1996.jpg",
                 use_container_width=True)