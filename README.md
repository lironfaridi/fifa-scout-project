# ⚽ FIFA AI Scout Pro
**An Advanced Machine Learning Decision Support System for Football Scouting**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fifa-scout-project-78u6oqwnnexxsqxfvuqvjg.streamlit.app/)

## 📌 Overview
Developed as a final academic project in Industrial Engineering & Management, **FIFA AI Scout Pro** is an interactive web application designed to assist football scouts and managers. It leverages Machine Learning to predict a player's true market value and uses Content-Based Filtering (CBF) to discover young talents and tactical soulmates based on multi-dimensional attribute analysis.

## 🚀 Key Features
* **Accurate Valuation:** Uses an **XGBoost** regression model trained on thousands of players to predict market value based on 20+ technical and physical attributes.
* **AI Insights:** Explains the prediction by highlighting the top value-driving attributes and suggesting specific areas for improvement (Feature Importance).
* **Versus Mode:** Allows side-by-side graphical comparison (Radar Charts) between a custom prospect and real-world database stars.
* **Talent Discovery Engine:** Finds "Wonderkids" (young high-potential matches) and "Tactical Soulmates" using Cosine Similarity on player features.
* **Automated Reporting:** Exports a comprehensive scouting dossier directly to a CSV file.

## 🛠️ Technology Stack
* **Python** (Pandas, NumPy)
* **Machine Learning:** Scikit-Learn, XGBoost
* **Frontend/UI:** Streamlit
* **Visualization:** Matplotlib

## 📂 Repository Structure
* `app.py`: The main Streamlit application code.
* `fifa_model_pipeline.pkl`: The trained XGBoost prediction pipeline.
* `model_features.pkl`: Saved feature columns for model consistency.
* `fifa_players_lite.pkl`: The optimized player database for the discovery engine.

## 👨‍💻 Authors
* **Liron Faridi**
* **Dean Ashur**
