# ------------------------------
# 🚗 Car Price Prediction App (Fixed Version)
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

st.set_page_config(page_title="Car Price Prediction", layout="wide")

# ------------------------------
# 1️⃣ Sidebar
# ------------------------------
st.sidebar.title("Car Price Predictor")
st.sidebar.markdown("""
- Compare models  
- Predict car prices  
- Visualize results  
""")

# ------------------------------
# 2️⃣ Load dataset (FIXED)
# ------------------------------
DATA_FILE = "car_data.csv"

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, sep=None, engine="python")
        except:
            df = pd.read_csv(DATA_FILE)
    else:
        df = None
    return df

df = load_data()

# إذا الملف غير موجود → نمنع crash
if df is None:
    st.error("❌ car_data.csv not found. Please place it in the same folder.")
    st.stop()

# تنظيف الأعمدة (IMPORTANT FIX)
df.columns = df.columns.str.strip()

# ------------------------------
# 3️⃣ Features / Target
# ------------------------------
features = ["EngineSize", "Horsepower", "Weight", "MPG", "Cylinders"]

# تحقق من الأعمدة
missing = [c for c in features + ["Price"] if c not in df.columns]
if missing:
    st.error(f"❌ Missing columns in dataset: {missing}")
    st.stop()

X = df[features]
y = df["Price"]

# ------------------------------
# 4️⃣ Train/Test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 5️⃣ Models
# ------------------------------
MODEL_FILES = {
    "ridge": "ridge_model.pkl",
    "rf": "rf_model.pkl",
    "gb": "gb_model.pkl"
}

all_exist = all(os.path.exists(f) for f in MODEL_FILES.values())

if not all_exist:
    st.info("Training models...")

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

    ridge.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    joblib.dump(ridge, MODEL_FILES["ridge"])
    joblib.dump(rf, MODEL_FILES["rf"])
    joblib.dump(gb, MODEL_FILES["gb"])
else:
    ridge = joblib.load(MODEL_FILES["ridge"])
    rf = joblib.load(MODEL_FILES["rf"])
    gb = joblib.load(MODEL_FILES["gb"])

# ------------------------------
# 6️⃣ Evaluation
# ------------------------------
ridge_pred = ridge.predict(X_test)
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

rmse_values = [
    np.sqrt(mean_squared_error(y_test, ridge_pred)),
    np.sqrt(mean_squared_error(y_test, rf_pred)),
    np.sqrt(mean_squared_error(y_test, gb_pred))
]

r2_values = [
    r2_score(y_test, ridge_pred),
    r2_score(y_test, rf_pred),
    r2_score(y_test, gb_pred)
]

model_names = ["RidgeCV", "RandomForest", "GradientBoosting"]

# ------------------------------
# 7️⃣ UI
# ------------------------------
st.title("🚗 Car Price Prediction App")

st.subheader("📊 Model Performance")

eval_df = pd.DataFrame({
    "Model": model_names,
    "RMSE": [round(x, 2) for x in rmse_values],
    "R2": [round(x, 3) for x in r2_values]
})

st.dataframe(eval_df)

# ------------------------------
# 8️⃣ Plots
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(x=model_names, y=rmse_values, ax=ax)
    ax.set_title("RMSE")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.barplot(x=model_names, y=r2_values, ax=ax)
    ax.set_title("R2 Score")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ------------------------------
# 9️⃣ Prediction
# ------------------------------
st.subheader("🧠 Predict Price")

with st.form("form"):
    engine = st.number_input("Engine Size", 500, 10000, 2000)
    hp = st.number_input("Horsepower", 50, 1000, 150)
    weight = st.number_input("Weight", 500, 5000, 1500)
    mpg = st.number_input("MPG", 5, 100, 25)
    cyl = st.selectbox("Cylinders", [3,4,5,6,8,12])

    submit = st.form_submit_button("Predict")

if submit:
    input_data = np.array([[engine, hp, weight, mpg, cyl]])

    ridge_p = ridge.predict(input_data)[0]
    rf_p = rf.predict(input_data)[0]
    gb_p = gb.predict(input_data)[0]

    st.success("Predictions")

    st.metric("Ridge", f"${ridge_p:,.0f}")
    st.metric("RandomForest", f"${rf_p:,.0f}")
    st.metric("GradientBoosting", f"${gb_p:,.0f}")

# ------------------------------
# 🔟 Data View
# ------------------------------
st.subheader("📊 Dataset Preview")

if st.checkbox("Show data"):
    st.dataframe(df)