import streamlit as st
import pandas as pd
import numpy as np
import pickle
from transformers import *
# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    with open("car_price_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="🚗 Car Price Predictor", layout="wide")

st.title("🚗 AI Car Price Predictor")
st.markdown("### Get accurate car price using Machine Learning")

# =========================================
# SIDEBAR QUICK FILTER
# =========================================
st.sidebar.header("💰 Quick Price Range")

price_range = st.sidebar.slider(
    "Select Budget (₹)",
    100000, 3000000, (500000, 1500000), step=50000
)

# =========================================
# MAIN FORM
# =========================================
st.markdown("## 🧾 Enter Car Details")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.selectbox("📅 Year", list(range(2005, 2025)))
        km = st.number_input("🚗 Kilometers Driven", 0, 300000, 50000)
        fuel = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG"])

    with col2:
        transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("👤 Owner Type", ["first", "second", "third"])
        seats = st.selectbox("💺 Seats", [4, 5, 7])

    with col3:
        power = st.number_input("⚡ Engine Power (bhp)", 40, 500, 100)
        torque = st.number_input("🔩 Max Torque", 50, 1000, 200)
        cylinders = st.selectbox("🔧 Cylinders", [3, 4, 6, 8])

    model_name = st.text_input("🚘 Car Model (e.g. Swift, Creta, City)")
    city = st.selectbox("🌆 City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Other"])

    submitted = st.form_submit_button("🔍 Predict Price")

# =========================================
# PREDICTION
# =========================================
if submitted:

    input_data = pd.DataFrame([{
        "myear": year,
        "km": km,
        "model": model_name,
        "fuel": fuel,
        "transmission": transmission,
        "owner_type": owner,
        "Seats": seats,
        "Drive Type": "FWD",
        "Engine Type": "normal",
        "No of Cylinder": cylinders,
        "Max Power Delivered": power,
        "Max Torque Delivered": torque,
        "Length": 4000,
        "Width": 1700,
        "Height": 1600,
        "Gear Box": transmission,
        "City": city,
    }])

    log_price = model.predict(input_data)
    price = int(np.exp(log_price)[0])

    st.success(f"💰 Estimated Price: ₹ {price:,.0f}")

    # =========================================
    # FEATURE IMPORTANCE MESSAGE
    # =========================================
    st.info("⚡ Price mainly depends on Power, Brand, and Car Size")

    # =========================================
    # SIMPLE RECOMMENDATION MOCK
    # =========================================
    st.markdown("## 🚘 Similar Cars in Your Budget")

    sample_cars = [
        {"name": "Hyundai i20", "price": 700000},
        {"name": "Maruti Swift", "price": 600000},
        {"name": "Honda City", "price": 1200000},
        {"name": "Kia Seltos", "price": 1500000},
    ]

    for car in sample_cars:
        if price_range[0] <= car["price"] <= price_range[1]:
            st.write(f"👉 {car['name']} — ₹{car['price']:,}")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning")