# 🚗 Car Price Prediction using Machine Learning

## 📌 Overview

This project builds a **production-ready end-to-end machine learning system** to predict used car prices using structured data.
It includes **data preprocessing, feature engineering, model training, and deployment via an interactive Streamlit web application**.

---

## 🎯 Problem Statement

Predict the **listed price of used cars** based on factors such as:

* Engine specifications
* Vehicle usage
* Ownership history
* Structural and categorical attributes

---

## ⚙️ End-to-End Pipeline

A complete **Scikit-learn Pipeline** was implemented to automate:

* Column selection from raw dataset
* Data type conversion
* Missing value imputation (median, mode, group-based)
* Outlier treatment (IQR clipping)
* Feature engineering:

  * Car Age
  * Car Size
  * Engine-based flags
* Encoding:

  * OneHot Encoding
  * Ordinal Encoding
  * Frequency Encoding
* Feature scaling

👉 This ensures **consistent preprocessing during training and inference**

---

## 📊 Dataset Features

* Engine: Power, Torque, Cylinders
* Usage: Kilometers driven, Ownership
* Car Details: Fuel type, Transmission, Drive type
* Engineered Features: Car Age, Car Size, Usage Type

---

## 🤖 Models Implemented

* Linear Regression
* Ridge Regression (Polynomial)
* KNN Regressor
* Random Forest Regressor
* **XGBoost Regressor (Final Model)**

---

## 🏆 Final Model: XGBoost

### 📊 Performance

* **R² Score:** 0.94 – 0.95
* **MAE:** ₹61,000
* **Median Error:** ~9–10%

---

## 🧠 Why XGBoost?

* Captures complex non-linear relationships
* Boosting reduces bias and variance
* Outperforms traditional models on structured data

---

## 🔍 Error Analysis

* Median error ≈ **9–10%**
* Majority predictions within **10–17% range**
* Higher errors observed in:

  * Low-priced vehicles
  * Rare configurations

---

## 📈 Model Interpretability (SHAP)

SHAP was used to analyze feature impact:

### 🔑 Key Insights

* **Car Age** → Strong negative impact
* **Max Power Delivered** → Strong positive impact
* **Max Torque Delivered** → Moderate impact
* **Km Driven** → Negative correlation

---

## 🖥️ Streamlit Web App

An interactive UI was built using Streamlit:

### Features:

* User input form for car details
* Real-time price prediction
* Dynamic price range slider
* Recommended cars based on selected price range

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/car-price-prediction-streamlit-app.git
cd car-price-prediction-streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 💼 Project Highlights

* Built **end-to-end ML pipeline**
* Achieved **high accuracy (R² ≈ 0.95)**
* Implemented **custom transformers**
* Integrated **ML model with UI**
* Applied **SHAP for interpretability**
* Designed for **real-world deployment**

---

## 🧠 Key Learnings

* Feature engineering significantly improves performance
* Log transformation stabilizes regression
* Tree-based models outperform linear models
* Pipelines ensure production consistency
* Interpretability is critical in ML systems

---

## 📂 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* SHAP

---

## 📌 Conclusion

This project demonstrates a **complete ML lifecycle**, from raw data to deployment.
The system achieves **high accuracy and reliability**, making it suitable for real-world applications in automated car pricing.

