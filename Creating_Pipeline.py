# =========================================
# 1. IMPORTS
# =========================================
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBRegressor


# =========================================
# 2. LOAD RAW DATA
# =========================================
df = pd.read_csv("datasets/Car_Dekho_dataset.csv")


# =========================================
# 3. CUSTOM TRANSFORMERS
# (each transformer mirrors one notebook step exactly)
# =========================================

# -----------------------------------------
# Step 3 — Select the 17 important columns
# (mirrors notebook cell 9 + 11)
# -----------------------------------------
class ColumnSelector(BaseEstimator, TransformerMixin):
    IMPORTANT_COLS = [
        "myear", "km", "model", "fuel", "transmission", "owner_type", "Seats",
        "Drive Type", "Engine Type", "No of Cylinder", "Max Power Delivered",
        "Max Torque Delivered", "Length", "Width", "Height",
        "Gear Box", "City",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Case-insensitive column matching
        col_lower_map = {c.lower(): c for c in X.columns}
        selected = []
        for c in self.IMPORTANT_COLS:
            matched = col_lower_map.get(c.lower())
            if matched is None:
                raise ValueError(f"Column not found in CSV: '{c}'")
            selected.append(matched)
        X = X[selected]

        # Standardise: replace _ with space, Title Case  (mirrors notebook cell 11)
        X.columns = X.columns.str.replace("_", " ").str.title()
        return X


# -----------------------------------------
# Step 5 — Data Type Conversion
# (mirrors notebook cells 13–14)
# -----------------------------------------
class DataTypeConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Coerce all numeric columns (bad strings → NaN, filled in next step)
        for col in ["Seats", "No Of Cylinder", "Max Power Delivered",
                    "Max Torque Delivered", "Km", "Length", "Width", "Height"]:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Gear Box → Gear Type  (mirrors notebook cell 14)
        def classify_gear(x):
            if pd.isna(x):       return np.nan
            x = str(x).lower()
            if "cvt"       in x: return "CVT"
            if "automatic" in x: return "Automatic"
            if "direct"    in x: return "Direct"
            return "Manual"

        X["Gear Type"] = X["Gear Box"].apply(classify_gear)
        X.drop("Gear Box", axis=1, inplace=True)

        return X


# -----------------------------------------
# Step 6 — Missing Value Imputation
# (mirrors notebook cells 21–25; Kerb Weight already excluded from selection)
# -----------------------------------------
class MissingValueImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Medians for skewed numerics (cell 21)
        self.median_map_ = {}
        for col in ["No Of Cylinder", "Max Power Delivered", "Max Torque Delivered"]:
            self.median_map_[col] = X[col].median()

        # Modes for categoricals + Seats (cell 22)
        self.mode_map_ = {}
        for col in ["Drive Type", "Gear Type", "Seats"]:
            self.mode_map_[col] = X[col].mode()[0]

        # Group means for dimensions, keyed by Model (cell 23)
        self.dim_group_mean_ = {}
        self.dim_global_mean_ = {}
        for col in ["Length", "Width", "Height"]:
            self.dim_group_mean_[col] = X.groupby("Model")[col].mean().to_dict()
            self.dim_global_mean_[col] = X[col].mean()

        # Engine Type: group mode per Model, global mode fallback (cell 24)
        self.engine_group_mode_ = (
            X.groupby("Model")["Engine Type"]
             .agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
             .to_dict()
        )
        self.engine_global_mode_ = X["Engine Type"].mode()[0]

        return self

    def transform(self, X):
        X = X.copy()

        # Numeric medians
        for col, val in self.median_map_.items():
            X[col] = X[col].fillna(val)
        X["No Of Cylinder"] = X["No Of Cylinder"].astype("int64")

        # Categorical modes
        for col, val in self.mode_map_.items():
            X[col] = X[col].fillna(val)
        X["Seats"] = X["Seats"].astype("int64")

        # Dimension group mean → global mean fallback
        for col in ["Length", "Width", "Height"]:
            mask = X[col].isna()
            X.loc[mask, col] = X.loc[mask, "Model"].map(self.dim_group_mean_[col])
            X[col] = X[col].fillna(self.dim_global_mean_[col])

        # Engine Type group mode → global mode fallback
        mask = X["Engine Type"].isna()
        X.loc[mask, "Engine Type"] = X.loc[mask, "Model"].map(self.engine_group_mode_)
        X["Engine Type"] = X["Engine Type"].fillna(self.engine_global_mode_)

        return X


# -----------------------------------------
# Step 7 — Outlier Clipping (IQR)
# (mirrors notebook cell 30 — X-side cols only; target clipped before split)
# -----------------------------------------
class OutlierClipper(BaseEstimator, TransformerMixin):
    CLIP_COLS = ["Km", "Max Power Delivered", "Max Torque Delivered"]

    def fit(self, X, y=None):
        self.bounds_ = {}
        for col in self.CLIP_COLS:
            Q1  = X[col].quantile(0.25)
            Q3  = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            X[col] = X[col].clip(lo, hi)
        return X


# -----------------------------------------
# Step 8 — Feature Engineering
# (mirrors notebook cells 33–42)
# -----------------------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Car Age (cell 33)
        X["Car Age"] = 2026 - X["Myear"]
        X.drop("Myear", axis=1, inplace=True)

        # Usage Type — string, ordinal-encoded in preprocessor (cell 33)
        def usage_type(x):
            if x < 20000:   return "Low"
            elif x < 80000: return "Median"
            else:           return "High"
        X["Usage Type"] = X["Km"].apply(usage_type)

        # Brand (cell 35)
        X["Brand"] = X["Model"].astype(str).str.split().str[0]

        # Car Segment (cell 36)
        def segment(x):
            x = str(x).lower()
            if any(k in x for k in ["swift","i10","alto","wagon","polo","kwid"]): return "Hatchback"
            elif any(k in x for k in ["city","verna","ciaz","amaze"]):            return "Sedan"
            elif any(k in x for k in ["creta","scorpio","fortuner","xuv","harrier","seltos"]): return "SUV"
            else: return "Other"
        X["Car Segment"] = X["Model"].apply(segment)
        X.drop("Model", axis=1, inplace=True)

        # Car Size (cell 38)
        X["Car Size"] = (X["Length"] * X["Width"] * X["Height"]) / 1e9
        X.drop(columns=["Length","Width","Height"], inplace=True)

        # Is Turbo (cell 40)
        X["Is Turbo"] = X["Engine Type"].str.lower().apply(
            lambda x: 1 if "turbo" in str(x) else 0)

        # Is Advanced Engine (cell 41)
        X["Is Advanced Engine"] = X["Engine Type"].str.lower().apply(
            lambda x: 1 if any(k in str(x) for k in ["dohc","vvt","tsi","tfsi","gdi"]) else 0)
        X.drop("Engine Type", axis=1, inplace=True)

        return X


# -----------------------------------------
# Step 9a — City Frequency Encoding
# (mirrors notebook cell 47)
# -----------------------------------------
class CityFrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.freq_map_ = X["City"].value_counts().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X["City Freq"] = X["City"].map(self.freq_map_).fillna(1)
        X.drop("City", axis=1, inplace=True)
        return X


# -----------------------------------------
# Step 9b — Brand → Category Tier
# (mirrors notebook cell 49)
# -----------------------------------------
class BrandCategoryTransformer(BaseEstimator, TransformerMixin):
    BRAND_MAP = {
        "maruti":"Budget","hyundai":"Budget","tata":"Budget","datsun":"Budget","renault":"Budget",
        "honda":"Mid","toyota":"Mid","kia":"Mid","nissan":"Mid","volkswagen":"Mid","skoda":"Mid","ford":"Mid",
        "jeep":"Premium","mg":"Premium","citroen":"Premium","fiat":"Premium","isuzu":"Premium",
        "bmw":"Luxury","audi":"Luxury","mercedes-benz":"Luxury","jaguar":"Luxury","volvo":"Luxury","lexus":"Luxury",
        "porsche":"Ultra_Luxury","ferrari":"Ultra_Luxury","lamborghini":"Ultra_Luxury",
        "bentley":"Ultra_Luxury","aston":"Ultra_Luxury","rolls-royce":"Ultra_Luxury","maserati":"Ultra_Luxury",
        "mahindra":"Commercial","force":"Commercial","ashok":"Commercial","icml":"Commercial",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Brand Category"] = X["Brand"].str.lower().map(self.BRAND_MAP).fillna("Other")
        X.drop("Brand", axis=1, inplace=True)
        return X


# =========================================
# 4. CLIP TARGET OUTLIERS (before split)
# mirrors notebook cell 30 for Listed Price
# =========================================
X = df.drop("listed_price", axis=1)
y = df["listed_price"].copy()

Q1  = y.quantile(0.25)
Q3  = y.quantile(0.75)
IQR = Q3 - Q1
y   = y.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# Log-transform target (stabilises variance for XGBoost)
y = np.log(y)


# =========================================
# 5. SPLIT DATA
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# 6. COLUMN DEFINITIONS
# (columns present after all transformations)
# =========================================
num_cols = [
    "Km", "Seats", "No Of Cylinder",
    "Max Power Delivered", "Max Torque Delivered",
    "Car Size", "Car Age", "City Freq",
    "Is Turbo", "Is Advanced Engine",
]

# One-hot cols with drop_first=True (mirrors notebook cell 51)
ohe_cols = ["Brand Category", "Fuel", "Car Segment", "Transmission", "Drive Type", "Gear Type"]

# Ordinal cols (mirrors notebook cells 54–55)
owner_order = [["unregistered car", "first", "second", "third", "fourth", "fifth"]]
usage_order = [["Low", "Median", "High"]]


# =========================================
# 7. PREPROCESSOR
# =========================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num",   StandardScaler(),
                  num_cols),

        ("ohe",   OneHotEncoder(drop="first",                    # mirrors drop_first=True
                                handle_unknown="ignore",
                                sparse_output=False),
                  ohe_cols),

        ("owner", OrdinalEncoder(categories=owner_order,
                                 handle_unknown="use_encoded_value",
                                 unknown_value=-1),
                  ["Owner Type"]),

        ("usage", OrdinalEncoder(categories=usage_order,
                                 handle_unknown="use_encoded_value",
                                 unknown_value=-1),
                  ["Usage Type"]),
    ],
    remainder="drop"
)


# =========================================
# 8. MODEL
# =========================================
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
)


# =========================================
# 9. FINAL PIPELINE
# =========================================
pipeline = Pipeline([
    ("column_selector",     ColumnSelector()),           # Select 17 cols + rename
    ("dtype_converter",     DataTypeConverter()),         # Fix dtypes; Gear Box → Gear Type
    ("imputer",             MissingValueImputer()),       # Fill NaN: median/mode/group mean
    ("outlier_clipper",     OutlierClipper()),            # IQR clip Km, Power, Torque
    ("feature_engineering", FeatureEngineering()),        # Car Age, Car Size, Turbo flags…
    ("city_freq",           CityFrequencyEncoder()),      # City → frequency count
    ("brand_category",      BrandCategoryTransformer()),  # Brand → price-tier category
    ("preprocessing",       preprocessor),               # Scale + OHE + OrdinalEncode
    ("model",               model),
])


# =========================================
# 10. TRAIN
# =========================================
pipeline.fit(X_train, y_train)
print("✅ Pipeline trained successfully!")

import pandas as pd

model = pipeline.named_steps["model"]

importances = pd.Series(model.feature_importances_)
print(importances.sort_values(ascending=False))
# =========================================
# 11. EVALUATE
# =========================================
from sklearn.metrics import mean_absolute_error, r2_score

y_pred_log = pipeline.predict(X_test)
y_pred     = np.exp(y_pred_log)
y_true     = np.exp(y_test)

mae = mean_absolute_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)

print(f"📊 MAE  : ₹{mae:,.0f}")
print(f"📊 R²   : {r2:.4f}")


# =========================================
# 12. SAVE PIPELINE
# =========================================
with open("car_price_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("💾 Pipeline saved to car_price_pipeline.pkl")


# =========================================
# 13. INFERENCE — raw CSV in, price out
# =========================================
# with open("car_price_pipeline.pkl", "rb") as f:
#     loaded = pickle.load(f)
# new_df     = pd.read_csv("new_cars.csv")   # raw, no preprocessing needed
# log_prices = loaded.predict(new_df)
# prices     = np.exp(log_prices)
# print(prices)