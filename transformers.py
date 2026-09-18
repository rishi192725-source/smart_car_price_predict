
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

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

