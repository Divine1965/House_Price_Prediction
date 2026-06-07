import os
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("Fast Mumbai House Price Training Started")

DATA_PATH = os.path.join("data", "Mumbai House Prices.csv")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# --- Load Dataset ---
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

# --- Use only a sample for faster training ---
df = df.sample(n=min(10000, len(df)), random_state=42)
print(f"Using a sample of {len(df)} rows for quick training")

# --- Helper Functions ---
def parse_area(x):
    """Convert area text into sqft float."""
    try:
        if pd.isna(x):
            return None
        s = str(x).replace(",", "").strip()
        if "-" in s:
            parts = [float(p) for p in s.split("-") if p.strip().isdigit()]
            return sum(parts) / len(parts)
        m = re.search(r"([0-9]*\.?[0-9]+)", s)
        return float(m.group(1)) if m else None
    except:
        return None

def price_to_lakhs(val, unit, area_sqft=None):
    """Convert price/unit text to lakhs."""
    try:
        num = float(str(val).replace(",", "").strip())
    except:
        return None
    if isinstance(unit, str):
        u = unit.lower()
        if "crore" in u or "cr" in u:
            return num * 100
        if "lakh" in u or "lac" in u:
            return num
        if "per" in u or "psf" in u:
            if area_sqft:
                return (num * area_sqft) / 100000
    return num

# --- Preprocess Columns ---
df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")
df["area_sqft"] = df["area"].apply(parse_area)
df["price_lakhs"] = df.apply(lambda r: price_to_lakhs(r["price"], r["price_unit"], r["area_sqft"]), axis=1)

# Clean string columns
for col in ["type", "locality", "region", "status"]:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

# Drop invalid or missing rows
df.dropna(subset=["bhk", "area_sqft", "price_lakhs"], inplace=True)
df = df[(df["area_sqft"] / df["bhk"]) >= 120]

print(f"Cleaned dataset: {len(df)} rows remain")

# Simplify locality column
loc_counts = df["locality"].value_counts()
rare_locs = loc_counts[loc_counts < 10].index
df["locality_clean"] = df["locality"].apply(lambda x: "other" if x in rare_locs else x)

# Optional: numeric age column
if "age" in df.columns:
    df["age_num"] = pd.to_numeric(df["age"], errors="coerce").fillna(0)
else:
    df["age_num"] = 0

# --- Features ---
features = ["locality_clean", "area_sqft", "bhk", "age_num"]
X = df[features]
y = df["price_lakhs"]

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocessing Pipeline ---
ohe_kwargs = {"handle_unknown": "ignore"}
if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
    ohe_kwargs["sparse_output"] = False
else:
    ohe_kwargs["sparse"] = False

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["area_sqft", "bhk", "age_num"]),
    ("cat", OneHotEncoder(**ohe_kwargs), ["locality_clean"])
])

# --- Model ---
model = Pipeline([
    ("pre", preprocessor),
    ("reg", RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1))
])

# --- Train ---
print("⚙️ Training model (this may take ~30–60 seconds)...")
model.fit(X_train, y_train)

# --- Evaluate ---
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Model trained successfully! RMSE: {rmse:.2f} lakhs")

# --- Save Model ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/mumbai_price_model.pkl")
print("💾 Model saved to model/mumbai_price_model.pkl")