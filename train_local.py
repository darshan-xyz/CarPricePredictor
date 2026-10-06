import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import kagglehub
import os
import glob
import numpy as np

print("--- STARTING SAFE TRAINING ---")

# 1. Download Dataset
print("Downloading Dataset...")
path = kagglehub.dataset_download("manishkr1754/cardekho-used-car-data")
csv_files = glob.glob(os.path.join(path, "*.csv"))

# Load the file
# We know from your logs that 'cardekho_dataset.csv' is the one being loaded
df = pd.read_csv(csv_files[0])

print(f"✅ Loaded file with {len(df.columns)} columns.")

# 2. COLUMN CLEANUP (Explicit Logic)
# Clean header names (remove spaces, make lowercase)
df.columns = df.columns.str.lower().str.strip()

# Drop "unnamed" columns (index numbers) to prevent confusion
cols_to_drop = [c for c in df.columns if 'unnamed' in c]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)

# Standardize names using a dictionary (No fuzzy guessing)
rename_map = {
    'fuel_type': 'fuel',
    'transmission_type': 'transmission',
    'km_driven': 'km_driven',
    'seller_type': 'seller_type',
    'selling_price': 'selling_price',
    'min_cost_price': 'min_price', # sometimes exists
    'max_cost_price': 'max_price', # sometimes exists
    'vehicle_age': 'age'
}
df.rename(columns=rename_map, inplace=True)

print("Columns available:", df.columns.tolist())

# 3. EXTRACT BRAND & MODEL
print("Processing Brands & Models...")
# Ensure car_name is strictly string
df['car_name'] = df['car_name'].astype(str)

def get_brand(name):
    # Fix for simple strings
    return name.split(' ')[0].strip()

def get_model(name):
    parts = name.split(' ')
    if len(parts) > 1:
        return parts[1].strip()
    return "Unknown"

df['brand'] = df['car_name'].apply(get_brand)
df['model'] = df['car_name'].apply(get_model)

# Hierarchy Map
brand_to_models = df.groupby('brand')['model'].unique().to_dict()
for b in brand_to_models:
    brand_to_models[b] = list(brand_to_models[b])

# 4. CLEANING DATA
print("Cleaning Data...")

def clean_currency(x):
    if isinstance(x, str):
        # Remove units like 'kmpl', 'CC', ','
        clean_str = x.replace('kmpl', '').replace('km/kg', '').replace('CC', '').replace('bhp', '').replace(',', '')
        return clean_str.strip()
    return x

# Clean technical columns if they exist
tech_cols = ['mileage', 'engine', 'max_power']
for col in tech_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].apply(clean_currency), errors='coerce')
    else:
        # If missing, fill with mean logic later or drop
        print(f"⚠️ Warning: {col} missing, filling with 0")
        df[col] = 0.0

# Clean Text Categories
cat_cols = ['fuel', 'transmission', 'owner', 'seller_type']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    else:
        df[col] = "Unknown"

df.dropna(inplace=True)

# 5. ENCODING
print("Encoding Data...")
brand_map = df['brand'].value_counts(normalize=True).to_dict()
model_map = df['model'].value_counts(normalize=True).to_dict()

df['brand_encoded'] = df['brand'].map(brand_map)
df['model_encoded'] = df['model'].map(model_map)

# Drop raw text columns
drop_cols = ['car_name', 'brand', 'model', 'torque', 'min_price', 'max_price', 'age']
for c in drop_cols:
    if c in df.columns:
        df.drop(c, axis=1, inplace=True)

# 6. TRAIN
print("Training Model...")
df = pd.get_dummies(df, drop_first=True)

if 'selling_price' not in df.columns:
    print("❌ ERROR: Selling Price column lost. Check CSV headers.")
    exit()

X = df.drop('selling_price', axis=1)
y = df['selling_price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 7. SAVE
print("Saving to 'car_price_model.joblib'...")
save_data = {
    'model': model,
    'columns': X.columns,
    'brand_map': brand_map,
    'model_map': model_map,
    'brand_to_models': brand_to_models
}
joblib.dump(save_data, 'car_price_model.joblib')

print("✅ SUCCESS! Safe training complete.")