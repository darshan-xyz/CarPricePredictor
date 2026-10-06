import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. IMPORT THE 5 ALGORITHMS ---
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score
import kagglehub
import os
import glob
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

print("--- GENERATING 5-MODEL COMPARISON ---")

# --- 2. LOAD DATA ---
print("Loading Data...")
path = kagglehub.dataset_download("manishkr1754/cardekho-used-car-data")
csv_files = glob.glob(os.path.join(path, "*.csv"))

df = pd.read_csv(csv_files[0])
df.columns = df.columns.str.lower().str.strip()
rename_map = {'fuel_type': 'fuel', 'transmission_type': 'transmission', 'selling_price': 'selling_price', 
              'km_driven': 'km_driven', 'seller_type': 'seller_type', 'owner_type': 'owner', 'owner': 'owner'}
df.rename(columns=rename_map, inplace=True)
cols_to_drop = [c for c in df.columns if 'unnamed' in c]
if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)

# --- 3. CLEAN DATA ---
df['car_name'] = df['car_name'].astype(str)
df['brand'] = df['car_name'].apply(lambda x: x.split(' ')[0].strip())
df['model'] = df['car_name'].apply(lambda x: x.split(' ')[1].strip() if len(x.split(' ')) > 1 else "Unknown")

def clean_currency(x):
    if isinstance(x, str):
        return x.replace('kmpl', '').replace('km/kg', '').replace('CC', '').replace('bhp', '').replace(',', '').strip()
    return x

for col in ['mileage', 'engine', 'max_power']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].apply(clean_currency), errors='coerce')

df.dropna(inplace=True)

available_cols = df.columns.tolist()
encode_cols = [c for c in ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model'] if c in available_cols]
df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
df.drop(['car_name', 'torque'], axis=1, errors='ignore', inplace=True)

# --- 4. SCALE & SPLIT ---
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Scaling is REQUIRED for KNN/SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 5. TRAIN 5 MODELS ---
print("Training models...")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
}

results = []

for name, model in models.items():
    print(f"  -> Running {name}...")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "R2 Score": r2})
    except Exception as e:
        print(f"Error in {name}: {e}")

results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

# --- 6. PLOT COMPARISON ---
print("Generating 5-Model Bar Chart...")
plt.figure(figsize=(12, 6))

# FIX 1: Added hue="Model" and legend=False to fix deprecated warning
ax = sns.barplot(x="Model", y="R2 Score", data=results_df, palette="viridis", hue="Model", legend=False)
plt.title("Model Performance Comparison (Accuracy)", fontsize=16)
plt.ylabel("R² Score (Higher is Better)")
plt.ylim(0, 1.1)
plt.xticks(rotation=15)

for i, v in enumerate(results_df['R2 Score']):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("static/model_comparison.png")
print("✅ Saved 'model_comparison.png'")

# --- 7. PLOT SCATTER (Best Model) ---
print("Generating Scatter Plot...")
plt.figure(figsize=(8, 8))

best_model_name = results_df.iloc[0]['Model']
best_model_instance = models[best_model_name]
y_pred_best = best_model_instance.predict(X_test)

sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.6, color="royalblue")

p1 = max(max(y_pred_best), max(y_test))
p2 = min(min(y_pred_best), min(y_test))
plt.plot([p1, p2], [p1, p2], 'r--', linewidth=2, label=f"Perfect Prediction")

plt.title(f"Actual vs. Predicted Price ({best_model_name})", fontsize=16)

# FIX 2: Changed (₹) to (INR) to prevent missing font warning
plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")
plt.legend()
plt.grid(True)

plt.savefig("static/prediction_scatter.png")
print("✅ Saved 'prediction_scatter.png'")