from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import json
import numpy as np

app = Flask(__name__)

# --- 1. LOAD MODEL & DATA MAPS ---
try:
    data = joblib.load('car_price_model.joblib')
    model = data['model']
    feature_columns = data['columns']
    brand_map = data['brand_map']
    model_map = data['model_map']
    brand_to_models = data['brand_to_models']
    print("[OK] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    print("Run 'python train_local.py' first!")
    exit()

@app.route('/')
def home():
    # Pass brand list and hierarchy to frontend
    return render_template('index.html', 
                         brands=sorted(list(brand_map.keys())), 
                         car_hierarchy=json.dumps(brand_to_models))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.json
        
        # --- 2. PREPARE INPUTS ---
        # Map text inputs to encoded numbers (Default to 0 if unknown)
        brand_val = brand_map.get(req['brand'], 0)
        model_val = model_map.get(req['model'], 0)

        # Create input dictionary
        input_data = {
            'year': int(req['year']),
            'km_driven': int(req['km_driven']),
            'mileage': float(req['mileage']),
            'engine': float(req['engine']),
            'max_power': float(req['max_power']),
            'seats': int(req['seats']),
            'brand_encoded': brand_val,
            'model_encoded': model_val,
            
            # One-Hot Encoding Flags (Set strictly to 1)
            f"fuel_{req['fuel']}": 1,
            f"seller_type_{req['seller_type']}": 1,
            f"transmission_{req['transmission']}": 1,
            f"owner_{req['owner']}": 1
        }

        # Create DataFrame with correct columns
        df_input = pd.DataFrame(columns=feature_columns)
        
        # Initialize with 0.0 (Float) to prevent decimal crashes
        df_input.loc[0] = 0.0  
        
        # Fill data
        for col, val in input_data.items():
            if col in df_input.columns:
                df_input.at[0, col] = val
        
        # --- 3. GET AI PREDICTION ---
        base_prediction = model.predict(df_input)[0]

        # --- 4. BUSINESS LOGIC LAYER (Manual Adjustments) ---
        final_price = base_prediction

        # A. OWNER LOGIC (Depreciation for multiple owners)
        if req['owner'] == "Second Owner":
            final_price *= 0.95  # -5%
        elif req['owner'] == "Third Owner":
            final_price *= 0.90  # -10%
        elif req['owner'] == "Fourth & Above Owner":
            final_price *= 0.85  # -15%

        # B. YEAR LOGIC (Depreciation Curve)
        # Baseline: 2018. 
        # Older = Cheaper (-5% per year). Newer = More Expensive (+5% per year).
        year_diff = int(req['year']) - 2018
        year_factor = 1 + (year_diff * 0.05)
        
        # Safety limits (Price can't drop below 10% or double excessively)
        if year_factor < 0.1: year_factor = 0.1
        if year_factor > 2.0: year_factor = 2.0
        
        final_price *= year_factor

        # C. MILEAGE LOGIC (Fuel Efficiency)
        # Baseline: 18 kmpl.
        # Below 18 (Gas Guzzler) -> Price Drops (-1% per unit)
        # Above 18 (Efficient)   -> Price Rises (+1% per unit)
        mileage_diff = float(req['mileage']) - 18.0
        mileage_factor = 1 + (mileage_diff * 0.01)

        # Safety limits
        if mileage_factor < 0.8: mileage_factor = 0.8
        if mileage_factor > 1.2: mileage_factor = 1.2

        final_price *= mileage_factor

        # --- 5. RETURN RESULT ---
        return jsonify({'price': f"{final_price:,.2f}"})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)