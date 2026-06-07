# app_mumbai.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

MODEL_PATH = os.path.join('model', 'mumbai_price_model.pkl')
DATA_PATH = os.path.join('data', 'Mumbai House Prices.csv')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run train_mumbai.py first to create model/mumbai_price_model.pkl")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found at data/Mumbai House Prices.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
df['locality'] = df['locality'].apply(lambda x: x.strip() if isinstance(x, str) else x)
loc_counts = df['locality'].value_counts()
locations = sorted([loc for loc, c in loc_counts.items() if c > 10])
# include 'other' because training grouped rare ones into other
locations.insert(0, 'other')

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    loc = request.form.get('locality')
    area = request.form.get('area')
    bhk = request.form.get('bhk')
    age = request.form.get('age')

    # basic validation & conversions
    try:
        area_val = float(str(area).replace(',', '').strip())
    except:
        return render_template('index.html', locations=locations, error="Invalid area value")

    try:
        bhk_val = int(float(bhk))
    except:
        return render_template('index.html', locations=locations, error="Invalid BHK value")

    # prepare DataFrame with same columns as training features
    row = {'locality_clean': loc, 'area_sqft': area_val, 'bhk': bhk_val}
    if 'age_num' in model.named_steps['pre'].transformers_[0][2] or 'age_num' in df.columns:
        # if model expects age, try to include it; otherwise default to median
        try:
            age_val = float(age) if age is not None and age != '' else None
        except:
            age_val = None
        # if None, set to 0
        row['age_num'] = age_val if age_val is not None else 0

    X = pd.DataFrame([row])
    pred_lakhs = model.predict(X)[0]
    pred_inr = pred_lakhs * 100000.0
    readable = f"Estimated price: {pred_lakhs:.2f} lakhs (₹{pred_inr:,.0f})"
    return render_template('index.html', locations=locations, prediction_text=readable)

# JSON API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    loc = data.get('locality')
    area = float(data.get('area'))
    bhk = int(data.get('bhk'))
    row = {'locality_clean': loc, 'area_sqft': area, 'bhk': bhk}
    X = pd.DataFrame([row])
    pred_lakhs = model.predict(X)[0]
    return jsonify({'estimated_price_lakhs': float(pred_lakhs), 'estimated_price_inr': float(pred_lakhs * 100000)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)