from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
CORS(app)

MODEL_PATH = "health_model_xgb_dynamic.pkl"

# Risk recommendations
RISK_RECOMMENDATIONS = {
    "Low": {
        "recommendation": "Maintain hygiene, drink clean water, and monitor health.",
        "actions": ["Boil water before drinking", "Wash hands regularly"]
    },
    "Medium": {
        "recommendation": "Take preventive measures and consult local health worker if symptoms persist.",
        "actions": ["Use water purifier", "Avoid street food", "Monitor family members"]
    },
    "High": {
        "recommendation": "Immediate attention required. Inform health authorities.",
        "actions": ["Seek medical help", "Isolate affected individuals", "Test water sources"]
    },
    "Critical": {
        "recommendation": "Emergency! High risk of outbreak. Mobilize medical and sanitation response.",
        "actions": ["Call health authorities", "Provide medical aid", "Disinfect water sources", "Evacuate vulnerable households"]
    }
}

# Train or load model
def train_or_load_model():
    global model, le
    if os.path.exists(MODEL_PATH):
        model, le = joblib.load(MODEL_PATH)
        print("✅ Loaded trained XGBoost model from disk")
    else:
        print("⚡ Training new dynamic XGBoost model...")
        # Generate synthetic diverse dataset for symptoms + water quality
        np.random.seed(42)
        X_train = []
        y_train = []
        for _ in range(200):
            # Symptoms (0 or 1)
            symptoms = np.random.randint(0, 2, 8).tolist()
            # Water quality features
            pH = np.random.uniform(6.0, 9.0)
            turbidity = np.random.uniform(0, 100)
            coliform = np.random.randint(0, 500)
            X_train.append(symptoms + [pH, turbidity, coliform])
            # Risk assignment logic
            score = sum(symptoms) + (turbidity > 50) + (coliform > 100) + (pH < 6.5 or pH > 8.5)
            if score <= 2:
                y_train.append("Low")
            elif score <= 4:
                y_train.append("Medium")
            elif score <= 6:
                y_train.append("High")
            else:
                y_train.append("Critical")

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)

        model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42
        )
        model.fit(X_train, y_encoded)
        joblib.dump((model, le), MODEL_PATH)
        print("✅ Dynamic XGBoost model trained and saved")

train_or_load_model()

# Helper: make prediction
def make_prediction(record):
    # Features
    features = [
        record.get("Diarrhea", 0),
        record.get("Vomiting", 0),
        record.get("Fever", 0),
        record.get("AbdominalPain", 0),
        record.get("Dehydration", 0),
        record.get("Nausea", 0),
        record.get("Headache", 0),
        record.get("Fatigue", 0),
        float(record.get("pH", 7.0)),
        float(record.get("Turbidity", 10.0)),
        int(record.get("Coliform", 10))
    ]

    pred_encoded = model.predict([features])[0]
    risk_level = le.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba([features])[0]
    probability = int(max(proba) * 100)

    # Only include symptoms that are present
    factors = [symptom for symptom in ["Diarrhea","Vomiting","Fever","AbdominalPain",
               "Dehydration","Nausea","Headache","Fatigue"] if record.get(symptom, 0) == 1]

    recommendation_data = RISK_RECOMMENDATIONS.get(risk_level, {
        "recommendation": "Follow general preventive measures",
        "actions": []
    })

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "location": record.get("location", "Unknown"),
        "waterSource": record.get("waterSource", "Unknown"),
        "sanitationAccess": record.get("sanitationAccess", "Unknown"),
        "disease": "Waterborne Disease Risk",
        "riskLevel": risk_level,
        "probability": probability,
        "factors": factors,
        "recommendation": recommendation_data["recommendation"],
        "actions": recommendation_data["actions"],
        "affectedHouseholds": record.get("affectedHouseholds", 1)
    }


# Routes
@app.route("/predict", methods=["GET"])
def predict_get():
    return "Use POST method with JSON data or upload a CSV file for predictions."

@app.route("/predict", methods=["POST"])
def predict_json():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        predictions = [make_prediction(record) for record in data]
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        df = pd.read_csv(file)

        required_cols = ["Diarrhea", "Vomiting", "Fever", "AbdominalPain",
                         "Dehydration", "Nausea", "Headache", "Fatigue",
                         "pH", "Turbidity", "Coliform"]
        for col in required_cols:
            if col not in df.columns:
                return jsonify({"error": f"Missing required column: {col}"}), 400

        predictions = [make_prediction(row.to_dict()) for _, row in df.iterrows()]
        results_df = df.copy()
        results_df["RiskLevel"] = [p["riskLevel"] for p in predictions]
        results_df["Probability"] = [p["probability"] for p in predictions]
        results_df["Recommendation"] = [p["recommendation"] for p in predictions]

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        results_df.to_csv(tmp_file.name, index=False)
        return send_file(tmp_file.name, as_attachment=True, download_name="predicted_analysis.csv", mimetype="text/csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's port if available
    app.run(debug=True, host="0.0.0.0", port=port)

