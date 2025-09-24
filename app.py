from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
from flask import Flask, request, jsonify, send_file
import tempfile 

app = Flask(__name__)
CORS(app)

MODEL_PATH = "health_model.pkl"

# Define risk recommendations
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
        print("✅ Loaded trained model from disk")
    else:
        print("⚡ Training new model...")
        # Example training data (you can replace with real dataset)
        X_train = [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 1]
        ]
        y_train = ["Medium", "High", "Critical", "Low", "Low"]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_encoded)

        joblib.dump((model, le), MODEL_PATH)
        print("✅ Model trained and saved")

train_or_load_model()

# Helper function to make prediction
def make_prediction(record):
    features = [
        record.get("Diarrhea", 0),
        record.get("Vomiting", 0),
        record.get("Fever", 0),
        record.get("AbdominalPain", 0),
        record.get("Dehydration", 0),
        record.get("Nausea", 0),
        record.get("Headache", 0),
        record.get("Fatigue", 0),
    ]

    pred_encoded = model.predict([features])[0]
    risk_level = le.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba([features])[0]
    probability = int(max(proba) * 100)

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
        "factors": [k for k, v in record.items() if v == 1],
        "recommendation": recommendation_data["recommendation"],
        "actions": recommendation_data["actions"],
        "affectedHouseholds": record.get("affectedHouseholds", 1)
    }

# GET test route
@app.route("/predict", methods=["GET"])
def predict_get():
    return "Use POST method with JSON data or upload a CSV file for predictions."

# POST: JSON input
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

# POST: CSV upload
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
                         "Dehydration", "Nausea", "Headache", "Fatigue"]
        for col in required_cols:
            if col not in df.columns:
                return jsonify({"error": f"Missing required column: {col}"}), 400

        predictions = [make_prediction(row.to_dict()) for _, row in df.iterrows()]

        results_df = df.copy()
        results_df["RiskLevel"] = [p["riskLevel"] for p in predictions]
        results_df["Probability"] = [p["probability"] for p in predictions]
        results_df["Recommendation"] = [p["recommendation"] for p in predictions]

        # Use temporary file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        results_df.to_csv(tmp_file.name, index=False)

        return send_file(tmp_file.name, as_attachment=True, download_name="predicted_analysis.csv", mimetype="text/csv")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

