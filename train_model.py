import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset (or start empty and append collected data later)
df = pd.read_csv("health_data.csv")  # columns: Diarrhea, Vomiting, Fever, AbdominalPain, Dehydration, RiskLevel

X = df[["Diarrhea", "Vomiting", "Fever", "AbdominalPain", "Dehydration"]]
y = df["RiskLevel"]  # 'Low', 'Medium', 'High', 'Critical'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "health_model.pkl")
print("Model trained and saved!")
