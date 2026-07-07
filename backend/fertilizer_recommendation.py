import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Load dataset
print("Loading dataset...")
data = pd.read_csv("Dataset/fertilizer_recommendation.csv")

# Feature columns
X = data[['Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level', 'Temperature', 'Humidity', 'Rainfall', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Previous_Crop']]
y = data['Recommended_Fertilizer']

# Encoding Features
print("Encoding categorical features...")
encoder = OrdinalEncoder(
    categories=[
        sorted(X["Soil_Type"].unique()),
        sorted(X["Crop_Type"].unique()),
        ["Sowing", "Vegetative", "Flowering", "Harvest"],
        ["Kharif", "Rabi", "Zaid"],
        sorted(X["Previous_Crop"].unique())
    ]
)

cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Previous_Crop"]
X[cat_cols] = encoder.fit_transform(X[cat_cols])

# Encoding Target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Splitting data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Model Building
print("Training XGBoost Classifier (this may take a moment)...")
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

# Evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save models and encoders
print("Saving model and encoders to .pkl files...")
joblib.dump(model, "fertilizer_model.pkl")
joblib.dump(encoder, "fertilizer_encoder.pkl")
joblib.dump(target_encoder, "fertilizer_target_encoder.pkl")

print("\nModel saved as fertilizer_model.pkl")
print("Encoder saved as fertilizer_encoder.pkl")
print("Target Encoder saved as fertilizer_target_encoder.pkl")
print("Done!")
