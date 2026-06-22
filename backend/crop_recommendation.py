import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Dataset/Crop_recommendation_Dataset.csv")

# Label Encoding
le = LabelEncoder()
data['crop'] = le.fit_transform(data['label'])

# Data splitting
X = data.drop(columns=['label', 'crop'])
y = data['crop']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model Building
forest_model = RandomForestClassifier(n_estimators=200, random_state=42)
forest_model.fit(X_train, y_train)


# Model Evaluation
y_pred = forest_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model + encoder
joblib.dump(forest_model, "crop_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel saved as crop_model.pkl")
print("Label Encoder saved as label_encoder.pkl")