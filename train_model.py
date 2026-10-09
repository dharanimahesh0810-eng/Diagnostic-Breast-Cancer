"""
train_model.py
----------------
Trains the Logistic Regression model on the Diagnostic Breast Cancer Dataset
and saves it (along with the feature column order) to disk so the Streamlit
app can load it instantly without retraining every time.

Run this ONCE before starting the Streamlit app:
    python train_model.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Load dataset
#    Place "Diagnostic Breast Cancer Dataset.csv" in this same folder
# ---------------------------------------------------------
df = pd.read_csv("Diagnostic Breast Cancer Dataset.csv")

# ---------------------------------------------------------
# 2. Encode target (Benign -> 0, Malignant -> 1)
# ---------------------------------------------------------
le = LabelEncoder()
df["Diagnosis"] = le.fit_transform(df["Diagnosis"])

# ---------------------------------------------------------
# 3. Features / target split
# ---------------------------------------------------------
X = df.drop(["Diagnosis", "ID"], axis=1)
y = df["Diagnosis"]

feature_columns = list(X.columns)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ---------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------
model = make_pipeline(LogisticRegression(max_iter=5000))
model.fit(x_train, y_train)

# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------
prediction = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))

# ---------------------------------------------------------
# 6. Save model + feature column order + label encoder classes
# ---------------------------------------------------------
joblib.dump(model, "breast_cancer_model.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(le.classes_, "label_classes.pkl")

print("\nSaved: breast_cancer_model.pkl, feature_columns.pkl, label_classes.pkl")
