"""
Script untuk melatih model machine learning menggunakan MLflow Autolog.
Dataset: mybca_preprocessing.csv (sentimen analisis review aplikasi MyBCA)
Model: Random Forest Classifier
Tracking: MLflow via MLflow Project (CI-safe)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# =====================================================
# ENABLE AUTOLOG (TANPA start_run & set_experiment)
# =====================================================
mlflow.sklearn.autolog()

# =====================================================
# LOAD DATASET (PREPROCESSED)
# =====================================================
print("Loading dataset...")
df = pd.read_csv("mybca_preprocessing.csv")

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nDistribusi target (sentiment_encoded):")
print(df["sentiment_encoded"].value_counts())

# =====================================================
# FEATURES & TARGET
# =====================================================
X = df[["score", "thumbsUpCount", "content_length", "word_count", "has_reply"]]
y = df["sentiment_encoded"]

# =====================================================
# SPLIT DATA
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# =====================================================
# TRAIN MODEL
# =====================================================
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nModel training completed!")
print(f"Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Negatif", "Netral", "Positif"]
))

print("\nMLflow Autolog telah menyimpan:")
print("- Model")
print("- Parameters")
print("- Metrics")
print("- Artifacts (model, environment, dll)")
