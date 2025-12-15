"""
Script untuk melatih model machine learning menggunakan MLflow Autolog.
Dataset: mybca_preprocessing.csv (sentimen analisis review aplikasi MyBCA)
Model: Random Forest Classifier
Tracking: MLflow lokal (tanpa hyperparameter tuning)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

# Set experiment name
mlflow.set_experiment("MyBCA_Sentiment_Analysis_Autolog")

# Enable autolog untuk scikit-learn
mlflow.sklearn.autolog()

# Load dataset
print("Loading dataset...")
df = pd.read_csv('mybca_preprocessing.csv')

# Menampilkan info dataset
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nDistribusi target (sentiment_encoded):")
print(df['sentiment_encoded'].value_counts())

# Memisahkan features dan target
# Menggunakan features numerik untuk prediksi sentiment
X = df[['score', 'thumbsUpCount', 'content_length', 'word_count', 'has_reply']]
y = df['sentiment_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Start MLflow run dengan autolog
with mlflow.start_run(run_name="RandomForest_Autolog"):
    
    # Log additional parameters manual (opsional)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Autolog akan otomatis log metrics, parameters, dan model
    # Tapi kita bisa tambahkan info tambahan
    print("\nModel training completed!")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Menampilkan classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Negatif', 'Netral', 'Positif']))
    
    print("\n" + "="*50)
    print("MLflow Autolog telah menyimpan:")
    print("- Model (Random Forest)")
    print("- Parameters (n_estimators, max_depth, dll)")
    print("- Metrics (accuracy, precision, recall, f1-score)")
    print("- Feature importance")
    print("="*50)
    
    print("\nUntuk melihat hasil tracking, jalankan:")
    print("mlflow ui")
    print("\nLalu buka browser di: http://localhost:5000")
