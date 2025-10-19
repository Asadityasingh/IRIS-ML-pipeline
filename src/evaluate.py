# src/evaluate.py
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

def evaluate_model(model_path='models/iris_model.pkl'):
    # Load model
    model = joblib.load(model_path)
    
    # Load iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Split data (same split as training for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
    }
    
    # Save evaluation results
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/evaluation.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

if __name__ == "__main__":
    metrics = evaluate_model()
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
