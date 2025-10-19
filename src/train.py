# src/train.py
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import os

def train_model():
    # Load iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/iris_model.pkl')
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save metrics
    metrics = {'accuracy': accuracy, 'test_samples': len(y_test)}
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    return model, metrics

if __name__ == "__main__":
    train_model()
