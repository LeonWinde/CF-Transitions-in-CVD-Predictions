import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent         
    project_root = script_dir.parent                      
    data_path = project_root / "data" / "ess_model_ready.csv"

    df = pd.read_csv(data_path)

    print("Loaded dataset:", data_path)
    print("Shape:", df.shape)
    print()

    target_col = "cvd_any"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("Number of features:", X.shape[1])
    print("Target distribution:")
    print(y.value_counts(normalize=True))
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])
    print()

    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1   
    )

    rf_model.fit(X_train, y_train)

    print("Random Forest model trained.")
    print()

  
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("Model performance on test set:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC AUC:  {roc_auc:.3f}")
    print()

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print()

  
    model_path = project_root / "models" / "rf_cvd.pkl"
    joblib.dump(rf_model, model_path)

    print(f"Saved trained Random Forest model to: {model_path}")
    print("\n Random Forest training completed.")


if __name__ == "__main__":
    main()
