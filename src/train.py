import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Allow importing from src/ when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import preprocess_data


def evaluate_model(model, X_test, y_test):
    """Return a dict of key classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
    }


def get_feature_importances(model, feature_names):
    """
    Extract feature importances from the model.
    - Tree-based models expose .feature_importances_
    - Logistic Regression exposes .coef_ (use absolute coefficients)
    Returns a DataFrame sorted by importance descending.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    return importance_df


def main():
    # ------------------------------------------------------------------
    # 1. Preprocess the data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading and preprocessing data ...")
    print("=" * 60)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data()

    # ------------------------------------------------------------------
    # 2. Define the three models (all use class_weight='balanced' and
    #    random_state=42 for reproducibility and handling class imbalance)
    # ------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_estimators=100,
        ),
    }

    # ------------------------------------------------------------------
    # 3. Train each model and evaluate on the test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2 & 3: Training and evaluating models ...")
    print("=" * 60)

    results = {}          # model_name -> metrics dict
    trained_models = {}   # model_name -> fitted model object

    for name, model in models.items():
        print(f"  Training {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model
        print("done.")

    # ------------------------------------------------------------------
    # 4. Print a clean comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Model Comparison")
    print("=" * 60)

    metrics_order = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    comparison_df = pd.DataFrame(results, index=metrics_order).T
    comparison_df.index.name = "Model"
    print(comparison_df.to_string())

    # ------------------------------------------------------------------
    # 5. Select the best model based on F1 Score
    # ------------------------------------------------------------------
    best_model_name = max(results, key=lambda name: results[name]["F1 Score"])
    best_model = trained_models[best_model_name]
    best_f1 = results[best_model_name]["F1 Score"]

    print(f"\n{'=' * 60}")
    print(f"Step 5: Best model selected -> '{best_model_name}'  (F1 = {best_f1})")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 6. Save the best model to models/best_model.pkl
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to '{model_path}'")

    # ------------------------------------------------------------------
    # 7. Save feature importances to models/feature_importance.csv
    # ------------------------------------------------------------------
    importance_df = get_feature_importances(best_model, feature_names)
    importance_path = "models/feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importances saved to '{importance_path}'")

    # ------------------------------------------------------------------
    # 8. Final confirmation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Selected model : {best_model_name}")
    print(f"  F1 Score       : {best_f1}")
    print(f"  Model file     : {model_path}")
    print(f"  Importances    : {importance_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
