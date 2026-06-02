import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(data_path="data/telco_churn.csv", scaler_path="models/scaler.pkl"):
    """
    Load, clean, encode, split, and scale the Telco Churn dataset.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """

    # ------------------------------------------------------------------
    # 1. Load the dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path)

    # ------------------------------------------------------------------
    # 2. Drop the customerID column (not a predictive feature)
    # ------------------------------------------------------------------
    df.drop(columns=["customerID"], inplace=True)

    # ------------------------------------------------------------------
    # 3. Convert TotalCharges to numeric
    #    The column contains spaces that pandas reads as strings;
    #    coerce forces those invalid entries to NaN so they can be dropped.
    # ------------------------------------------------------------------
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # ------------------------------------------------------------------
    # 4. Drop rows with any missing values (includes coerced NaNs above)
    # ------------------------------------------------------------------
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 5. Encode the target column: Yes -> 1, No -> 0
    # ------------------------------------------------------------------
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # ------------------------------------------------------------------
    # 6. Label-encode all remaining categorical (object) columns
    #    Each column gets its own LabelEncoder fitted on that column's values.
    # ------------------------------------------------------------------
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # ------------------------------------------------------------------
    # 7. Split features (X) and target (y)
    # ------------------------------------------------------------------
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_names = X.columns.tolist()

    # ------------------------------------------------------------------
    # 8. Split into train / test sets (80 % train, 20 % test)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 9. Scale numeric features using StandardScaler
    #    Fit only on training data to prevent data leakage.
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 10. Persist the fitted scaler so it can be reused at inference time
    # ------------------------------------------------------------------
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to '{scaler_path}'")

    print(f"Preprocessing complete — train size: {len(X_train)}, test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = preprocess_data()
    print(f"Features ({len(feature_names)}): {feature_names}")
