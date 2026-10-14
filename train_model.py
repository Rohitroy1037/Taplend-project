from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "AssessmentDataScientist.xlsx"
MODEL_PATH = BASE_DIR / "best_loan_model.joblib"
FEATURES_PATH = BASE_DIR / "model_features.joblib"


def train_and_save_model():
    application_df = pd.read_excel(DATA_PATH, sheet_name="Application Data", engine="openpyxl")
    loan_performance_df = pd.read_excel(DATA_PATH, sheet_name="Loan Performance", engine="openpyxl")

    application_df["customer_id_clean"] = (
        application_df["customer_id"].astype(str).str.strip().str.upper()
    )
    loan_performance_df["customer_id_clean"] = (
        loan_performance_df["idLoan"].astype(str).str.split("-").str[0].str.strip().str.upper()
    )

    application_df["payment_amount_approved"] = application_df["payment_amount_approved"].fillna(
        application_df["payment_amount"]
    )
    application_df["other_phone_type"] = application_df["other_phone_type"].fillna("None")
    application_df["how_use_money"] = application_df["how_use_money"].fillna("Unknown")
    application_df["bank_account_duration"] = application_df["bank_account_duration"].fillna("Unknown")
    application_df["age_at_application"] = (
        (application_df["application_when"] - application_df["birth_date"]).dt.days // 365
    )

    merged_df = pd.merge(application_df, loan_performance_df, on="customer_id_clean", how="inner")
    features_df = merged_df.copy()
    features_df["target"] = features_df["flgGood"].map({"Good": 1, "Bad": 0})

    columns_to_drop = [
        "customer_id",
        "customer_id_clean",
        "idLoan",
        "flgGood",
        "target",
        "status",
        "birth_date",
        "application_when",
        "email",
    ]

    y = features_df["target"]
    X_raw = features_df.drop(columns=columns_to_drop)
    categorical_cols = X_raw.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model validation accuracy: {accuracy:.4f}")
    print(f"Features count: {len(X_encoded.columns)}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(X_encoded.columns.tolist(), FEATURES_PATH)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved feature list to: {FEATURES_PATH}")

    return model, X_encoded.columns.tolist()


if __name__ == "__main__":
    train_and_save_model()
