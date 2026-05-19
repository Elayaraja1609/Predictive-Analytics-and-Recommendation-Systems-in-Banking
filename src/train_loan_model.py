"""
Supervised learning: loan default prediction.
Models: Logistic Regression, Random Forest, Gradient Boosting.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import LOAN_FILE, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR


FEATURE_COLS = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "loan_term",
    "employment_years",
    "num_existing_loans",
    "savings_balance",
    "debt_to_income_ratio",
    "monthly_payment",
    "payment_to_income_ratio",
    "credit_utilization",
    "loan_to_income_ratio",
]
CATEGORICAL = ["credit_risk_band"]
TARGET = "repayment_status"


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(LOAN_FILE)
    df = df.dropna()
    X = df[FEATURE_COLS + CATEGORICAL]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def train_and_evaluate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, random_state=RANDOM_STATE
        ),
    }

    metrics = {}
    best_name, best_auc, best_pipe = None, -1.0, None

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        if name == "logistic_regression":
            param_grid = {"clf__C": [0.1, 1.0, 10.0]}
        elif name == "random_forest":
            param_grid = {"clf__n_estimators": [100, 150], "clf__max_depth": [8, 12]}
        else:
            param_grid = {"clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [100, 150]}

        search = GridSearchCV(pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        m = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "best_params": search.best_params_,
        }
        metrics[name] = m
        print(f"\n{name.upper()}: {m}")

        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            best_name = name
            best_pipe = pipe

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC={m['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"roc_{name}.png", dpi=120)
        plt.close()

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"confusion_{name}.png", dpi=120)
        plt.close()

    joblib.dump(best_pipe, MODELS_DIR / "loan_default_model.joblib")
    with open(REPORTS_DIR / "loan_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"models": metrics, "best_model": best_name, "best_roc_auc": best_auc}, f, indent=2)

    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f})")
    print(classification_report(y_test, best_pipe.predict(X_test), target_names=["No Default", "Default"]))
    return best_pipe, metrics


if __name__ == "__main__":
    train_and_evaluate()
