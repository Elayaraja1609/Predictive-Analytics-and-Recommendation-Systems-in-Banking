"""
Unsupervised learning: customer segmentation from transaction behavior.
Algorithm: K-Means with silhouette-based k selection; PCA visualization.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TRANSACTIONS_FILE


def build_customer_features() -> pd.DataFrame:
    df = pd.read_csv(TRANSACTIONS_FILE)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    agg = df.groupby("customer_id").agg(
        txn_count=("transaction_id", "count"),
        total_amount=("transaction_amount", "sum"),
        avg_amount=("transaction_amount", "mean"),
        std_amount=("transaction_amount", "std"),
        max_amount=("transaction_amount", "max"),
        deposit_count=("transaction_type", lambda x: (x == "deposit").sum()),
        withdrawal_count=("transaction_type", lambda x: (x == "withdrawal").sum()),
        transfer_count=("transaction_type", lambda x: (x == "transfer").sum()),
        last_txn_date=("transaction_date", "max"),
        first_txn_date=("transaction_date", "min"),
    ).reset_index()

    agg["std_amount"] = agg["std_amount"].fillna(0)
    agg["active_days"] = (agg["last_txn_date"] - agg["first_txn_date"]).dt.days + 1
    agg["txn_frequency"] = agg["txn_count"] / agg["active_days"].clip(lower=1)
    agg["deposit_ratio"] = agg["deposit_count"] / agg["txn_count"].clip(lower=1)
    agg["withdrawal_ratio"] = agg["withdrawal_count"] / agg["txn_count"].clip(lower=1)
    agg["avg_daily_spend"] = agg["total_amount"] / agg["active_days"].clip(lower=1)

    feature_cols = [
        "txn_count",
        "total_amount",
        "avg_amount",
        "std_amount",
        "max_amount",
        "txn_frequency",
        "deposit_ratio",
        "withdrawal_ratio",
        "avg_daily_spend",
    ]
    return agg, feature_cols


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> int:
    best_k, best_score = 2, -1.0
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        scores[k] = sil
        if sil > best_score:
            best_score = sil
            best_k = k
    return best_k, scores


def train_and_evaluate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    customer_df, feature_cols = build_customer_features()
    X = customer_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    optimal_k, k_scores = find_optimal_k(X_scaled)
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=15)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    customer_df["cluster"] = labels

    cluster_summary = (
        customer_df.groupby("cluster")[feature_cols]
        .mean()
        .round(2)
        .reset_index()
    )
    cluster_summary["segment_size"] = customer_df.groupby("cluster").size().values

    segment_names = {
        0: "Low Activity",
        1: "High Value",
        2: "Frequent Small",
        3: "Balanced",
        4: "Withdrawal Heavy",
    }
    customer_df["segment_label"] = customer_df["cluster"].map(
        lambda c: segment_names.get(c, f"Segment {c}")
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    customer_df["pca1"], customer_df["pca2"] = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=customer_df,
        x="pca1",
        y="pca2",
        hue="cluster",
        palette="tab10",
        alpha=0.6,
        s=40,
    )
    plt.title(f"Customer Segments (K={optimal_k}, Silhouette={sil:.3f})")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "segmentation_pca.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(list(k_scores.keys()), list(k_scores.values()))
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("K Selection via Silhouette Score")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "silhouette_by_k.png", dpi=120)
    plt.close()

    joblib.dump(kmeans, MODELS_DIR / "segmentation_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "segmentation_scaler.joblib")
    customer_df.to_csv(REPORTS_DIR / "customer_segments.csv", index=False)

    metrics = {
        "optimal_k": optimal_k,
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(dbi, 4),
        "k_scores": {str(k): round(v, 4) for k, v in k_scores.items()},
        "cluster_summary": cluster_summary.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "segmentation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Optimal K={optimal_k} | Silhouette={sil:.4f} | Davies-Bouldin={dbi:.4f}")
    print(cluster_summary)
    return customer_df, metrics


if __name__ == "__main__":
    train_and_evaluate()
