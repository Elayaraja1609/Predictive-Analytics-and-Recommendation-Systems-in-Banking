"""
Recommendation engine: item-based collaborative filtering + content similarity.
Evaluation: leave-one-out Precision@K, Recall@K, MAP, NDCG.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.config import INTERACTIONS_FILE, MODELS_DIR, PRODUCTS_FILE, RANDOM_STATE, REPORTS_DIR

INTERACTION_WEIGHTS = {"purchased": 5.0, "applied": 4.0, "clicked": 2.0, "viewed": 1.0}
TOP_K = 5
MIN_INTERACTIONS = 3


def build_interaction_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(INTERACTIONS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)
    interactions["weight"] = interactions["interaction_type"].map(INTERACTION_WEIGHTS)
    matrix = interactions.pivot_table(
        index="customer_id",
        columns="product_id",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )
    return matrix, products


def item_cf_similarity(matrix: pd.DataFrame) -> np.ndarray:
    item_matrix = matrix.values.T
    return cosine_similarity(item_matrix)


def content_similarity(products: pd.DataFrame, prod_ids: list) -> np.ndarray:
    products = products.set_index("product_id").loc[prod_ids].reset_index()
    cat_dummies = pd.get_dummies(products["category"])
    fee_scaled = MinMaxScaler().fit_transform(products[["annual_fee", "min_balance"]])
    features = np.hstack([cat_dummies.values, fee_scaled])
    return cosine_similarity(features)


def score_candidates(
    user_vector: np.ndarray,
    item_sim: np.ndarray,
    content_sim: np.ndarray,
    beta: float = 0.3,
) -> np.ndarray:
    cf_scores = user_vector @ item_sim
    content_scores = np.zeros(len(user_vector))
    interacted = np.where(user_vector > 0)[0]
    for idx in interacted:
        content_scores += content_sim[idx] * user_vector[idx]
    scores = cf_scores + beta * content_scores
    scores[user_vector > 0] = -np.inf
    return scores


def recommend(
    user_vector: np.ndarray,
    item_sim: np.ndarray,
    content_sim: np.ndarray,
    prod_ids: list,
    top_k: int = TOP_K,
) -> list[str]:
    scores = score_candidates(user_vector, item_sim, content_sim)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [prod_ids[i] for i in top_indices]


def leave_one_out_evaluate(
    matrix: pd.DataFrame,
    item_sim: np.ndarray,
    content_sim: np.ndarray,
    top_k: int = TOP_K,
) -> dict:
    prod_ids = list(matrix.columns)
    precisions, recalls, maps, ndcgs = [], [], [], []

    for cust_idx in range(len(matrix)):
        row = matrix.iloc[cust_idx].values
        positives = np.where(row > 0)[0]
        if len(positives) < MIN_INTERACTIONS:
            continue

        for test_idx in positives:
            train_row = row.copy()
            train_row[test_idx] = 0
            if train_row.sum() == 0:
                continue

            recs = recommend(train_row, item_sim, content_sim, prod_ids, top_k)
            rec_indices = [prod_ids.index(r) for r in recs]
            hit = int(test_idx in rec_indices)

            precisions.append(hit / top_k)
            recalls.append(hit)

            ap = 0.0
            dcg = 0.0
            for rank, ri in enumerate(rec_indices, start=1):
                rel = 1 if ri == test_idx else 0
                dcg += rel / np.log2(rank + 1)
                if rel:
                    ap = 1.0 / rank
                    break
            idcg = 1.0 / np.log2(1 + 1)
            maps.append(ap)
            ndcgs.append(dcg / idcg)

    n = len(precisions)
    return {
        f"precision_at_{top_k}": round(float(np.mean(precisions)), 4) if n else 0.0,
        f"recall_at_{top_k}": round(float(np.mean(recalls)), 4) if n else 0.0,
        "map": round(float(np.mean(maps)), 4) if n else 0.0,
        "ndcg": round(float(np.mean(ndcgs)), 4) if n else 0.0,
        "evaluated_holdouts": n,
    }


def generate_all_recommendations(
    matrix: pd.DataFrame,
    item_sim: np.ndarray,
    content_sim: np.ndarray,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    prod_ids = list(matrix.columns)
    rows = []
    for cust_idx, cust_id in enumerate(matrix.index):
        recs = recommend(matrix.iloc[cust_idx].values, item_sim, content_sim, prod_ids, top_k)
        for rank, pid in enumerate(recs, start=1):
            rows.append({"customer_id": cust_id, "rank": rank, "product_id": pid})
    return pd.DataFrame(rows)


def train_and_evaluate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    matrix, products = build_interaction_matrix()
    prod_ids = list(matrix.columns)
    item_sim = item_cf_similarity(matrix)
    content_sim = content_similarity(products, prod_ids)

    metrics = leave_one_out_evaluate(matrix, item_sim, content_sim)
    recommendations = generate_all_recommendations(matrix, item_sim, content_sim)
    recommendations = recommendations.merge(
        products[["product_id", "product_name", "category"]],
        on="product_id",
    )
    recommendations.to_csv(REPORTS_DIR / "product_recommendations.csv", index=False)

    joblib.dump(
        {
            "item_sim": item_sim,
            "content_sim": content_sim,
            "product_ids": prod_ids,
        },
        MODELS_DIR / "recommendation_model.joblib",
    )

    with open(REPORTS_DIR / "recommendation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Recommendation metrics:", metrics)
    print(recommendations.head(10))
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
