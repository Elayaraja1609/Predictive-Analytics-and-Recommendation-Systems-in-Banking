# Final Project Report — Banking Analytics

## Executive Summary

This project delivers three ML solutions for a fictional retail bank using synthetic data generated with Python Faker. The pipeline is fully reproducible and deployed via Streamlit.

## 1. Loan Default Prediction (Supervised)

**Objective:** Predict whether a customer will default (`repayment_status = 1`).

**Approach:**
- Feature engineering: debt-to-income, payment-to-income, credit risk band
- Models compared: Logistic Regression, Random Forest, Gradient Boosting
- Hyperparameter tuning via GridSearchCV (ROC-AUC)

**Business value:** Improve loan approval decisions and reduce NPAs.

## 2. Customer Segmentation (Unsupervised)

**Objective:** Group customers by transaction behavior for targeted marketing.

**Approach:**
- Aggregated per-customer features from transactions
- K-Means with optimal K from silhouette score
- PCA visualization for cluster interpretation

**Business value:** Personalized campaigns for high-value vs low-activity segments.

## 3. Product Recommendations

**Objective:** Recommend top banking products per customer.

**Approach:**
- Matrix factorization (TruncatedSVD) for collaborative filtering
- Content-based similarity on product category and fees
- Hybrid scoring with Precision@5, Recall@5, MAP, NDCG

**Business value:** Increase cross-sell conversion and customer lifetime value.

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Class imbalance in defaults | `class_weight='balanced'` in classifiers |
| Sparse interaction matrix | Hybrid CF + content-based boosting |
| Choosing cluster count | Silhouette score across K=2..8 |

## Stakeholder Recommendations

1. Use the loan model as a **decision support** tool, not sole approval criterion (regulatory compliance).
2. Refresh segmentation quarterly as transaction patterns shift.
3. A/B test recommended products in digital banking channels.

## Reproducibility

```bash
python run_pipeline.py
```

All metrics are saved under `reports/` as JSON and PNG visualizations.
