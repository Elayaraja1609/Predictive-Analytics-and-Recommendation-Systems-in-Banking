# Predictive Analytics and Recommendation Systems in Banking

**Domain:** Banking  
**Skills:** Python, Data Science, SQL, Machine Learning  

## Problem Statement

Banks process large volumes of customer and transaction data daily. This project addresses three core use cases:

1. **Loan default prediction** (supervised learning) — minimize credit risk
2. **Customer segmentation** (unsupervised learning) — targeted marketing
3. **Product recommendations** (recommendation engine) — cross-selling and loyalty

## Project Structure

```
Final Project/
├── app/streamlit_app.py      # Deployment demo
├── data/raw/                 # Synthetic CSV datasets
├── data/banking.db           # SQLite database
├── docs/DEPLOYMENT.md        # Deployment steps
├── models/                   # Trained model artifacts
├── notebooks/                # EDA & analysis
├── reports/                  # Metrics, plots, recommendations
├── sql/banking_analytics.sql # SQL analytics queries
├── src/                      # Python modules
├── run_pipeline.py           # End-to-end training
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
```

## Datasets (Synthetic — Faker)

| Dataset | Key Columns | Derived Features |
|---------|-------------|------------------|
| Loans | customer_id, age, income, credit_score, loan_amount, ... | debt_to_income_ratio, payment_to_income_ratio, credit_risk_band |
| Transactions | customer_id, transaction_id, amount, type, date | txn_frequency, deposit_ratio (in segmentation) |
| Interactions | customer_id, product_id, interaction_type, date | weighted implicit ratings |

## Models & Evaluation Metrics

| Use Case | Algorithms | Metrics |
|----------|------------|---------|
| Loan Default | Logistic Regression, Random Forest, Gradient Boosting | Accuracy, Precision, Recall, F1, ROC-AUC |
| Segmentation | K-Means (silhouette-based K) | Silhouette Score, Davies-Bouldin, PCA plot |
| Recommendations | SVD + Hybrid Collaborative/Content | Precision@5, Recall@5, MAP, NDCG |

## Business Impact

- **Risk:** Flag high-default customers before loan approval
- **Marketing:** Tailor campaigns to behavioral segments
- **Revenue:** Recommend relevant products per customer profile
