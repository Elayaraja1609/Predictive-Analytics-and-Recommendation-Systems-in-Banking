-- High-risk loan portfolio summary
SELECT
    credit_risk_band,
    COUNT(*) AS loan_count,
    ROUND(AVG(loan_amount), 2) AS avg_loan_amount,
    ROUND(SUM(repayment_status) * 100.0 / COUNT(*), 2) AS default_rate_pct
FROM loans
GROUP BY credit_risk_band
ORDER BY default_rate_pct DESC;

-- Monthly transaction volume by type
SELECT
    strftime('%Y-%m', transaction_date) AS month,
    transaction_type,
    COUNT(*) AS txn_count,
    ROUND(SUM(transaction_amount), 2) AS total_amount
FROM transactions
GROUP BY month, transaction_type
ORDER BY month, transaction_type;

-- Top products by interaction strength
SELECT
    p.product_name,
    p.category,
    COUNT(*) AS interaction_count,
    SUM(CASE WHEN i.interaction_type = 'purchased' THEN 1 ELSE 0 END) AS purchases
FROM interactions i
JOIN products p ON i.product_id = p.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY interaction_count DESC
LIMIT 15;

-- Customer transaction profile for segmentation input
SELECT
    t.customer_id,
    COUNT(*) AS txn_count,
    ROUND(AVG(t.transaction_amount), 2) AS avg_amount,
    ROUND(SUM(t.transaction_amount), 2) AS total_amount,
    SUM(CASE WHEN t.transaction_type = 'deposit' THEN 1 ELSE 0 END) AS deposit_count,
    SUM(CASE WHEN t.transaction_type = 'withdrawal' THEN 1 ELSE 0 END) AS withdrawal_count
FROM transactions t
GROUP BY t.customer_id;
