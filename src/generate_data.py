"""
Synthetic banking dataset generation with Faker and derived features.
Covers: loan defaults, transactions, product interactions.
"""
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.config import (
    INTERACTIONS_FILE,
    LOAN_FILE,
    N_CUSTOMERS,
    N_INTERACTIONS,
    N_PRODUCTS,
    N_TRANSACTIONS,
    PRODUCTS_FILE,
    RAW_DIR,
    RANDOM_STATE,
    TRANSACTIONS_FILE,
)

fake = Faker()
Faker.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

TRANSACTION_TYPES = ["deposit", "withdrawal", "transfer", "payment", "atm"]
INTERACTION_TYPES = ["purchased", "viewed", "clicked", "applied"]
PRODUCT_CATALOG = [
    ("Savings Plus", "savings"),
    ("Premium Checking", "checking"),
    ("Gold Credit Card", "credit_card"),
    ("Personal Loan", "loan"),
    ("Home Loan", "mortgage"),
    ("Auto Loan", "auto_loan"),
    ("Fixed Deposit 1Y", "fixed_deposit"),
    ("Mutual Fund Starter", "investment"),
    ("Health Insurance", "insurance"),
    ("Travel Insurance", "insurance"),
    ("Business Account", "business"),
    ("Student Savings", "savings"),
    ("Rewards Credit Card", "credit_card"),
    ("Education Loan", "loan"),
    ("Wealth Management", "investment"),
    ("Senior Citizen FD", "fixed_deposit"),
    ("NRI Account", "checking"),
    ("Overdraft Facility", "credit"),
    ("Debit Card Premium", "card"),
    ("Mobile Banking Pro", "digital"),
    ("Forex Card", "card"),
    ("PPF Account", "savings"),
    ("Demat Account", "investment"),
    ("Two-Wheeler Loan", "auto_loan"),
    ("Crop Insurance", "insurance"),
]


def _customer_ids(n: int) -> list[str]:
    return [f"CUST{str(i + 1).zfill(6)}" for i in range(n)]


def generate_products() -> pd.DataFrame:
    rows = []
    for idx, (name, category) in enumerate(PRODUCT_CATALOG[:N_PRODUCTS], start=1):
        rows.append(
            {
                "product_id": f"PROD{str(idx).zfill(4)}",
                "product_name": name,
                "category": category,
                "annual_fee": round(random.uniform(0, 5000), 2),
                "min_balance": round(random.uniform(0, 50000), 2),
            }
        )
    return pd.DataFrame(rows)


def generate_loans(customers: list[str]) -> pd.DataFrame:
    rows = []
    for cid in customers:
        age = random.randint(22, 65)
        income = round(np.random.lognormal(mean=10.8, sigma=0.45), 2)
        credit_score = int(np.clip(np.random.normal(680, 80), 300, 850))
        loan_amount = round(np.random.lognormal(mean=10.2, sigma=0.6), 2)
        loan_amount = min(loan_amount, income * 5)
        interest_rate = round(
            np.clip(6 + (850 - credit_score) / 50 + random.uniform(-1, 2), 3.5, 24), 2
        )
        loan_term = random.choice([12, 24, 36, 48, 60, 72, 84])
        employment_years = random.randint(0, max(1, age - 22))
        num_existing_loans = random.randint(0, 4)
        savings_balance = round(income * random.uniform(0.05, 1.2), 2)

        debt_to_income = round(loan_amount / max(income, 1), 4)
        monthly_income = round(income / 12, 2)
        monthly_rate = interest_rate / 100 / 12
        if monthly_rate > 0:
            monthly_payment = round(
                loan_amount
                * (monthly_rate * (1 + monthly_rate) ** loan_term)
                / ((1 + monthly_rate) ** loan_term - 1),
                2,
            )
        else:
            monthly_payment = round(loan_amount / loan_term, 2)
        payment_to_income = round(monthly_payment / max(monthly_income, 1), 4)
        credit_utilization = round(random.uniform(0.1, 0.95), 3)
        loan_to_income = round(loan_amount / max(income, 1), 4)

        default_prob = (
            0.15 * (credit_score < 600)
            + 0.12 * (debt_to_income > 0.5)
            + 0.1 * (payment_to_income > 0.4)
            + 0.08 * (credit_utilization > 0.7)
            + 0.06 * (num_existing_loans >= 3)
            + 0.05 * (employment_years < 2)
            + random.uniform(0, 0.15)
        )
        repayment_status = int(default_prob > random.uniform(0.35, 0.85))

        rows.append(
            {
                "customer_id": cid,
                "age": age,
                "income": income,
                "credit_score": credit_score,
                "loan_amount": loan_amount,
                "interest_rate": interest_rate,
                "loan_term": loan_term,
                "employment_years": employment_years,
                "num_existing_loans": num_existing_loans,
                "savings_balance": savings_balance,
                "debt_to_income_ratio": debt_to_income,
                "monthly_payment": monthly_payment,
                "payment_to_income_ratio": payment_to_income,
                "credit_utilization": credit_utilization,
                "loan_to_income_ratio": loan_to_income,
                "credit_risk_band": (
                    "high" if credit_score < 580 else "medium" if credit_score < 700 else "low"
                ),
                "repayment_status": repayment_status,
            }
        )
    return pd.DataFrame(rows)


def generate_transactions(customers: list[str]) -> pd.DataFrame:
    rows = []
    start = datetime(2023, 1, 1)
    end = datetime(2025, 12, 31)
    txn_count = 0

    while txn_count < N_TRANSACTIONS:
        cid = random.choice(customers)
        txn_type = random.choices(
            TRANSACTION_TYPES,
            weights=[0.35, 0.25, 0.2, 0.15, 0.05],
        )[0]
        base = random.uniform(100, 50000)
        if txn_type == "atm":
            amount = round(random.uniform(500, 10000), 2)
        elif txn_type == "deposit":
            amount = round(base * random.uniform(0.5, 3), 2)
        else:
            amount = round(base * random.uniform(0.1, 1.5), 2)

        delta = (end - start).days
        txn_date = start + timedelta(days=random.randint(0, delta))

        rows.append(
            {
                "customer_id": cid,
                "transaction_id": f"TXN{str(txn_count + 1).zfill(8)}",
                "transaction_amount": amount,
                "transaction_type": txn_type,
                "transaction_date": txn_date.strftime("%Y-%m-%d"),
                "channel": random.choice(["branch", "mobile", "net_banking", "atm"]),
            }
        )
        txn_count += 1

    return pd.DataFrame(rows)


def generate_interactions(customers: list[str], products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    product_ids = products["product_id"].tolist()
    weights = {"purchased": 0.15, "viewed": 0.45, "clicked": 0.3, "applied": 0.1}
    start = datetime(2024, 1, 1)
    end = datetime(2025, 12, 31)

    for i in range(N_INTERACTIONS):
        cid = random.choice(customers)
        pid = random.choice(product_ids)
        itype = random.choices(
            list(weights.keys()), weights=list(weights.values())
        )[0]
        delta = (end - start).days
        idate = start + timedelta(days=random.randint(0, delta))
        rows.append(
            {
                "customer_id": cid,
                "product_id": pid,
                "interaction_type": itype,
                "interaction_date": idate.strftime("%Y-%m-%d"),
                "rating": (
                    random.randint(4, 5)
                    if itype == "purchased"
                    else random.randint(1, 4)
                ),
            }
        )
    return pd.DataFrame(rows)


def run():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    customers = _customer_ids(N_CUSTOMERS)

    print("Generating products catalog...")
    products = generate_products()
    products.to_csv(PRODUCTS_FILE, index=False)

    print("Generating loan data...")
    loans = generate_loans(customers)
    loans.to_csv(LOAN_FILE, index=False)

    print("Generating transactions...")
    transactions = generate_transactions(customers)
    transactions.to_csv(TRANSACTIONS_FILE, index=False)

    print("Generating product interactions...")
    interactions = generate_interactions(customers, products)
    interactions.to_csv(INTERACTIONS_FILE, index=False)

    print(f"Loans: {len(loans)} | Default rate: {loans['repayment_status'].mean():.2%}")
    print(f"Transactions: {len(transactions)} | Interactions: {len(interactions)}")
    print(f"Data saved to {RAW_DIR}")


if __name__ == "__main__":
    run()
