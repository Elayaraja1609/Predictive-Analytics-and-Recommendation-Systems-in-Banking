"""
Load raw CSVs into SQLite and expose analytics queries for the banking domain.
"""
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import (
    DB_PATH,
    INTERACTIONS_FILE,
    LOAN_FILE,
    PRODUCTS_FILE,
    RAW_DIR,
    SQL_DIR,
    TRANSACTIONS_FILE,
)


def load_csvs_to_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    tables = {
        "loans": LOAN_FILE,
        "transactions": TRANSACTIONS_FILE,
        "interactions": INTERACTIONS_FILE,
        "products": PRODUCTS_FILE,
    }
    for table, path in tables.items():
        if path.exists():
            df = pd.read_csv(path)
            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"Loaded {table}: {len(df)} rows")

    conn.close()
    print(f"Database ready at {db_path}")


def run_analytics_queries(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    queries_file = SQL_DIR / "banking_analytics.sql"

    results = {}
    if queries_file.exists():
        sql_text = queries_file.read_text(encoding="utf-8")
        blocks = [b.strip() for b in sql_text.split(";") if b.strip() and not b.strip().startswith("--")]
        for i, block in enumerate(blocks):
            name = f"query_{i + 1}"
            try:
                results[name] = pd.read_sql_query(block, conn)
            except Exception as exc:
                print(f"Skipped {name}: {exc}")

    conn.close()
    return results


if __name__ == "__main__":
    load_csvs_to_db()
    results = run_analytics_queries()
    for name, df in results.items():
        print(f"\n--- {name} ({len(df)} rows) ---")
        print(df.head())
