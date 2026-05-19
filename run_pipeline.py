import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.generate_data import run as generate_data
from src.sql_setup import load_csvs_to_db, run_analytics_queries
from src.train_loan_model import train_and_evaluate as train_loan
from src.train_segmentation import train_and_evaluate as train_segmentation
from src.train_recommendation import train_and_evaluate as train_recommendation


def main():
    print("=" * 60)
    print("STEP 1: Generate synthetic banking datasets")
    print("=" * 60)
    generate_data()

    print("\n" + "=" * 60)
    print("STEP 2: Load data into SQLite & run SQL analytics")
    print("=" * 60)
    load_csvs_to_db()
    run_analytics_queries()

    print("\n" + "=" * 60)
    print("STEP 3: Train loan default prediction model")
    print("=" * 60)
    train_loan()

    print("\n" + "=" * 60)
    print("STEP 4: Customer segmentation (K-Means)")
    print("=" * 60)
    train_segmentation()

    print("\n" + "=" * 60)
    print("STEP 5: Product recommendation engine")
    print("=" * 60)
    train_recommendation()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("Deploy: streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
