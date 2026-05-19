from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SQL_DIR = PROJECT_ROOT / "sql"
DB_PATH = DATA_DIR / "banking.db"

RANDOM_STATE = 42
N_CUSTOMERS = 5000
N_TRANSACTIONS = 50000
N_INTERACTIONS = 30000
N_PRODUCTS = 25

LOAN_FILE = RAW_DIR / "loans.csv"
TRANSACTIONS_FILE = RAW_DIR / "transactions.csv"
INTERACTIONS_FILE = RAW_DIR / "interactions.csv"
PRODUCTS_FILE = RAW_DIR / "products.csv"
