"""Loading and cleaning of the transactions CSV."""
import pandas as pd

REQUIRED_COLUMNS = {"date", "category", "type", "amount"}
TYPE_MAP = {"income": "income", "приход": "income",
            "expense": "expense", "разход": "expense"}
UNKNOWN = "Unknown"


def load_and_clean(file) -> tuple[pd.DataFrame, dict]:
    """Read a CSV (path or file-like) and return (clean_df, cleaning_report)."""
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    report = {"rows_loaded": len(df)}

    df = df.drop_duplicates()
    report["duplicates_removed"] = report["rows_loaded"] - len(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(
        df["amount"].astype(str).str.replace(",", ".", regex=False)
                    .str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce",
    )
    df["type"] = df["type"].astype(str).str.strip().str.lower().map(TYPE_MAP)

    df["category"] = df["category"].fillna("").astype(str).str.strip().str.title()
    df.loc[df["category"] == "", "category"] = UNKNOWN

    if "description" not in df.columns:
        df["description"] = ""
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    before = len(df)
    df = df.dropna(subset=["date", "amount", "type"])
    report["invalid_rows_removed"] = before - len(df)
    report["rows_final"] = len(df)

    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df, report
