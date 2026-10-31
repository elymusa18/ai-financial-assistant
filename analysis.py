"""Financial calculations on a cleaned transactions DataFrame."""
import pandas as pd


def compute_summary(df: pd.DataFrame) -> dict:
    income = df.loc[df["type"] == "income", "amount"].sum()
    expenses = df.loc[df["type"] == "expense", "amount"].sum()

    monthly = (df.pivot_table(index="month", columns="type", values="amount",
                              aggfunc="sum", fill_value=0)
                 .reindex(columns=["income", "expense"], fill_value=0))
    monthly["profit"] = monthly["income"] - monthly["expense"]

    by_category = (df[df["type"] == "expense"].groupby("category")["amount"]
                   .sum().sort_values(ascending=False))
    return {"income": income, "expenses": expenses, "profit": income - expenses,
            "monthly": monthly, "by_category": by_category}


def build_context(summary: dict) -> str:
    """Compact text version of the computed data that is sent to the LLM."""
    return (
        f"Total income: {summary['income']:.2f}\n"
        f"Total expenses: {summary['expenses']:.2f}\n"
        f"Total profit: {summary['profit']:.2f}\n\n"
        f"Monthly totals:\n{summary['monthly'].round(2).to_string()}\n\n"
        f"Expenses by category:\n{summary['by_category'].round(2).to_string()}"
    )
