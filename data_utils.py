"""
Data loading, cleaning and validation utilities for the Colonial Fresh
demand-forecasting & inventory-management dashboard.

Dataset: 2023-01-01 to 2025-04-30, simulated for academic use only.
A chronological 70/30 split falls naturally inside this range: training on
~2023-01 -> ~2024-08, testing on ~2024-08 -> 2025-04, per Sprint 2 lecturer
feedback.
"""
import pandas as pd
import streamlit as st

DATA_PATH = "data/Colonial_Fresh_Mock_Data_2023_2025Apr_3_SKUs.xlsx"

REQUIRED_COLUMNS = [
    "Date", "Store", "Product_ID", "Product_Name", "Movement_Class",
    "Opening_Stock", "Incoming_Stock", "Simulated_True_Demand", "Units_Sold",
    "Closing_Stock", "Wastage", "Promotion", "Selling_Price_AUD", "Stockout",
    "Lead_Time_Days", "Shelf_Life_Days", "Expiry_Date",
]

NUMERIC_COLS = [
    "Opening_Stock", "Incoming_Stock", "Simulated_True_Demand", "Units_Sold",
    "Closing_Stock", "Wastage", "Selling_Price_AUD", "Stockout",
    "Lead_Time_Days", "Shelf_Life_Days",
]


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load, validate and clean the Colonial Fresh mock dataset."""
    df = pd.read_excel(path, sheet_name="Mock_Data")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Expiry_Date"] = pd.to_datetime(df["Expiry_Date"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["Date", "Store", "Product_ID"])
    n_after = len(df)
    if n_after < n_before:
        st.warning(f"Dropped {n_before - n_after} rows with missing Date/Store/Product_ID.")

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    df["Promotion"] = df["Promotion"].astype(str).str.strip().str.lower().eq("yes").astype(int)

    df["Store"] = df["Store"].astype(str).str.strip()
    df["Product_ID"] = df["Product_ID"].astype(str).str.strip()
    df["Product_Name"] = df["Product_Name"].astype(str).str.strip()

    df = df.sort_values(["Product_ID", "Store", "Date"]).reset_index(drop=True)
    return df


def get_stores(df: pd.DataFrame) -> list:
    return sorted(df["Store"].unique().tolist())


def get_products(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["Product_ID", "Product_Name"]]
        .drop_duplicates()
        .sort_values("Product_ID")
        .reset_index(drop=True)
    )


def get_as_of_date(df: pd.DataFrame):
    """Most recent date in the dataset, treated as 'today' for all
    current-inventory, lead-time and expiry calculations."""
    return df["Date"].max()
