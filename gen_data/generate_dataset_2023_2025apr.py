"""
Colonial Fresh synthetic dataset generator.

Total dataset span = 2023-01-01 -> 2025-04-30 (851 days) — nothing beyond
April 2025. A chronological 70/30 train/test split falls naturally inside
this range:
  70% (train) = 2023-01-01 -> ~2024-08-18
  30% (test)  = ~2024-08-19 -> 2025-04-30

This matches the lecturer's Sprint 3 feedback ("use 70% of the data ... to
see if it's actually doing better than the actual 30% data") without
inventing any future/simulated dates beyond the requested window.

Everything else (stores, SKUs, statistical parameters, accounting-identity
integrity) matches the dataset used in earlier sprints.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

START_DATE = "2023-01-01"
END_DATE = "2025-04-30"

STORES = ["Chadstone", "Doncaster", "Forest Hill", "Glen Waverley", "Malvern", "The Pines"]
STORE_SCALE = {
    "Chadstone": 1.35,
    "Doncaster": 1.00,
    "Forest Hill": 0.85,
    "Glen Waverley": 1.10,
    "Malvern": 0.55,
    "The Pines": 0.95,
}

PRODUCTS = {
    "SKU001": {"name": "Bananas", "base_rate": 14.0, "lead_time": 1, "shelf_life": 5, "price": 0.69},
    "SKU002": {"name": "Apples", "base_rate": 6.0, "lead_time": 2, "shelf_life": 10, "price": 0.99},
    "SKU003": {"name": "Strawberries", "base_rate": 1.5, "lead_time": 2, "shelf_life": 4, "price": 3.49},
}

WEEKDAY_FACTOR = {0: 0.85, 1: 0.85, 2: 0.90, 3: 0.95, 4: 1.10, 5: 1.35, 6: 1.20}
PROMO_PROBABILITY = 0.0758
PROMO_BOOST = 1.266
DEMAND_NOISE_STD = 0.233
WASTAGE_LAMBDA = 1.3

REORDER_POINT_MULT = 1.6
ORDER_QTY_TARGET_MULT = 3.2
INITIAL_STOCK_MULT = 2.3

rng = np.random.default_rng(42)


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def generate_combo(store, sku):
    meta = PRODUCTS[sku]
    scale = STORE_SCALE[store]
    base_rate = meta["base_rate"] * scale
    lead_time = meta["lead_time"]
    shelf_life = meta["shelf_life"]
    price = meta["price"]

    reorder_point = int(round(base_rate * lead_time * REORDER_POINT_MULT))
    order_qty_target = int(round(base_rate * lead_time * ORDER_QTY_TARGET_MULT))
    current_stock = int(round(base_rate * lead_time * INITIAL_STOCK_MULT))

    start = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)

    pending_orders = {}  # arrival_date -> qty
    rows = []

    for d in daterange(start, end):
        wd = d.weekday()
        wf = WEEKDAY_FACTOR[wd]
        promo = rng.random() < PROMO_PROBABILITY
        promo_mult = PROMO_BOOST if promo else 1.0
        noise = max(0.05, rng.normal(1.0, DEMAND_NOISE_STD))
        true_demand = base_rate * wf * promo_mult * noise
        simulated_true_demand = true_demand

        arriving = int(pending_orders.pop(d, 0))

        opening_stock = current_stock  # PRE-delivery, matches original convention
        available_for_sale = current_stock + arriving  # POST-delivery, used for sales

        units_sold = int(min(available_for_sale, round(true_demand)))
        stockout = 1 if available_for_sale < true_demand else 0

        after_sales = available_for_sale - units_sold

        wastage = int(min(after_sales, rng.poisson(WASTAGE_LAMBDA)))
        closing_stock = after_sales - wastage

        current_stock = closing_stock

        # reorder logic (simple point-based, on post-sale position incl pipeline)
        pipeline_qty = sum(pending_orders.values())
        if (current_stock + pipeline_qty) <= reorder_point:
            order_qty = order_qty_target
            arrival_date = d + timedelta(days=lead_time)
            pending_orders[arrival_date] = pending_orders.get(arrival_date, 0) + order_qty

        expiry_date = d + timedelta(days=shelf_life)

        rows.append({
            "Date": d,
            "Store": store,
            "Product_ID": sku,
            "Product_Name": meta["name"],
            "Movement_Class": "",  # filled later from calculated averages
            "Opening_Stock": opening_stock,
            "Incoming_Stock": arriving,
            "Simulated_True_Demand": round(simulated_true_demand, 2),
            "Units_Sold": units_sold,
            "Closing_Stock": closing_stock,
            "Wastage": wastage,
            "Promotion": "Yes" if promo else "No",
            "Selling_Price_AUD": price,
            "Stockout": stockout,
            "Lead_Time_Days": lead_time,
            "Shelf_Life_Days": shelf_life,
            "Expiry_Date": expiry_date,
        })

        # accounting identity check
        assert opening_stock + arriving - units_sold - wastage == closing_stock

    return rows


def main():
    all_rows = []
    for store in STORES:
        for sku in PRODUCTS:
            all_rows.extend(generate_combo(store, sku))

    df = pd.DataFrame(all_rows)

    # Assign movement class from calculated network-wide average daily sales
    avg_sales = df.groupby("Product_ID")["Units_Sold"].mean().sort_values(ascending=False)
    ranked = list(avg_sales.index)
    movement_map = {ranked[0]: "Fast-moving", ranked[1]: "Moderate-moving", ranked[2]: "Slow-moving"}
    df["Movement_Class"] = df["Product_ID"].map(movement_map)

    out_path = "/home/claude/colonial_fresh_dashboard/data/Colonial_Fresh_Mock_Data_2023_2025Apr_3_SKUs.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Mock_Data", index=False)
        readme = pd.DataFrame({
            "Note": [
                "Colonial Fresh simulated dataset for MGT5ELS Assessment 3.",
                f"Date range: {START_DATE} to {END_DATE}.",
                "A chronological 70/30 train/test split falls naturally inside this "
                "range: training on ~2023-01 to ~2024-08, testing on ~2024-08 to 2025-04, "
                "per Sprint 2 lecturer feedback.",
                "Simulated data only — not real Colonial Fresh data, not connected to any live system.",
            ]
        })
        readme.to_excel(writer, sheet_name="README", index=False)

    print("Wrote:", out_path)
    print(df.shape)
    print(avg_sales)
    print(movement_map)


if __name__ == "__main__":
    main()
