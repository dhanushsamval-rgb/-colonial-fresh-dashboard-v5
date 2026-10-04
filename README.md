# Colonial Fresh — AI-Based Demand Forecasting & Inventory Management

MGT5ELS Assessment 3 · Sprint 3 — Technical Upgrade

## What changed in this update

1. **Dataset date range fixed.** Now spans **2023-01-01 to 2025-04-30** only —
   nothing extends into 2026. A chronological 70/30 train/test split falls
   naturally inside this range: training on **~2023-01 → ~2024-08**, testing
   on **~2024-08 → 2025-04**.

2. **Fixed a real bug: false "Stockout Risk" during Live Simulation.**
   The risk table was checking raw closing stock, ignoring stock that was
   already on its way (incoming pipeline). After running the simulation for
   2 years, this made the dashboard show ~83% Stockout Risk even though the
   *true* underlying stockout rate (from the simulation log itself) was only
   ~9% and falling. Fixed by feeding the real, tracked pending-order pipeline
   into the risk classification instead of a rough historical-average guess.
   Verified: after the fix, a 2-year run shows 0% false stockout risk and a
   risk mix that matches the true simulated outcomes.

3. **Learning visibility.** The Live Simulation tab now shows an explicit
   **stockout rate: first N simulated days (fixed 20% baseline) vs most
   recent N days (adaptive rate)** comparison, plus the current average
   adaptive safety-stock rate vs the 20% starting point — not just a line
   chart you have to interpret yourself.

4. **Order Management restored to Stock Intake.** The top of the Stock
   Intake tab now shows every store × SKU that currently needs an order,
   with a **Place order** button per row. Placing an order shows its lead
   time and expected arrival date, and — when Live Simulation is running —
   actually queues it into the same pipeline the simulation uses, so it
   arrives and affects stock on the correct future day.

## Sprint 3 upgrades (per Sprint 2 lecturer feedback)

1. **Explicit 70/30 chronological split** (`ml_forecast.py`) — never shuffled.
2. **Tracking signal** (RSFE ÷ cumulative MAD, ±4 control limit).
3. **Order Management Dashboard** — Reorder Recommendation tab + the new
   Recommended Orders panel in Stock Intake.
4. **Bias-correction attempt** — reported honestly either way.

## Structure

- `data_utils.py` — load/clean/validate the Excel dataset
- `forecasting.py` — 7-day moving-average baseline forecast
- `inventory_analysis.py` — reorder calculation, risk classification, stock assessment
- `ml_forecast.py` — Random Forest model, 70/30 backtest, tracking signal
- `simulation.py` — Live Simulation engine (day-by-day, adaptive safety-stock learning, real order pipeline)
- `app.py` — Streamlit dashboard (13 tabs, all showing all 6 stores × all 3 SKUs)
- `gen_data/generate_dataset_2023_2025apr.py` — synthetic dataset generator
- `data/Colonial_Fresh_Mock_Data_2023_2025Apr_3_SKUs.xlsx` — the dataset itself

## Running locally

```
pip install -r requirements.txt
streamlit run app.py
```

Simulated data only — not real Colonial Fresh data, not connected to any live system.
