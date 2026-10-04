import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_utils
import forecasting
import inventory_analysis as inv
import ml_forecast
import simulation

st.set_page_config(page_title="Colonial Fresh — AI Demand Forecasting & Inventory", layout="wide")

# ---------------------------------------------------------------- load data
df = data_utils.load_data()
STORES = data_utils.get_stores(df)
PRODUCTS_DF = data_utils.get_products(df)
PRODUCTS = PRODUCTS_DF["Product_ID"].tolist()
PRODUCT_NAME = dict(zip(PRODUCTS_DF["Product_ID"], PRODUCTS_DF["Product_Name"]))
HISTORICAL_DF = df
HISTORICAL_AS_OF_DATE = data_utils.get_as_of_date(df)

# ---------------------------------------------------------------- session state
if "sim_state" not in st.session_state:
    st.session_state.sim_state = simulation.init_simulation_state(
        HISTORICAL_DF, STORES, PRODUCTS, HISTORICAL_AS_OF_DATE, seed=7
    )
if "sim_autoplay" not in st.session_state:
    st.session_state.sim_autoplay = False
if "sim_speed" not in st.session_state:
    st.session_state.sim_speed = 1
if "stock_ledger" not in st.session_state:
    st.session_state.stock_ledger = []

_pending_reset = st.session_state.pop("_intake_pending_reset", None)
if _pending_reset:
    for _k, _v in _pending_reset.items():
        st.session_state[_k] = _v

sim_state = st.session_state.sim_state
LIVE_MODE = sim_state["days_simulated"] > 0

if LIVE_MODE:
    df = simulation.extended_dataframe(HISTORICAL_DF, sim_state["log"], PRODUCTS)
    AS_OF_DATE = sim_state["sim_date"]
    safety_stock_rates = simulation.current_safety_stock_rates(sim_state)
    # Real pending-order pipeline (not a historical-average guess) — this is
    # what stops Live Mode from misreading stock that's already on its way
    # as a stockout risk.
    incoming_pipeline = simulation.current_incoming_pipeline(sim_state)
else:
    AS_OF_DATE = HISTORICAL_AS_OF_DATE
    safety_stock_rates = {}
    incoming_pipeline = None


def _stock_adjustments_dict():
    agg = {}
    for entry in st.session_state.stock_ledger:
        key = (entry["store"], entry["sku"])
        agg[key] = agg.get(key, 0) + entry["qty"]
    return agg


stock_adjustments = _stock_adjustments_dict()

# ---------------------------------------------------------------- header
st.title("🥑 Colonial Fresh — AI-Based Demand Forecasting & Inventory Management")
st.caption("MGT5ELS Assessment 3 · Sprint 3 — Technical Upgrade")

st.info(
    "**Prototype notice:** This dashboard runs on a fully **simulated** dataset "
    f"({HISTORICAL_DF['Date'].min().date()} to {HISTORICAL_AS_OF_DATE.date()}) built for academic purposes. "
    "It is **not** real Colonial Fresh data and is **not** connected to any live system. "
    f"The most recent date in the dataset ({HISTORICAL_AS_OF_DATE.date()}) is treated as **'today'** "
    "for all current-inventory, lead-time and expiry calculations, to demonstrate how a real-time "
    "dashboard would behave. Live system integration is proposed as future work beyond this Sprint design."
)

if LIVE_MODE:
    st.success(
        f"🔴 **LIVE MODE ACTIVE** — the simulation has run **{sim_state['days_simulated']} day(s)** "
        f"forward, to **{AS_OF_DATE.date()}**. Every tab below (Overview, Movement, Forecasting, "
        "Inventory, Lead-Time, Expiry, Reorder, Risk) now reads this extended, simulated timeline — "
        "including each combo's **adaptively learned safety-stock rate** instead of the fixed 20% "
        "baseline. Go to 🔴 **Live Simulation** to advance further or reset back to the real historical data only."
    )

with st.sidebar:
    st.header("Settings")
    forecast_period = st.selectbox("Forecast period (days)", [7, 14, 30], index=0)
    st.caption("Every tab below always shows **all 6 stores × all 3 SKUs** together — nothing is filtered or hidden.")

# ---------------------------------------------------------------- shared tables
risk_table = inv.build_risk_table(df, STORES, PRODUCTS, AS_OF_DATE, forecast_period,
                                   stock_adjustments, safety_stock_rates, incoming_pipeline)
full_table = inv.build_full_snapshot_table(df, STORES, PRODUCTS, AS_OF_DATE, forecast_period,
                                            stock_adjustments, safety_stock_rates, incoming_pipeline)

tabs = st.tabs([
    "1️⃣ Overview", "2️⃣ Movement Analysis", "3️⃣ Demand Forecasting", "🧠 ML Forecast",
    "4️⃣ Inventory Monitoring", "📥 Stock Intake", "5️⃣ Lead-Time Tracking", "🚚 Lead-Time Anticipation",
    "6️⃣ Expiry Tracking", "7️⃣ Reorder Recommendation", "8️⃣ Risk Dashboard",
    "🔴 Live Simulation", "9️⃣ Solution Architecture",
])
(tab_overview, tab_movement, tab_forecast, tab_ml, tab_inventory, tab_intake,
 tab_leadtime, tab_leadtime_anticipation, tab_expiry, tab_reorder, tab_risk,
 tab_live, tab_arch) = tabs

# ============================================================== 1. OVERVIEW
with tab_overview:
    st.header("Overview Dashboard")
    st.caption(f"All 6 stores × all 3 SKUs · forecast period = {forecast_period} days · as of {AS_OF_DATE.date()}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Current Inventory (units)", f"{full_table['Current_Stock'].sum():,.0f}")
    c2.metric(f"Forecasted Demand (next {forecast_period}d)", f"{full_table['Forecast_Next_Period'].sum():,.0f}")
    today_incoming = df[df["Date"] == AS_OF_DATE]["Incoming_Stock"].sum()
    c3.metric("Incoming Stock (today)", f"{today_incoming:,.0f}")
    c4.metric("Wastage (last 30d)", f"{full_table['Wastage_Last30d'].sum():,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Stockout Occurrences (last 30d)", f"{full_table['Stockout_Last30d'].sum():,.0f}")
    n_attention = int((full_table["Risk"] != "Normal").sum())
    c6.metric("Store × SKU Combos Needing Attention", f"{n_attention} / {len(full_table)}")
    c7.metric("Total Recommended Reorder Qty", f"{full_table['Recommended_Order'].sum():,.0f} units")

    st.divider()
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.subheader("Products / Stores Requiring Attention")
        attention = full_table[full_table["Risk"] != "Normal"][
            ["Store", "SKU", "Product", "Risk", "Current_Stock", "Recommended_Order"]
        ].sort_values("Risk")
        st.dataframe(attention, use_container_width=True, hide_index=True)
    with col_b:
        st.subheader("Risk Mix (current scope)")
        risk_counts = full_table["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig = px.pie(risk_counts, names="Risk", values="Count", hole=0.5,
                     color="Risk", color_discrete_map={
                         "Normal": "#2ca02c", "Overstock Risk": "#ff7f0e",
                         "Stockout Risk": "#d62728", "Expiry Risk": "#9467bd",
                     })
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== 2. MOVEMENT ANALYSIS
with tab_movement:
    st.header("Product Movement Analysis")
    st.caption("Movement class is **calculated** from average daily `Units_Sold` across the full "
               f"{HISTORICAL_DF['Date'].min().year}-{HISTORICAL_AS_OF_DATE.year} history for each SKU — not assumed.")

    move_summary = HISTORICAL_DF.groupby(["Product_ID", "Product_Name"]).agg(
        Avg_Daily_Units_Sold=("Units_Sold", "mean"),
        Max_Daily_Units_Sold=("Units_Sold", "max"),
        Min_Daily_Units_Sold=("Units_Sold", "min"),
    ).reset_index()
    move_class = HISTORICAL_DF.groupby("Product_ID")["Movement_Class"].first()
    move_summary["Calculated_Movement_Class"] = move_summary["Product_ID"].map(move_class)
    move_summary = move_summary.sort_values("Avg_Daily_Units_Sold", ascending=False)
    st.dataframe(move_summary, use_container_width=True, hide_index=True)

    fig = px.bar(move_summary, x="Product_Name", y="Avg_Daily_Units_Sold",
                 color="Calculated_Movement_Class", text_auto=".1f",
                 title=f"Average Daily Units Sold by Product ({HISTORICAL_DF['Date'].min().year}-{HISTORICAL_AS_OF_DATE.year} full history)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================== 3. DEMAND FORECASTING (baseline)
with tab_forecast:
    st.header("Demand Forecasting — All Stores × All SKUs (Baseline)")
    st.markdown(
        "**Baseline model:** 7-day moving average of historical `Units_Sold`, projected forward as a "
        "flat daily rate. `Simulated_True_Demand` is **not** used as a model input, to avoid data "
        "leakage — it is reserved for later evaluation only."
    )
    network_daily = df.groupby(["Date", "Product_ID"])["Units_Sold"].sum().reset_index()
    fig = go.Figure()
    colors = {"SKU001": "green", "SKU002": "orange", "SKU003": "purple"}
    for sku in PRODUCTS:
        d = network_daily[network_daily["Product_ID"] == sku].sort_values("Date").tail(60)
        name = PRODUCT_NAME[sku]
        fig.add_trace(go.Scatter(x=d["Date"], y=d["Units_Sold"], name=f"{name} — Historical",
                                  line=dict(color=colors[sku])))
        daily_series = d.set_index("Date")["Units_Sold"]
        avg = forecasting.baseline_moving_average_forecast(daily_series, forecast_period)
        future_dates = pd.date_range(AS_OF_DATE + pd.Timedelta(days=1), periods=forecast_period)
        fig.add_trace(go.Scatter(x=future_dates, y=[avg] * forecast_period,
                                  name=f"{name} — Forecast", line=dict(color=colors[sku], dash="dash")))
    fig.update_layout(title="Network-Wide Historical Sales vs Forecast (all stores combined, per SKU)",
                       xaxis_title="Date", yaxis_title="Units Sold (all 6 stores)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast summary — every store × SKU")
    st.dataframe(
        full_table[["Store", "Product", "SKU", "Avg_Daily_Forecast", "Forecast_Next_Period"]]
        .rename(columns={"Forecast_Next_Period": f"Forecast (next {forecast_period}d)"}),
        use_container_width=True, hide_index=True,
    )

# ============================================================== ML FORECAST
with tab_ml:
    st.header("Machine Learning Forecast — Random Forest (network-wide, per SKU)")
    st.markdown(
        "Genuine machine-learning model — scikit-learn `RandomForestRegressor` — trained on lag, "
        "rolling-average and calendar features engineered from historical `Units_Sold`. "
        "`Simulated_True_Demand` is never used as a feature or target, for the same data-leakage reasons as the baseline model."
    )
    st.markdown(
        "**Sprint 3 upgrade (per lecturer feedback):** the model is trained on a **chronological 70% "
        "split** and evaluated on the **untouched 30%** — never shuffled, so no future information "
        "leaks into training. The dataset's date range (2023-01 to 2025-04) is sized so this 70/30 "
        "split falls naturally at roughly **2023-01 → 2024-08 (train)** and **2024-08 → 2025-04 (test)**."
    )

    sku_choice = st.selectbox("SKU", PRODUCTS, format_func=lambda s: f"{s} — {PRODUCT_NAME[s]}")

    @st.cache_data(show_spinner="Training and backtesting Random Forest…")
    def _cached_backtest(sku):
        return ml_forecast.train_and_backtest(HISTORICAL_DF, sku)

    bundle = _cached_backtest(sku_choice)

    c1, c2 = st.columns(2)
    c1.metric("Training days (70% split, fed to model.fit())", f"{bundle['train_days']}")
    c1.caption(f"{bundle['train_date_range'][0].date()} → {bundle['train_date_range'][1].date()}")
    c2.metric("Test days (30% split, never seen in training)", f"{bundle['test_days']}")
    c2.caption(f"{bundle['test_date_range'][0].date()} → {bundle['test_date_range'][1].date()}")
    st.caption(
        f"Of the 70% training window, the last {bundle['bias_val_days']} days are held back purely to "
        "estimate a bias correction — never shown to `model.fit()`."
    )

    st.subheader("Backtest accuracy — network-wide daily totals, on the untouched 30% test set")
    m1, m2, m3 = st.columns(3)
    m1.metric("Random Forest MAE", f"{bundle['rf_metrics']['MAE']:.1f}",
              delta=f"{bundle['rf_metrics']['MAE'] - bundle['baseline_metrics']['MAE']:.1f} vs baseline",
              delta_color="inverse")
    m2.metric("Random Forest RMSE", f"{bundle['rf_metrics']['RMSE']:.1f}")
    m3.metric("Random Forest MAPE", f"{bundle['rf_metrics']['MAPE']:.1f}%")
    st.caption(f"Baseline (7-day MA) on the same test set — MAE {bundle['baseline_metrics']['MAE']:.1f}, "
               f"RMSE {bundle['baseline_metrics']['RMSE']:.1f}, MAPE {bundle['baseline_metrics']['MAPE']:.1f}%.")

    if bundle["rf_metrics"]["MAE"] < bundle["baseline_metrics"]["MAE"]:
        st.success(f"On this backtest, Random Forest beats the baseline for {sku_choice} — "
                   f"{bundle['rf_metrics']['MAE']:.1f} vs {bundle['baseline_metrics']['MAE']:.1f} "
                   "average units of error per day, measured only on the 30% test set.")
    else:
        st.warning(f"On this backtest, Random Forest does **not** beat the baseline for {sku_choice}.")

    st.subheader("📶 Tracking Signal (RSFE ÷ cumulative MAD)")
    st.markdown(
        "The tracking signal shows whether the forecast is *going around the actual* (staying near "
        "zero) or drifting consistently one direction (systematic over/under-estimation) — something "
        "a single MAE number can hide. Conventional control limit: **±4**."
    )
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=bundle["test_dates"], y=bundle["rf_tracking_signal"],
                                 name="Random Forest", line=dict(color="green")))
    fig_ts.add_trace(go.Scatter(x=bundle["test_dates"], y=bundle["baseline_tracking_signal"],
                                 name="Baseline (7-day MA)", line=dict(color="orange")))
    fig_ts.add_hline(y=ml_forecast.TRACKING_SIGNAL_LIMIT, line_dash="dot", line_color="red")
    fig_ts.add_hline(y=-ml_forecast.TRACKING_SIGNAL_LIMIT, line_dash="dot", line_color="red")
    fig_ts.update_layout(title="Tracking Signal over the 30% test window", yaxis_title="RSFE / MAD")
    st.plotly_chart(fig_ts, use_container_width=True)

    tsc1, tsc2 = st.columns(2)
    tsc1.metric("Random Forest — days beyond ±4", bundle["rf_ts_breaches"])
    tsc2.metric("Baseline — days beyond ±4", bundle["baseline_ts_breaches"])
    if bundle["rf_ts_breaches"] > bundle["baseline_ts_breaches"]:
        st.warning(
            "Random Forest has **lower average error (MAE)** but **more tracking-signal breaches** than "
            "the baseline — meaning it beats the baseline day-to-day, but drifts into a systematic "
            "over/under-forecast for longer stretches. This is exactly the kind of thing MAE alone hides."
        )

    with st.expander("Extra step: attempted per-weekday bias correction"):
        st.markdown(
            "A held-out slice inside the 70% training window (never shown to `model.fit()`) is used to "
            "estimate, per weekday, how far off the model's predictions run — and that correction is "
            "added back onto the test-set forecasts below."
        )
        bc1, bc2 = st.columns(2)
        bc1.metric("Corrected MAE", f"{bundle['rf_metrics_corrected']['MAE']:.1f}")
        bc2.metric("Corrected — days beyond ±4", bundle["rf_ts_breaches_corrected"])
        if bundle["rf_ts_breaches_corrected"] >= bundle["rf_ts_breaches"]:
            st.warning(
                "Honestly: the bias correction did **not** reliably reduce tracking-signal breaches on "
                "this SKU. A likely reason is that the validation slice used to estimate the correction "
                f"(~{bundle['bias_val_days']} days) is small and noisy relative to weekday-level demand "
                "swings. Sprint 4 directions: a larger validation window, more frequent model retraining, "
                "or quantile regression instead of a flat mean correction."
            )
        else:
            st.success("The bias correction reduced tracking-signal breaches on this backtest.")

    with st.expander("Feature importances (what the model actually learned from)"):
        fi = pd.Series(bundle["feature_importances"]).sort_values(ascending=False)
        st.bar_chart(fi)

    st.subheader(f"Forward forecast — next {forecast_period} days (recursive, network-wide)")
    fwd = ml_forecast.forecast_forward(HISTORICAL_DF, sku_choice, bundle, AS_OF_DATE, forecast_period)
    st.dataframe(fwd, use_container_width=True, hide_index=True)
    st.caption("Recursive forecast: each predicted day feeds the lag features for the next day. "
               "Multi-step forecasts like this naturally get less certain the further out they go.")

# ============================================================== 4. INVENTORY MONITORING
with tab_inventory:
    st.header("Inventory Monitoring — All Stores × All SKUs")
    st.caption(f"As of {AS_OF_DATE.date()}.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Opening Stock", f"{full_table['Opening_Stock'].sum():,.0f}")
    c2.metric("Total Incoming Today", f"{full_table['Incoming_Stock'].sum():,.0f}")
    c3.metric("Total Units Sold Today", f"{full_table['Units_Sold'].sum():,.0f}")
    c4.metric("Total Closing Stock", f"{full_table['Closing_Stock'].sum():,.0f}")

    display_cols = ["Store", "Product", "SKU", "Opening_Stock", "Incoming_Stock", "Units_Sold",
                     "Closing_Stock", "Wastage", "Stockout", "Risk"]
    st.dataframe(full_table[display_cols], use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(full_table, x="Store", y="Closing_Stock", color="Product",
                     barmode="group", title="Closing Stock by Store, split by SKU")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.bar(full_table, x="Store", y="Wastage", color="Product",
                     barmode="group", title="Today's Wastage by Store, split by SKU")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== STOCK INTAKE
with tab_intake:
    st.header("📥 Stock Intake — Order & Log Received Stock")

    if "placed_orders" not in st.session_state:
        st.session_state.placed_orders = []

    # ---- Recommended Orders (Order Management), always at the top ----
    st.subheader("📋 Recommended Orders right now")
    if not LIVE_MODE:
        st.caption(
            "Calculated live from the current forecast, stock and safety-stock rate. Placing an order "
            "here is recorded with its expected lead time, but since you're viewing the static "
            "historical data (Live Simulation hasn't been started), the order won't actually arrive "
            "and change stock — start Live Simulation to see placed orders arrive and affect every "
            "other tab."
        )
    else:
        st.caption(
            "Calculated live from the current forecast, stock, real incoming pipeline and each "
            "combo's adaptively learned safety-stock rate. Placing an order here queues it into the "
            "same pipeline the Live Simulation uses — it will actually arrive after its lead time."
        )

    needs_order = full_table[full_table["Recommended_Order"] > 0].sort_values("Recommended_Order", ascending=False)
    if needs_order.empty:
        st.success("✅ No store × SKU combination currently needs an order.")
    else:
        for _, r in needs_order.iterrows():
            rc1, rc2, rc3, rc4, rc5 = st.columns([2, 2, 1.3, 1.3, 1.4])
            rc1.markdown(f"**{r['Store']}**")
            rc2.markdown(f"{r['Product']} ({r['SKU']})")
            rc3.markdown(f"Order **{r['Recommended_Order']:.0f}** units")
            rc4.markdown(f"Lead time: **{r['Lead_Time_Days']:.0f}d**")
            order_key = f"place_order_{r['Store']}_{r['SKU']}"
            if rc5.button("Place order", key=order_key):
                qty = int(r["Recommended_Order"])
                lead_time = int(r["Lead_Time_Days"])
                if LIVE_MODE:
                    result = simulation.place_manual_order(sim_state, r["Store"], r["SKU"], qty, lead_time)
                    arrival = result["arrival_date"]
                else:
                    arrival = AS_OF_DATE + pd.Timedelta(days=lead_time)
                st.session_state.placed_orders.append({
                    "store": r["Store"], "sku": r["SKU"], "qty": qty,
                    "date_placed": str(AS_OF_DATE.date()), "lead_time_days": lead_time,
                    "expected_arrival": str(pd.Timestamp(arrival).date()), "live": LIVE_MODE,
                })
                st.success(f"Ordered {qty} units of {r['SKU']} for {r['Store']} — "
                           f"lead time {lead_time} day(s), expected {pd.Timestamp(arrival).date()}.")
                st.rerun()

    if st.session_state.placed_orders:
        st.subheader("Placed orders (this session)")
        po_df = pd.DataFrame(st.session_state.placed_orders)
        st.dataframe(po_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Log stock as it's physically received")
    st.markdown(
        "Use this once stock actually arrives at the store (whether from a placed order above or "
        "any other delivery). Every entry adds directly to that Store × SKU's Current Stock and "
        "immediately flows through to Lead-Time, Reorder, Risk and every other tab in this app. "
        "**The quantity below updates its own overstock check live, before you submit anything.**"
    )

    intake_store = st.selectbox("Store", STORES, key="intake_store")
    product_labels = [f"{sku} – {PRODUCT_NAME[sku]}" for sku in PRODUCTS]
    intake_product_label = st.selectbox("Product", product_labels, key="intake_product")
    intake_sku = intake_product_label.split(" – ")[0]
    if "intake_qty" not in st.session_state:
        st.session_state["intake_qty"] = 50
    intake_qty = st.number_input("Quantity received", min_value=0, step=1, key="intake_qty")
    intake_note = st.text_input("Note (optional)", key="intake_note")

    row = full_table[(full_table["Store"] == intake_store) & (full_table["SKU"] == intake_sku)]
    if not row.empty:
        r = row.iloc[0]
        hypothetical_stock = r["Current_Stock"] + intake_qty
        assessment = inv.assess_stock_position(hypothetical_stock, r["Forecast_During_Lead_Time"], r["Safety_Stock"])
        if assessment["status"] == "Overstocked":
            st.warning(f"⚠️ **This would overstock it.** {assessment['message']}")
        elif assessment["status"] == "Understocked":
            st.info(f"ℹ️ {assessment['message']}")
        else:
            st.success(f"✅ **Looks right.** {assessment['message']}")

    if st.button("➕ Log stock receipt", type="primary"):
        st.session_state.stock_ledger.append({
            "store": intake_store, "sku": intake_sku, "qty": int(intake_qty), "note": intake_note,
        })
        st.session_state["_intake_pending_reset"] = {"intake_qty": 50, "intake_note": ""}
        st.success(f"Logged {intake_qty} units of {intake_sku} at {intake_store}.")
        st.rerun()

    st.divider()
    st.subheader("🤖 AI Stock Assessment")
    st.caption("Reads the same demand forecast and reorder-point numbers already calculated "
               "elsewhere in the app — no separate model, no new assumptions.")
    a_store = st.selectbox("Store ", STORES, key="assess_store")
    a_product_label = st.selectbox("Product ", product_labels, key="assess_product")
    a_sku = a_product_label.split(" – ")[0]
    a_row = full_table[(full_table["Store"] == a_store) & (full_table["SKU"] == a_sku)]
    if not a_row.empty:
        r = a_row.iloc[0]
        assessment = inv.assess_stock_position(r["Current_Stock"], r["Forecast_During_Lead_Time"], r["Safety_Stock"])
        st.markdown(f"### {assessment['status']}")
        st.write(assessment["message"])
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Current Stock", f"{r['Current_Stock']:.0f}")
        cB.metric("Forecast During Lead Time", f"{r['Forecast_During_Lead_Time']:.0f}")
        cC.metric("Reorder Point", f"{assessment['reorder_point']:.0f}")
        cD.metric("Overstock Threshold", f"{assessment['overstock_threshold']:.0f}")

    st.divider()
    st.subheader("Intake ledger (this session)")
    if st.session_state.stock_ledger:
        st.dataframe(pd.DataFrame(st.session_state.stock_ledger), use_container_width=True, hide_index=True)
    else:
        st.caption("No manual stock entries logged yet.")

# ============================================================== 5. LEAD-TIME TRACKING
with tab_leadtime:
    st.header("Lead-Time Tracking — All Stores × All SKUs")
    st.caption("Incoming stock shown here is a SIMULATED ESTIMATE, since the dataset does not record pending purchase orders explicitly.")
    cols = ["Store", "Product", "SKU", "Lead_Time_Days", "Forecast_During_Lead_Time", "Current_Stock", "Incoming_Stock_Estimate"]
    st.dataframe(
        full_table[cols].rename(columns={
            "Lead_Time_Days": "Supplier Lead Time (days)",
            "Forecast_During_Lead_Time": "Forecasted Demand During Lead Time",
            "Incoming_Stock_Estimate": "Incoming Stock (est.)",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"As of {AS_OF_DATE.date()} · Forecast period = {forecast_period} days.")

    fig = px.bar(full_table, x="Store", y=["Current_Stock", "Incoming_Stock_Estimate", "Forecast_During_Lead_Time"],
                 facet_col="Product", barmode="group",
                 title="Current Stock vs Forecasted Demand During Lead Time, by Store & SKU")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================== LEAD-TIME ANTICIPATION
with tab_leadtime_anticipation:
    st.header("Lead-Time Anticipation — All Stores × All SKUs")
    st.markdown(
        "The historical dataset records a single **fixed** lead time per SKU. This tab **anticipates** "
        "the next delivery using variability actually observed in the Live Simulation, rather than "
        "repeating that fixed textbook number."
    )
    lt_log = sim_state["lead_time_log"]
    if lt_log.empty:
        st.info("🚚 No delivery history yet — open the 🔴 Live Simulation tab and advance a few days "
                "(or turn on Auto-play). As orders are placed and arrive, this tab will start "
                "anticipating each Store × SKU's real delivery variability from what actually happened.")
        planned = full_table[["Store", "Product", "SKU", "Lead_Time_Days"]].rename(
            columns={"Lead_Time_Days": "Planned Lead Time (days)"})
        st.dataframe(planned, use_container_width=True, hide_index=True)
    else:
        rows = []
        for _, r in full_table.iterrows():
            a = simulation.anticipate_lead_time(lt_log, r["Store"], r["SKU"], r["Lead_Time_Days"])
            rows.append({"Store": r["Store"], "Product": r["Product"], "SKU": r["SKU"],
                         "Observations": a["n_observations"], "Avg Actual Lead Time": round(a["avg_actual"], 1),
                         "Min": a["min_actual"], "Max": a["max_actual"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================== 6. EXPIRY TRACKING
with tab_expiry:
    st.header("Expiry Tracking — All Stores × All SKUs")
    st.warning("⚠️ **Simulated/prototype values.** This dataset is not connected to Colonial Fresh's "
               "live operational systems. Days Remaining reflects the shelf life of the most recently "
               "received stock lot as of the dataset's latest date — a simplification appropriate for this design.")
    cols = ["Store", "Product", "SKU", "Shelf_Life_Days", "Expiry_Date", "Days_Remaining", "Risk"]
    expiry_view = full_table[cols].copy()
    expiry_view["Expiry Risk"] = np.where(expiry_view["Days_Remaining"] <= inv.EXPIRY_RISK_DAYS_THRESHOLD, "Watch", "OK")
    st.dataframe(expiry_view.drop(columns=["Risk"]), use_container_width=True, hide_index=True)
    st.caption(f"As of {AS_OF_DATE.date()}.")

# ============================================================== 7. REORDER RECOMMENDATION
with tab_reorder:
    st.header("Demand-Based Reorder Recommendation")
    st.caption("This is the **Order Management Dashboard**: every store × SKU's current order status, "
               "calculated live from the demand forecast and inventory position — not manually maintained.")
    st.markdown(
        "**Recommended Order = Forecast During Lead Time + Safety Stock − Current Stock − Incoming Stock** "
        "(floored at 0). Safety Stock = adaptively learned rate × Forecast During Lead Time "
        "(20% baseline until Live Simulation has learned otherwise)."
    )
    cols = ["Store", "Product", "SKU", "Forecast_During_Lead_Time", "Safety_Stock", "Safety_Stock_Rate",
            "Current_Stock", "Incoming_Stock_Estimate", "Raw_Calculation", "Recommended_Order", "Risk"]
    st.dataframe(
        full_table[cols].rename(columns={
            "Forecast_During_Lead_Time": "Forecast During Lead Time", "Safety_Stock": "Safety Stock",
            "Safety_Stock_Rate": "Safety Stock Rate", "Current_Stock": "Current Stock",
            "Incoming_Stock_Estimate": "Incoming Stock (est.)", "Raw_Calculation": "Raw Calculation",
            "Recommended_Order": "Recommended Order",
        }),
        use_container_width=True, hide_index=True,
    )
    st.info("The system recommends an order because available inventory is below the expected "
            "demand requirement during the replenishment period.")

# ============================================================== 8. RISK DASHBOARD
with tab_risk:
    st.header("Risk Dashboard")
    st.caption(f"Dynamically calculated for all 6 stores × 3 SKUs — as of {AS_OF_DATE.date()} · "
               f"forecast period = {forecast_period} days.")
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        wastage_by_sku = full_table.groupby("Product")["Wastage_Last30d"].sum().reset_index()
        fig = px.bar(wastage_by_sku, x="Product", y="Wastage_Last30d", color="Product",
                     title="Wastage by SKU (last 30d, current scope)")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        stockout_by_sku = full_table.groupby("Product")["Stockout_Last30d"].sum().reset_index()
        fig = px.bar(stockout_by_sku, x="Product", y="Stockout_Last30d", color="Product",
                     title="Stockout Occurrences by SKU (last 30d, current scope)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================== LIVE SIMULATION
with tab_live:
    st.header("🔴 Live Simulation")
    st.caption("Simulates ongoing grocery operations day-by-day across the full 6-store × 3-SKU "
               "network, adaptively learning each combo's safety-stock rate from real stockout/overstock outcomes.")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("▶️ Next Day"):
        st.session_state.sim_state = simulation.simulate_next_day(st.session_state.sim_state)
        st.rerun()
    st.session_state.sim_autoplay = c2.toggle("Auto-play", value=st.session_state.sim_autoplay)
    st.session_state.sim_speed = c3.select_slider("Days per tick", options=[1, 3, 7], value=st.session_state.sim_speed)
    if c4.button("🔄 Reset to historical data"):
        st.session_state.sim_state = simulation.init_simulation_state(
            HISTORICAL_DF, STORES, PRODUCTS, HISTORICAL_AS_OF_DATE, seed=7)
        st.rerun()

    st.metric("Days simulated", sim_state["days_simulated"])
    st.metric("Current simulated date", str(AS_OF_DATE.date()))

    st.divider()
    st.subheader("📈 Is it actually learning? — stockout rate, start vs now")
    learning_log = sim_state["learning_log"]
    sim_log = sim_state["log"]
    if learning_log.empty:
        st.info("Run a few simulated days to see whether the adaptive safety-stock rate is working.")
    else:
        n_days = sim_state["days_simulated"]
        window = min(30, max(1, n_days // 4))  # compare first vs last quarter (capped at 30 days)
        first_dates = sorted(sim_log["Date"].unique())[:window]
        last_dates = sorted(sim_log["Date"].unique())[-window:]
        start_rate = sim_log[sim_log["Date"].isin(first_dates)]["Stockout"].mean()
        now_rate = sim_log[sim_log["Date"].isin(last_dates)]["Stockout"].mean()
        delta = now_rate - start_rate

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Stockout rate — first {window} simulated days (fixed 20% baseline)", f"{start_rate:.1%}")
        m2.metric(f"Stockout rate — most recent {window} days (adaptive rate)", f"{now_rate:.1%}",
                  delta=f"{delta:+.1%}", delta_color="inverse")
        rates_now = simulation.current_safety_stock_rates(sim_state)
        avg_rate = float(np.mean(list(rates_now.values()))) if rates_now else 0.20
        m3.metric("Avg. adaptive safety-stock rate now", f"{avg_rate:.0%}", delta=f"{avg_rate-0.20:+.0%} vs 20% baseline")

        if delta < 0:
            st.success(f"The adaptive policy is working: the network stockout rate fell by "
                       f"{abs(delta):.1%} (from {start_rate:.1%} to {now_rate:.1%}) as the model raised "
                       "safety-stock rates on combos that actually stocked out.")
        elif n_days < 14:
            st.info("Not enough simulated days yet to draw a conclusion — run more days.")
        else:
            st.warning("The stockout rate has not clearly improved yet on this run — worth investigating further.")

        st.caption(
            "Trade-off to watch: pushing the safety-stock rate up to kill every stockout also raises "
            "the overstock threshold, so a long-running simulation with mostly stockout-driven "
            "adjustments can end up with more Overstock Risk than it started with. That's a real "
            "trade-off the model is making, not a bug."
        )

        st.subheader("Network Stockout Rate over simulated time")
        rolling = simulation.rolling_stockout_rate(sim_log)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling["Date"], y=rolling["Daily_Stockout_Rate"], name="Daily", opacity=0.4))
        fig.add_trace(go.Scatter(x=rolling["Date"], y=rolling["Rolling_Stockout_Rate"], name="7-day rolling avg"))
        fig.update_layout(title="Network Stockout Rate — Daily vs 7-day Rolling Average", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Adaptively learned safety-stock rates (current)")
        rate_df = pd.DataFrame([
            {"Store": s, "SKU": k, "Product": PRODUCT_NAME[k], "Safety Stock Rate": f"{r:.1%}"}
            for (s, k), r in rates_now.items()
        ]).sort_values("Safety Stock Rate", ascending=False)
        st.dataframe(rate_df, use_container_width=True, hide_index=True)

        st.download_button("⬇️ Download learning log (CSV)", learning_log.to_csv(index=False),
                            file_name="learning_log.csv", mime="text/csv")

    if st.session_state.sim_autoplay:
        for _ in range(st.session_state.sim_speed):
            st.session_state.sim_state = simulation.simulate_next_day(st.session_state.sim_state)
        st.rerun()

# ============================================================== SOLUTION ARCHITECTURE
with tab_arch:
    st.header("Solution Architecture")
    st.markdown(
        "**Data & processing pipeline:**\n\n"
        "```\n"
        "Excel Dataset (Colonial Fresh Simulated Data, 2023-01 -> 2025-04)\n"
        "    ↓\n"
        "Data Cleaning & Preparation   (pandas: date conversion, type correction, missing-value handling, validation)\n"
        "    ↓\n"
        "Product Movement Analysis     (average daily Units_Sold per SKU -> Fast/Moderate/Slow classification)\n"
        "    ↓\n"
        "Demand Forecasting            (7-day moving-average baseline; Random Forest ML model, chronological 70/30 split)\n"
        "    ↓\n"
        "Inventory Analysis            (Opening/Incoming/Sold/Closing/Wastage/Stockout monitoring)\n"
        "    ↓\n"
        "Lead-Time & Expiry Tracking   (supplier lead time, forecasted demand during lead time, shelf-life/expiry)\n"
        "    ↓\n"
        "Risk Detection                (Normal / Stockout Risk / Overstock Risk / Expiry Risk classification)\n"
        "    ↓\n"
        "Demand-Based Reorder Calculation  (Forecast during lead time + adaptive safety stock - current - incoming)\n"
        "    ↓\n"
        "Live Simulation                (day-by-day network simulation, adaptive safety-stock learning)\n"
        "    ↓\n"
        "Streamlit Dashboard            (Overview, Movement, Forecasting, ML Forecast, Inventory, Stock Intake, "
        "Lead-Time, Expiry, Reorder, Risk, Live Simulation views)\n"
        "```"
    )
    st.markdown(
        "**Technology stack:** Python, Pandas (data handling), scikit-learn (Random Forest forecasting model), "
        "Streamlit (interactive dashboard UI), Plotly (interactive charts)."
    )
    st.markdown(
        "**Sprint 3 upgrade, addressing Sprint 2 lecturer feedback:**\n"
        "1. Dataset sized to a chronological span (2023-01 to 2025-04) so a **70/30 train/test split** "
        "falls naturally at roughly 2023-01 → 2024-08 (train) and 2024-08 → 2025-04 (test).\n"
        "2. A **tracking signal** (RSFE ÷ cumulative MAD) is calculated on the 30% test set to detect whether "
        "the model is systematically over/under-forecasting, not just how large its average error is.\n"
        "3. The **Order Management Dashboard** (Reorder Recommendation tab) shows the live reorder calculation "
        "for every store × SKU combination.\n"
        "4. A bias-correction attempt is reported honestly — including when it does not clearly work."
    )
