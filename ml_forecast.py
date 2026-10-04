"""
Machine-learning demand forecast — Random Forest, network-wide, per SKU.

Sprint 3 upgrade per lecturer feedback:
  - Explicit chronological 70% train / 30% test split (never random —
    shuffling would leak future information into training).
  - Dataset date range (2023-01-01 -> 2026-04-30) is deliberately sized so
    this 70/30 split lands training on 2023-01-01 -> 2025-04-30 and testing
    on 2025-05-01 -> 2026-04-30.
  - Tracking signal (RSFE / cumulative MAD) to detect systematic
    over/under-forecasting that MAE/RMSE alone can't reveal.
  - A held-out slice inside the 70% training window (never shown to
    model.fit()) is used to attempt a bias correction; the result is
    reported honestly, whether or not it actually helps.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TRAIN_SPLIT_RATIO = 0.70
TEST_SPLIT_RATIO = 0.30
TRACKING_SIGNAL_LIMIT = 4.0
N_ESTIMATORS = 200
RANDOM_STATE = 42

FEATURE_COLS = [
    "lag_1", "lag_7", "roll_7_mean", "roll_14_mean", "roll_7_std",
    "dayofweek", "is_weekend", "Promotion_Rate",
]


def _network_daily_series(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    d = df[df["Product_ID"] == sku].groupby("Date").agg(
        Units_Sold=("Units_Sold", "sum"),
        Promotion_Rate=("Promotion", "mean"),
    ).reset_index().sort_values("Date")
    return d


def _engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["lag_1"] = d["Units_Sold"].shift(1)
    d["lag_7"] = d["Units_Sold"].shift(7)
    d["roll_7_mean"] = d["Units_Sold"].shift(1).rolling(7).mean()
    d["roll_14_mean"] = d["Units_Sold"].shift(1).rolling(14).mean()
    d["roll_7_std"] = d["Units_Sold"].shift(1).rolling(7).std()
    d["dayofweek"] = d["Date"].dt.dayofweek
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d = d.dropna().reset_index(drop=True)
    return d


def _baseline_predict(d: pd.DataFrame) -> pd.Series:
    """7-day moving-average baseline evaluated on the same rows as the ML test set."""
    return d["roll_7_mean"]


def _metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _tracking_signal(actual, pred) -> np.ndarray:
    """Running Sum of Forecast Errors / cumulative Mean Absolute Deviation.
    Detects whether a model is systematically over- or under-forecasting,
    which MAE/RMSE alone cannot reveal."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    errors = actual - pred
    rsfe = np.cumsum(errors)
    cum_mad = np.cumsum(np.abs(errors)) / np.arange(1, len(errors) + 1)
    cum_mad_safe = np.where(cum_mad == 0, np.nan, cum_mad)
    return rsfe / cum_mad_safe


def train_and_backtest(df: pd.DataFrame, sku: str) -> dict:
    daily = _network_daily_series(df, sku)
    feat = _engineer_features(daily)
    n_total = len(feat)

    split_idx = int(round(n_total * TRAIN_SPLIT_RATIO))
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]

    # Hold out the last 15% of the TRAINING window purely to estimate a bias
    # correction — this slice is never passed to model.fit().
    fit_split = int(round(len(train) * 0.85))
    fit_train, bias_val = train.iloc[:fit_split], train.iloc[fit_split:]

    X_train, y_train = fit_train[FEATURE_COLS], fit_train["Units_Sold"]
    X_test, y_test = test[FEATURE_COLS], test["Units_Sold"]

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    if len(bias_val) > 0:
        bias_val_pred = model.predict(bias_val[FEATURE_COLS])
        bias_val_resid = bias_val["Units_Sold"].values - bias_val_pred
        overall_bias_correction = float(np.mean(bias_val_resid))
        bias_by_dow = {}
        for dow in range(7):
            mask = bias_val["dayofweek"].values == dow
            bias_by_dow[dow] = float(np.mean(bias_val_resid[mask])) if mask.sum() > 0 else overall_bias_correction
    else:
        overall_bias_correction = 0.0
        bias_by_dow = {dow: 0.0 for dow in range(7)}

    rf_pred_test = model.predict(X_test)
    dow_corrections = test["dayofweek"].map(bias_by_dow).values
    rf_pred_test_corrected = rf_pred_test + dow_corrections
    baseline_pred_test = _baseline_predict(test).values

    rf_metrics = _metrics(y_test, rf_pred_test)
    rf_metrics_corrected = _metrics(y_test, rf_pred_test_corrected)
    baseline_metrics = _metrics(y_test, baseline_pred_test)

    rf_ts = _tracking_signal(y_test, rf_pred_test)
    rf_ts_corrected = _tracking_signal(y_test, rf_pred_test_corrected)
    baseline_ts = _tracking_signal(y_test, baseline_pred_test)

    def breaches(ts):
        return int(np.nansum(np.abs(ts) > TRACKING_SIGNAL_LIMIT))

    feature_importances = dict(zip(FEATURE_COLS, model.feature_importances_))

    return {
        "sku": sku,
        "n_total": n_total,
        "train_days": len(fit_train),
        "bias_val_days": len(bias_val),
        "test_days": len(test),
        "train_split_ratio": TRAIN_SPLIT_RATIO,
        "train_date_range": (train["Date"].min(), train["Date"].max()),
        "test_date_range": (test["Date"].min(), test["Date"].max()),
        "bias_correction": overall_bias_correction,
        "bias_by_dow": bias_by_dow,
        "test_dates": test["Date"].values,
        "test_actual": y_test.values,
        "rf_pred": rf_pred_test,
        "rf_pred_corrected": rf_pred_test_corrected,
        "baseline_pred": baseline_pred_test,
        "rf_metrics": rf_metrics,
        "rf_metrics_corrected": rf_metrics_corrected,
        "baseline_metrics": baseline_metrics,
        "rf_tracking_signal": rf_ts,
        "rf_tracking_signal_corrected": rf_ts_corrected,
        "baseline_tracking_signal": baseline_ts,
        "rf_ts_breaches": breaches(rf_ts),
        "rf_ts_breaches_corrected": breaches(rf_ts_corrected),
        "baseline_ts_breaches": breaches(baseline_ts),
        "feature_importances": feature_importances,
        "model": model,
        "full_features": feat,
    }


def forecast_forward(df: pd.DataFrame, sku: str, model_bundle: dict, as_of_date, forecast_days: int) -> pd.DataFrame:
    """Recursive multi-step forecast from as_of_date forward, applying the
    per-weekday bias correction learned during backtest."""
    model = model_bundle["model"]
    bias_by_dow = model_bundle["bias_by_dow"]
    feat = model_bundle["full_features"]

    history = feat[feat["Date"] <= as_of_date].copy()
    if history.empty:
        history = feat.copy()

    series = list(history["Units_Sold"].values)
    dates = list(history["Date"].values)
    promo_rate = float(history["Promotion_Rate"].tail(14).mean()) if len(history) else 0.0

    preds = []
    cur_date = pd.Timestamp(as_of_date)
    for _ in range(forecast_days):
        cur_date = cur_date + pd.Timedelta(days=1)
        lag_1 = series[-1]
        lag_7 = series[-7] if len(series) >= 7 else series[0]
        roll_7 = np.mean(series[-7:]) if len(series) >= 7 else np.mean(series)
        roll_14 = np.mean(series[-14:]) if len(series) >= 14 else np.mean(series)
        roll_7_std = np.std(series[-7:]) if len(series) >= 7 else 0.0
        dow = cur_date.dayofweek
        is_weekend = int(dow >= 5)

        X = pd.DataFrame([{
            "lag_1": lag_1, "lag_7": lag_7, "roll_7_mean": roll_7,
            "roll_14_mean": roll_14, "roll_7_std": roll_7_std,
            "dayofweek": dow, "is_weekend": is_weekend, "Promotion_Rate": promo_rate,
        }])[FEATURE_COLS]

        pred = float(model.predict(X)[0]) + bias_by_dow.get(dow, 0.0)
        pred = max(0.0, pred)
        preds.append({"Date": cur_date, "ML_Forecast": pred})
        series.append(pred)

    return pd.DataFrame(preds)
