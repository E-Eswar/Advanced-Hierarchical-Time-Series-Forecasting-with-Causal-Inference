"""
Advanced Time Series Forecasting with Hierarchical Modeling and Causal Inference (End-to-End Script)

Author: E. Eswar
Date: Dec 15, 2025
"""

# =========================================================
# STEP 0: Imports & Reproducibility
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


np.random.seed(42)


# =========================================================
# STEP 1: Synthetic Hierarchical Data Generation
# =========================================================

n_periods = 156          # 3 years weekly
forecast_horizon = 52

dates = pd.date_range("2021-01-01", periods=n_periods, freq="W")

products = ["P1", "P2", "P3", "P4", "P5"]
regions = {
    "North": ["P1", "P2", "P3"],
    "South": ["P4", "P5"]
}

# Trend & seasonality
trend = np.linspace(50, 120, n_periods)
seasonality = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 52)

# Simulated intervention (marketing campaign)
intervention_effect = np.zeros(n_periods)
intervention_effect[80:100] = 25


# =========================================================
# STEP 2: Bottom-Level Series Generation
# =========================================================

data = {}

for p in products:
    noise = np.random.normal(0, 5, n_periods)
    scale = np.random.uniform(0.8, 1.2)
    series = scale * (trend + seasonality + intervention_effect) + noise
    data[p] = np.maximum(series, 0)

df_bottom = pd.DataFrame(data, index=dates)


# =========================================================
# STEP 3: Build the Hierarchy
# =========================================================

df_region = pd.DataFrame({
    region: df_bottom[plist].sum(axis=1)
    for region, plist in regions.items()
})

df_total = pd.DataFrame({
    "Total": df_region.sum(axis=1)
})

df_all = pd.concat([df_total, df_region, df_bottom], axis=1)


# =========================================================
# STEP 4: Explicit Intervention Indicator (Causal Variable)
# =========================================================

df_all["Intervention"] = 0
df_all.iloc[80:100, df_all.columns.get_loc("Intervention")] = 1


# =========================================================
# STEP 5: Train / Test Split (No Leakage)
# =========================================================

train = df_all.iloc[:-forecast_horizon]
test = df_all.iloc[-forecast_horizon:]


# =========================================================
# STEP 6: Base Structural Time Series Forecast Function
# =========================================================

def forecast_series(y, steps):
    model = UnobservedComponents(
        y,
        level="local linear trend",
        seasonal=52
    )
    res = model.fit(disp=False)
    return res.forecast(steps)


# =========================================================
# STEP 7: Bottom-Up Hierarchical Forecasting
# =========================================================

# Bottom level
bottom_forecasts = {}
for p in products:
    bottom_forecasts[p] = forecast_series(train[p], forecast_horizon)

df_forecast_bottom = pd.DataFrame(bottom_forecasts, index=test.index)

# Reconciliation
df_forecast_region = pd.DataFrame({
    region: df_forecast_bottom[plist].sum(axis=1)
    for region, plist in regions.items()
})

df_forecast_total = pd.DataFrame({
    "Total": df_forecast_region.sum(axis=1)
})


# =========================================================
# STEP 8: Unreconciled Forecasts (Benchmark)
# =========================================================

df_unrec = {}
for col in train.columns.drop("Intervention"):
    df_unrec[col] = forecast_series(train[col], forecast_horizon)

df_unrec = pd.DataFrame(df_unrec, index=test.index)


# =========================================================
# STEP 9: Granger Causality (Diagnostic Only)
# =========================================================

print("\nGranger Causality Test: North → Total\n")

gc_data = df_region["North"].to_frame("North")
gc_data["Total"] = df_total["Total"]

grangercausalitytests(
    gc_data[["Total", "North"]],
    maxlag=8
)


# =========================================================
# STEP 10: TRUE CAUSAL MODEL (Structural + Intervention)
# =========================================================

model_intervention = UnobservedComponents(
    train["Total"],
    level="local linear trend",
    seasonal=52,
    exog=train[["Intervention"]]
)

res_intervention = model_intervention.fit(disp=False)

print("\nCausal Model Summary:\n")
print(res_intervention.summary())

beta = res_intervention.params["beta.Intervention"]
p_value = res_intervention.pvalues["beta.Intervention"]

print(f"\nEstimated Intervention Effect (beta): {beta:.2f}")
print(f"P-value: {p_value:.4e}")


# =========================================================
# STEP 11: Counterfactual Forecasting (Correct Method)
# =========================================================

future_intervention_on = np.ones((forecast_horizon, 1))
future_intervention_off = np.zeros((forecast_horizon, 1))

forecast_with_intervention = res_intervention.forecast(
    steps=forecast_horizon,
    exog=future_intervention_on
)

forecast_without_intervention = res_intervention.forecast(
    steps=forecast_horizon,
    exog=future_intervention_off
)

causal_impact = forecast_with_intervention - forecast_without_intervention

avg_weekly_uplift = causal_impact.mean()
total_uplift = causal_impact.sum()

print("\nCausal Impact Results:")
print(f"Average Weekly Uplift: {avg_weekly_uplift:.2f}")
print(f"Total Campaign Uplift: {total_uplift:.2f}")


# =========================================================
# STEP 12: Forecast Accuracy Metrics
# =========================================================

def mase(y_true, y_pred, y_train):
    naive_error = np.mean(np.abs(np.diff(y_train)))
    return np.mean(np.abs(y_true - y_pred)) / naive_error

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


metrics = []

for col in test.columns.drop("Intervention"):

    if col == "Total":
        rec_pred = df_forecast_total["Total"]
    elif col in df_region.columns:
        rec_pred = df_forecast_region[col]
    else:
        rec_pred = df_forecast_bottom[col]

    unrec_pred = df_unrec[col]

    metrics.append({
        "Series": col,
        "MASE_Reconciled": mase(test[col], rec_pred, train[col]),
        "MASE_Unreconciled": mase(test[col], unrec_pred, train[col]),
        "wMAPE_Reconciled": wmape(test[col], rec_pred),
        "wMAPE_Unreconciled": wmape(test[col], unrec_pred),
    })

metrics_df = pd.DataFrame(metrics)

print("\nForecast Accuracy Metrics:\n")
print(metrics_df)


# =========================================================
# STEP 13: Executive Visualization
# =========================================================

plt.figure(figsize=(10, 4))
plt.plot(forecast_with_intervention, label="With Intervention")
plt.plot(forecast_without_intervention, linestyle="--", label="Without Intervention")
plt.legend()
plt.title("Counterfactual Causal Impact of Marketing Campaign")
plt.tight_layout()
plt.show()
