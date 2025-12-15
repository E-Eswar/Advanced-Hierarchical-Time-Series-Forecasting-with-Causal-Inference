# Advanced Hierarchical Time Series Forecasting with Causal Inference

This project implements an **end-to-end Hierarchical Time Series (HTS) forecasting pipeline**
combined with **explicit causal inference using a structural state-space model**.

The solution goes beyond basic ARIMA or deep learning approaches by:
- Enforcing **hierarchical coherence** through bottom-up reconciliation
- Using **Structural Time Series (STS)** models for explainable forecasting
- Incorporating **intervention analysis** with counterfactual forecasting
- Evaluating forecasts using **MASE** and **wMAPE**

The project is implemented as a **single Python script (`.py`)** suitable for academic
submission and non-notebook execution.

---

## 📂 Project Structure

submission/<br>
├── hts_causal_forecasting.py # Main executable script <br>
├── requirements.txt # Python dependencies<br>
└── README.md # Project documentation


---

## 🧠 Problem Overview

The objective is to forecast sales data across a **three-level hierarchy**:

- **Level 1:** Total sales  
- **Level 2:** Regional sales (North, South)  
- **Level 3:** Product-level sales (P1–P5)  

Additionally, the project simulates a **marketing intervention** and estimates its
**causal impact** using counterfactual forecasting.

---

## 🏗️ Methodology

### 1️⃣ Synthetic Data Generation
- Weekly data for 3 years
- Components:
  - Linear trend
  - Annual seasonality (52 weeks)
  - Random noise
- A simulated marketing intervention active during a fixed window

---

### 2️⃣ Hierarchical Modeling
- Bottom-level product series are generated first
- Regional and total series are constructed via aggregation
- Hierarchy is strictly respected

---

### 3️⃣ Forecasting Model
- **Structural Time Series (STS)** using `statsmodels`
- Local linear trend + stochastic seasonal component
- Bottom-level series are forecast independently

---

### 4️⃣ Hierarchical Reconciliation
- **Bottom-up reconciliation**
- Guarantees:

  - Sum(products) → regions → total

  - Trade-off between coherence and aggregate accuracy is explicitly evaluated

---

### 5️⃣ Causal Inference

#### Granger Causality (Diagnostic Only)
- Tests predictive causality between regional and total series
- Expected to be non-significant due to deterministic aggregation
- Included to highlight methodological limitations

#### Intervention Analysis (Primary Causal Method)
- Marketing campaign modeled as an **exogenous regressor**
- Structural Time Series with intervention:

  - Sales = Trend + Seasonality + β × Intervention + Noise

- **Counterfactual forecasting** is used (not in-sample prediction):
  - Scenario A: Intervention ON
  - Scenario B: Intervention OFF
  - Difference between forecasts = causal impact

---

### 6️⃣ Evaluation Metrics
- **MASE (Mean Absolute Scaled Error)**
- **wMAPE (Weighted Mean Absolute Percentage Error)**
- Metrics are computed for:
  - Reconciled forecasts
  - Unreconciled forecasts
- Results confirm expected HTS behavior:
  - Identical errors at bottom level
  - Higher-level trade-offs after reconciliation

---

## ▶ How to Run

### 1️⃣ Create a virtual environment (optional)
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```


### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the script
```bash
python hts_causal_forecasting.py
```

## 📊 Outputs
### The script prints:
- Granger causality test results
- Structural Time Series model summary
- Estimated intervention effect and p-value 
- Counterfactual causal impact (weekly & total) 
- Forecast accuracy metrics (MASE, wMAPE)

### A plot is also generated showing:
- Forecast with intervention
- Forecast without intervention

## ⚠ Notes on Warnings

### You may see warnings related to:
- Sparse matrix efficiency
- Covariance estimation methods

These are performance or informational warnings from ```statsmodels``` and do not affect model correctness, forecasts, or causal conclusions.

## ✅ Key Takeaways

- Hierarchical coherence is enforced through reconciliation
- Structural models provide interpretability
- Causal impact is estimated using valid counterfactual reasoning
- The solution is suitable for advanced academic evaluation and decision-oriented analysis

## 🧪 Tested Environment

- Python 3.11
- Windows
