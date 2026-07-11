# Sales Forecasting & Demand Intelligence System

An end-to-end retail and e-commerce forecasting system designed to predict product demand, detect sales anomalies, cluster products by customer demand profiles, and deliver an interactive operational dashboard.

This repository implements the requirements for the **Final Internship Project (Week 3 & Week 4)** for Xylofy AI.

---

## 📂 Project Directory Structure

The project has been organized into logical sub-directories to maintain modularity:

```text
├── Analysis/
│   ├── Train.csv                     # Primary Superstore sales dataset
│   ├── analysis.py                   # Complete data exploration and modeling pipeline
│   ├── model_comparison.csv          # Pre-computed model errors and projections
│   ├── detected_anomalies.csv        # Pre-computed weekly anomalies dataset
│   └── product_segmentation.csv      # Pre-computed product clusters dataset
│
├── Dashboard/
│   ├── Train.csv                     # Dataset mirror for dashboard caching
│   ├── app.py                        # Streamlit interactive dashboard code
│   └── requirements.txt              # Project library dependencies
│
├── charts/                           # 12 analytical plots (saved automatically)
│   ├── overall_monthly_trend.png
│   ├── sales_decomposition.png
│   ├── monthly_sales_seasonality.png
│   ├── sarima_forecast.png
│   ├── prophet_forecast.png
│   ├── prophet_components.png
│   ├── xgboost_forecast.png
│   ├── segment_forecasts.png
│   ├── isolation_forest_anomalies.png
│   ├── zscore_anomalies.png
│   ├── kmeans_elbow.png
│   └── kmeans_clusters.png
│
├── generate_report.py                # ReportLab script to generate PDF summary
├── summary.pdf                       # Styled 2-page executive business report
└── README.md                         # Project documentation
```

---

## 🚀 Setup & Installation

### 1. Prerequisite
Ensure you have **Python 3.8+** installed on your system.

### 2. Create a Virtual Environment (Recommended)
Open your terminal in the project root directory and execute:
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all package requirements via pip:
```bash
pip install -r Dashboard/requirements.txt
```

---

## 💻 How to Run the Applications

### Run the Analytical Pipeline (Tasks 1 to 6)
To re-run data merging, stationarity tests, SARIMA/Prophet/XGBoost training, anomaly checks, and K-Means clustering:
```bash
cd Analysis
python analysis.py
```
*Note: This script automatically downloads the secondary Video Game Sales dataset via `kagglehub`, scales and merges it under the Technology category, performs all modeling, and writes output files and plots to `/charts`.*

### Compile the Executive Business Report (Task 8)
To regenerate the beautiful 2-page PDF report (`summary.pdf`) for the CFO and Head of Supply Chain:
```bash
# Execute from workspace root
python generate_report.py
```

### Run the Streamlit Dashboard (Task 7)
To launch the interactive dashboard locally:
```bash
cd Dashboard
streamlit run app.py
```
*This spins up a local web server (usually at `http://localhost:8501`) displaying the dynamic multi-page dashboard.*

---

## 📊 Analytical Pipeline Breakdown

### **Task 1: Loading, Merging & EDA**
* **Primary Dataset**: `Train.csv` (Superstore Sales over 4 years).
* **Secondary Dataset**: `vgsales.csv` (Video Game Sales, merged as a subcategory under the "Technology" category to simulate multi-source inventory consolidation).
* **Findings**:
  * **Top Revenue Category**: Technology ($1.16M).
  * **Most Consistent Region**: East Region (lowest YoY growth variance).
  * **Fulfillment Latency**: 3.97 days average (highly uniform across regions).
  * **Seasonality**: November and December consistently spike across all years, with January seeing a 50% slump.

### **Task 2: Time Series Decomposition & Stationarity**
* **Decomposition**: Additive decomposition isolates an upward trend, strong yearly seasonality, and residuals that variance-spike in Q4.
* **Stationarity (ADF)**: The monthly sales sequence is stationary ($p$-value = $0.0305 < 0.05$).

### **Task 3: Forecasting Model Comparison**
Models were evaluated on a 3-month validation backtest:

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute Percentage Error (MAPE) |
| :--- | :---: | :---: | :---: |
| **SARIMA** | $19,788.10 | $20,959.33 | 20.78% |
| **Prophet** | $20,172.42 | $21,474.86 | 21.97% |
| **XGBoost (Recursive)** | **$14,454.01** | **$16,129.09** | **14.74%** |

* **XGBoost (Recursive)** is recommended for operational use due to the lowest validation error (14.74% MAPE).

### **Task 5: Anomaly Detection**
* **Isolation Forest**: ML-based method using weekly volume, rolling average, and volatility features.
* **Z-Score**: Flags weeks deviating $> 2\sigma$ from the rolling average.
* Overlapping outliers correspond to Black Friday promo spikes and winter storm logistical lulls.

### **Task 6: Product Demand Clustering**
K-Means clustering ($K=4$) segments the portfolio into:
1. **High Volume, Stable**: Automated weekly restocking (Chairs, Phones, Binders).
2. **High Growth, Emerging**: Agile, short-contract buffers (Copiers, Video Games).
3. **Steady, Medium Volume**: Standard Min-Max cycles (Appliances, Art).
4. **Low Volume, Volatile**: Drop-ship or pull-replenishment to avoid holding costs (Bookcases, Fasteners).

---

## 🏆 Dashboard Features (`app.py`)
1. **Sales Overview**: Total metrics, annual revenue bars, region/category filters, category sunburst breakdown, and monthly timeline sliders.
2. **Forecast Explorer**: Dynamic forecasts using SARIMA, Prophet, or XGBoost for any Category or Region segment, showing validation error metrics and forecast confidence intervals.
3. **Anomaly Report**: Timeline scatter plot mapping sales anomalies with dynamic explanations.
4. **Demand Segments**: Interactive PCA 2D scatter plots of product clusters and custom supply chain stocking actions.
