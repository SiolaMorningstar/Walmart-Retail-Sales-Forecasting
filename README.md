# 🛒 Walmart Retail Sales Forecasting

## 📌 Project Overview

This project focuses on building an end-to-end retail sales forecasting system using classical time series models and modern forecasting techniques.

The objective is not only to predict weekly sales accurately, but also to:

- Understand trend and seasonality patterns
- Identify anomalies
- Quantify seasonality strength
- Evaluate macroeconomic impact
- Simulate off-season uplift strategies
- Translate forecasts into business insights

This project demonstrates the complete lifecycle of a data science forecasting problem — from exploration to business storytelling.

---

## 🎯 Business Problem

Retail sales are highly seasonal and influenced by holidays, promotions, and economic conditions.

Key questions addressed:

- How strong is yearly seasonality?
- Why do sales dip before the holiday spike?
- Can macroeconomic variables improve predictions?
- How can off-season performance be improved?
- Which forecasting model performs best?

---

## 📊 Dataset Description

Weekly Walmart sales data (2010–2012) including:

- `Weekly_Sales`
- `Temperature`
- `Fuel_Price`
- `CPI`
- `Unemployment`
- `Holiday_Flag`

Frequency: Weekly  
Seasonality: Yearly (52-week cycle)

---

## 🔎 Exploratory Data Analysis (EDA)

Key findings:

- Strong December sales spikes (Christmas effect)
- Structural drop in January
- Stable recurring yearly seasonality
- Moderate correlations with macroeconomic indicators
- Sales are more seasonality-driven than macro-driven

---

## 📈 Classical & Statistical Models

The following baseline and statistical models were implemented:

- Naïve Forecast
- Moving Average
- Rolling Mean
- ARIMA
- SARIMA
- Holt-Winters (Triple Exponential Smoothing)

### 📊 Model Performance (MAPE Approx.)

| Model | MAPE |
|-------|------|
| Naïve | ~7% |
| Moving Average | ~3% |
| ARIMA | ~3–4% |
| SARIMA | ~2.83% |
| Holt-Winters | **~1.94%** |

### Key Insight

Models that explicitly model seasonality significantly outperform simple autoregressive models.

Holt-Winters performed best among classical approaches due to stable yearly seasonality.

---

## 🚀 Prophet Model (Meta)

Facebook/Meta Prophet was implemented to:

- Capture trend
- Model yearly seasonality
- Provide uncertainty intervals
- Improve interpretability

**MAPE ≈ 2.35%**

Prophet provides component-level insights:

- Trend
- Yearly Seasonality
- Holiday Effects
- Uncertainty Intervals

---

## 📊 Prophet + Regressors

External regressors added:

- Temperature
- Fuel Price
- CPI
- Unemployment
- Holiday Flag

Result:

Performance slightly decreased (MAPE ≈ 3.38%), indicating macroeconomic variables have limited short-term predictive impact compared to seasonality.

Conclusion:

Retail sales in this dataset are strongly seasonality-dominated rather than macro-driven.

---

## 📏 Seasonality Strength Quantification

Seasonality strength was calculated to measure how much variance is explained by seasonal components.

Result:

Yearly seasonality explains a significant portion of total variance, confirming that December peaks and January drops are structural patterns.

---

## 🔮 Scenario Simulation – Off-Season Uplift

To move beyond forecasting, scenario simulations were performed to evaluate:

- Potential uplift in non-holiday months
- Promotional intervention impact
- Revenue improvement strategies

This transforms forecasting into business decision support.

---

## 💡 Business Insights

1. End-of-year period is consistently the strongest revenue window.
2. January sales dip is predictable and structural.
3. Macro indicators show weak short-term forecasting power.
4. Seasonality is the dominant sales driver.
5. Off-season optimization presents growth opportunity.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Statsmodels
- Prophet
- Matplotlib
- Seaborn
- Google Colab

---

## 📂 Project Structure
Walmart-Retail-Forecasting/
│
├── Walmart_Retail_Sales_Forecasting.ipynb
├── README.md
├── requirements.txt
└── data/


---

## 📌 Key Learnings

- Importance of stationarity in ARIMA
- When to use SARIMA vs Holt-Winters
- How Prophet decomposes trend and seasonality
- Why adding regressors doesn’t always improve performance
- Translating statistical models into business insights

---

## 🚀 Future Improvements

- Deploy as a Streamlit dashboard
- Add automated hyperparameter tuning
- Build store-level forecasting
- Integrate promotion calendar effects
- Convert into production API

---

## 📜 License

This project is licensed under the MIT License.

---

## 🤝 Connect

If you found this project interesting or have suggestions for improvement, feel free to connect and collaborate.
