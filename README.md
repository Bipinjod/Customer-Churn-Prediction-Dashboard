# 📉 Customer Churn Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Description

Customer churn — when a customer stops using a service — is one of the most costly problems in the telecom industry. This project builds an end-to-end machine learning pipeline to **predict which customers are likely to churn**, helping businesses take proactive retention actions. It includes exploratory data analysis, multi-model training with automated selection, and an interactive Streamlit dashboard for real-time what-if prediction.

---

## 🚀 Live Demo

> **Coming soon** — deployment in progress on Streamlit Cloud.

---

## ✨ Features

- 📊 **Exploratory Data Analysis** — class imbalance, churn by contract type, monthly charges, tenure, and payment method
- 🤖 **Multiple ML Models** — Logistic Regression, Decision Tree, and Random Forest trained with `class_weight='balanced'`
- 🏆 **Auto Model Selection** — best model chosen automatically based on F1 Score
- 📈 **Feature Importance** — top drivers of churn visualised as an interactive Plotly bar chart
- 🖥️ **Streamlit Dashboard** — real-time churn probability with colour-coded risk levels (Low / Medium / High)
- 🎛️ **What-If Simulator** — adjust any customer attribute via the sidebar and see predictions update instantly

---

## 🛠️ Tech Stack

| Category    | Tools / Libraries                                      |
|-------------|--------------------------------------------------------|
| Language    | Python 3.10                                            |
| ML & Data   | scikit-learn, pandas, NumPy, joblib                    |
| Visualisation | Matplotlib, Seaborn, Plotly                          |
| Dashboard   | Streamlit                                              |
| Notebook    | Jupyter Notebook                                       |
| IDE         | Visual Studio Code                                     |
| Version Control | Git & GitHub                                       |

---

## 📁 Project Structure

```
churn_project/
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── data/
│   └── telco_churn.csv         # Raw dataset (IBM Telco)
├── models/
│   ├── best_model.pkl          # Saved best ML model
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── feature_importance.csv  # Feature importance scores
├── notebooks/
│   └── analysis.ipynb          # EDA notebook
├── src/
│   ├── preprocess.py           # Data cleaning, encoding & splitting
│   └── train.py                # Model training & evaluation
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Bipinjod/Customer-Churn-Prediction-Dashboard.git
cd Customer-Churn-Prediction-Dashboard
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run preprocessing & training
```bash
python src/preprocess.py   # Cleans data and saves scaler.pkl
python src/train.py        # Trains models and saves best_model.pkl
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 ML Model Results

> Metrics evaluated on a **20% held-out test set** with `random_state=42`.

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7264   | 0.4909    | 0.7941 | 0.6067   | 0.8337  |
| Decision Tree       | 0.7271   | 0.4868    | 0.4947 | 0.4907   | 0.6532  |
| Random Forest       | 0.7825   | 0.6164    | 0.4813 | 0.5405   | 0.8121  |

> Run `python src/train.py` to populate these results in your terminal.

---

## 🔧 Recent Fixes & Improvements

### EDA Notebook (`notebooks/analysis.ipynb`)
- **Path fix** — replaced hardcoded machine-specific path with `os.chdir()` + relative `Path("data/telco_churn.csv")` for portability
- **FutureWarning (Section 2)** — added `hue="Churn"` and `legend=False` to `sns.countplot` to comply with seaborn's updated API
- **UserWarning (Sections 4 & 5)** — removed redundant `ax.legend()` calls; `sns.histplot` with `hue=` manages its own legend automatically

### Dashboard (`dashboard/app.py`)
- **Scaler feature-name warning** — `scaler.transform()` now receives a named `pd.DataFrame` (with `FEATURE_ORDER` column names) instead of a plain NumPy array
- **Deprecated `use_container_width`** — replaced all `use_container_width=True` with `width='stretch'` across all `st.plotly_chart()` calls

---

## 💡 Key Insights from EDA

- 📌 **~26% churn rate** — significant class imbalance requiring balanced training strategies
- 📋 **Month-to-month contracts** have the highest churn; two-year contracts show the lowest
- 💸 **Churned customers pay higher monthly charges** on average, suggesting price sensitivity
- ⏳ **New customers (tenure < 12 months) churn the most** — the first year is the critical retention window
- 💳 **Electronic check users churn more** than customers on automatic payment methods

---

## 👤 Author

**Bipin Jod**
BSc Computing Student | Internship Project

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/Bipinjod)

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
