# Customer Churn Analysis & Prediction

Project end-to-end untuk memprediksi customer churn pelanggan Telco — mencakup analisis data,
machine learning, dashboard BI, dan aplikasi prediksi yang bisa diakses publik.

🔗 **Live App**: [isi link Streamlit Cloud kamu di sini setelah deploy]

## Struktur Repository

```
telco-churn-analysis/
├── streamlit_telco_churn.py       # App Streamlit (Home, EDA, Model Comparison, Prediksi, dll)
├── telco_churn_artifacts.pkl      # Model & artifact hasil training (dipakai app, TIDAK training ulang)
├── requirements.txt               # Dependency untuk menjalankan app
├── notebooks/
│   └── telco_churn_analysis.ipynb # Notebook lengkap: EDA, preprocessing, modeling, evaluasi
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (IBM Telco Customer Churn)
├── dashboard/
│   └── Customer_Churn_dashboard.twb          # Dashboard Tableau
└── docs/
    └── Customer_Churn_Portfolio.pptx          # Slide presentasi portofolio
```

> **Catatan struktur:** `streamlit_telco_churn.py` dan `telco_churn_artifacts.pkl` sengaja diletakkan
> di root (bukan di dalam subfolder), karena kode app memuat model dengan relative path
> (`joblib.load("telco_churn_artifacts.pkl")`). Kalau dipindah ke subfolder, path itu perlu
> disesuaikan juga di kode.

## Cara Menjalankan Secara Lokal

```bash
git clone https://github.com/USERNAME/telco-churn-analysis.git
cd telco-churn-analysis
pip install -r requirements.txt
streamlit run streamlit_telco_churn.py
```

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Optuna, SMOTE (imbalanced-learn), Streamlit, Tableau

## Ringkasan Hasil
- Dataset: 7.043 pelanggan, 21 fitur, churn rate 26,5%
- Model final: Logistic Regression — Accuracy 78,6%, AUC-ROC 0,84
- Insight utama: kontrak bulanan, tenure pendek, dan pembayaran Electronic Check adalah
  prediktor churn terkuat

## Deploy Sendiri (Streamlit Community Cloud)
1. Fork/clone repo ini ke akun GitHub kamu.
2. Buka [share.streamlit.io](https://share.streamlit.io) → login GitHub → **Create app**.
3. Repository: repo kamu | Branch: `main` | Main file path: `streamlit_telco_churn.py`
4. Klik **Deploy**.
