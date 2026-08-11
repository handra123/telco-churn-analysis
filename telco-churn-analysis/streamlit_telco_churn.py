# ============================================================
#  TELCO CUSTOMER CHURN PREDICTION
#  Streamlit Web Application
#  Dataset  : IBM Telco Customer Churn
#  Model    : Logistic Regression (Optuna Tuned)
#  AUC-ROC  : 0.8394
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

import joblib

# Catatan: hanya butuh fungsi metrik untuk MENAMPILKAN hasil (dihitung dari
# y_test/y_pred/y_prob yang sudah disimpan), bukan untuk training ulang.
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Path artifact hasil training di notebook (lihat cell "Simpan Artifact untuk Deployment")
ARTIFACT_PATH = "telco_churn_artifacts.pkl"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telco Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main { background-color: #F8FAFC; }

    h1 { color: #1E3A5F; font-weight: 800; }
    h2 { color: #1C7293; font-weight: 700;
         border-bottom: 2px solid #1C7293; padding-bottom: 6px; }
    h3 { color: #065A82; font-weight: 600; }

    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #D1E8F5;
        border-left: 4px solid #1C7293;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }

    .pred-churn {
        background: #FFF0F0;
        border: 2px solid #E53935;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #B71C1C;
    }

    .pred-safe {
        background: #F0FFF4;
        border: 2px solid #388E3C;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B5E20;
    }

    .info-box {
        background: #EBF5FB;
        border-left: 4px solid #1C7293;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.92rem;
        color: #1E3A5F;
    }

    section[data-testid="stSidebar"] {
        background-color: #1E3A5F;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_and_clean_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# LOAD ARTIFACT (model, scaler, feature_cols, hasil evaluasi)
# ============================================================
# Semua proses training (split, SMOTE, scaling, training 6 model,
# tuning Optuna) sudah dilakukan SEKALI di notebook. Di sini kita
# hanya me-load hasilnya — TIDAK ada proses training di Streamlit.

@st.cache_resource(show_spinner=False)
def load_artifacts(path=ARTIFACT_PATH):
    artifacts = joblib.load(path)
    return (
        artifacts["models"],         # dict nama_model -> objek model terlatih (untuk feature importance, dsb)
        artifacts["results"],        # dict nama_model -> metrik + y_pred + y_prob (hasil evaluasi test set)
        artifacts["scaler"],         # StandardScaler yang sudah di-fit saat training
        artifacts["feature_cols"],   # urutan kolom fitur, wajib sama dengan saat training
        artifacts["y_test"],         # label test set (untuk confusion matrix / ROC di UI)
        artifacts["final_model"],    # model final (Logistic Regression hasil tuning Optuna)
        artifacts["final_metrics"],  # metrik + y_pred + y_prob model final
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_cm_fig(y_true, y_pred, title="Confusion Matrix"):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        linewidths=0.5, linecolor="#E0E0E0"
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def encode_input(row_dict, feature_cols):
    def yn(v):
        return 1 if v == "Yes" else 0

    d = {}
    d["gender"]           = 1 if row_dict["gender"] == "Male" else 0
    d["SeniorCitizen"]    = row_dict["SeniorCitizen"]
    d["Partner"]          = yn(row_dict["Partner"])
    d["Dependents"]       = yn(row_dict["Dependents"])
    d["tenure"]           = row_dict["tenure"]
    d["PhoneService"]     = yn(row_dict["PhoneService"])
    d["MultipleLines"]    = 1 if row_dict["MultipleLines"]    == "Yes" else 0
    d["OnlineSecurity"]   = 1 if row_dict["OnlineSecurity"]   == "Yes" else 0
    d["OnlineBackup"]     = 1 if row_dict["OnlineBackup"]     == "Yes" else 0
    d["DeviceProtection"] = 1 if row_dict["DeviceProtection"] == "Yes" else 0
    d["TechSupport"]      = 1 if row_dict["TechSupport"]      == "Yes" else 0
    d["StreamingTV"]      = 1 if row_dict["StreamingTV"]      == "Yes" else 0
    d["StreamingMovies"]  = 1 if row_dict["StreamingMovies"]  == "Yes" else 0
    d["PaperlessBilling"] = yn(row_dict["PaperlessBilling"])
    d["MonthlyCharges"]   = row_dict["MonthlyCharges"]
    d["TotalCharges"]     = row_dict["TotalCharges"]

    for cat in ["DSL", "Fiber optic", "No"]:
        d[f"InternetService_{cat}"] = 1 if row_dict["InternetService"] == cat else 0

    for cat in ["Month-to-month", "One year", "Two year"]:
        d[f"Contract_{cat}"] = 1 if row_dict["Contract"] == cat else 0

    for cat in ["Bank transfer (automatic)", "Credit card (automatic)",
                "Electronic check", "Mailed check"]:
        d[f"PaymentMethod_{cat}"] = 1 if row_dict["PaymentMethod"] == cat else 0

    inp = pd.DataFrame([d])
    for col in feature_cols:
        if col not in inp.columns:
            inp[col] = 0
    return inp[feature_cols]


# ============================================================
# LOAD DATA & ARTIFACT
# ============================================================
# df       -> hanya untuk keperluan EDA/tampilan (baca CSV, ringan)
# artifact -> model & hasil training, di-load dari file .pkl (TIDAK training)

df = load_and_clean_data()

with st.spinner("Memuat model..."):
    (
        models, results, scaler, feature_cols,
        y_test, final_model, final_metrics
    ) = load_artifacts()

churn_yes = (df["Churn"] == "Yes").sum()
churn_no  = (df["Churn"] == "No").sum()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## Telco Churn")
    st.markdown("---")
    menu = st.radio(
        "Navigasi",
        [
            "Beranda",
            "EDA dan Visualisasi",
            "Perbandingan Model",
            "Confusion Matrix",
            "Prediksi Pelanggan",
            "Tentang Aplikasi",
        ]
    )
    st.markdown("---")
    st.markdown("**Ringkasan Dataset**")
    st.markdown(f"Total baris  : {len(df):,}")
    st.markdown(f"Total fitur  : {df.shape[1] - 1}")
    st.markdown(f"Churn (Yes)  : {churn_yes:,}")
    st.markdown(f"No Churn     : {churn_no:,}")
    st.markdown("---")
    st.markdown("**Model Terbaik**")
    st.markdown("Regresi Logistik")
    st.markdown("(Optuna Tuned)")
    st.markdown(f"AUC-ROC  : {final_metrics['AUC-ROC']:.4f}")
    st.markdown(f"F1-score : {final_metrics['F1-score']:.4f}")
    st.markdown(f"Accuracy : {final_metrics['Accuracy']:.4f}")


# ============================================================
# PAGE: BERANDA
# ============================================================

if menu == "Beranda":
    st.title("Telco Customer Churn Prediction")
    st.markdown(
        "Aplikasi prediksi churn pelanggan perusahaan telekomunikasi menggunakan "
        "Machine Learning. Dataset yang digunakan adalah **IBM Telco Customer Churn** "
        "dengan 7.043 record pelanggan."
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pelanggan", f"{len(df):,}")
    col2.metric("Pelanggan Churn", f"{churn_yes:,}",
                f"{churn_yes / len(df) * 100:.1f}%")
    col3.metric("Tidak Churn",     f"{churn_no:,}",
                f"{churn_no / len(df) * 100:.1f}%")
    col4.metric("Jumlah Fitur",    f"{df.shape[1] - 1}")

    st.markdown("---")
    st.subheader("Metrik Final Model (Regresi Logistik - Optuna)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{final_metrics['Accuracy']:.4f}")
    c2.metric("Precision", f"{final_metrics['Precision']:.4f}")
    c3.metric("Recall",    f"{final_metrics['Recall']:.4f}")
    c4.metric("F1-score",  f"{final_metrics['F1-score']:.4f}")
    c5.metric("AUC-ROC",   f"{final_metrics['AUC-ROC']:.4f}")

    st.markdown("---")
    st.subheader("Sample Data (10 Baris Pertama)")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="info-box">'
        "<strong>Pipeline:</strong> "
        "Load Data - Cleaning - EDA - Encoding (Binary / Label / One-Hot) - "
        "Train-Test Split (80:20) - SMOTE Oversampling - StandardScaler - "
        "Pelatihan 6 Model - Evaluasi - Hyperparameter Tuning Optuna - Final Model"
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# PAGE: EDA DAN VISUALISASI
# ============================================================

elif menu == "EDA dan Visualisasi":
    st.title("EDA dan Visualisasi Data")
    st.markdown("Eksplorasi untuk memahami karakteristik dan pola churn pelanggan.")
    st.markdown("---")

    # ── 1. Distribusi Churn ─────────────────────────────────
    st.subheader("1. Distribusi Target (Churn)")
    churn_dist = df["Churn"].value_counts()
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 4))

    ax1[0].pie(
        churn_dist, labels=churn_dist.index,
        autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90
    )
    ax1[0].set_title("Distribusi Churn (Pie Chart)", fontweight="bold")

    bars1 = ax1[1].bar(
        churn_dist.index, churn_dist.values,
        color=["#4CAF50", "#F44336"], edgecolor="white", linewidth=1.2
    )
    ax1[1].set_title("Distribusi Churn (Bar Chart)", fontweight="bold")
    ax1[1].set_ylabel("Jumlah Pelanggan")
    for bar, v in zip(bars1, churn_dist.values):
        ax1[1].text(
            bar.get_x() + bar.get_width() / 2,
            v + 40, str(v), ha="center", fontweight="bold"
        )
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.markdown(
        '<div class="info-box">'
        "Dataset mengalami <strong>class imbalance</strong>: "
        "73.5% pelanggan tidak churn dan 26.5% churn. "
        "Kondisi ini ditangani menggunakan SMOTE pada tahap preprocessing."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── 2. Distribusi Variabel Numerik ──────────────────────
    st.subheader("2. Distribusi Variabel Numerik")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig2, ax2 = plt.subplots(2, 3, figsize=(15, 8))
    for i, col in enumerate(num_cols):
        ax2[0][i].hist(df[col], bins=30, color="#2196F3",
                       edgecolor="white", alpha=0.85)
        ax2[0][i].set_title(f"Distribusi {col}", fontweight="bold")
        ax2[0][i].set_ylabel("Frekuensi")

        ax2[1][i].hist(df[df["Churn"] == "No"][col], bins=25,
                       alpha=0.65, color="#4CAF50", label="No Churn")
        ax2[1][i].hist(df[df["Churn"] == "Yes"][col], bins=25,
                       alpha=0.65, color="#F44336", label="Churn")
        ax2[1][i].set_title(f"{col} by Churn", fontweight="bold")
        ax2[1][i].legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.markdown("---")

    # ── 3. Churn Rate Kategorikal ───────────────────────────
    st.subheader("3. Churn Rate Berdasarkan Variabel Kategorikal")
    cat_cols = [
        "Contract", "InternetService", "PaymentMethod",
        "OnlineSecurity", "TechSupport", "PaperlessBilling"
    ]
    fig3, ax3 = plt.subplots(2, 3, figsize=(18, 9))
    for i, col in enumerate(cat_cols):
        ax = ax3[i // 3][i % 3]
        cr = (
            df.groupby(col)["Churn"]
              .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
              .reset_index()
        )
        cr.columns = [col, "ChurnRate"]
        bars_h = ax.barh(cr[col], cr["ChurnRate"],
                         color="#FF5722", alpha=0.85, edgecolor="white")
        for bar_h, val in zip(bars_h, cr["ChurnRate"]):
            ax.text(
                val + 0.5,
                bar_h.get_y() + bar_h.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9
            )
        ax.set_title(f"Churn Rate by {col}", fontweight="bold")
        ax.set_xlabel("Churn Rate (%)")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.markdown("---")

    # ── 4. Correlation Heatmap ──────────────────────────────
    st.subheader("4. Correlation Heatmap")
    df_temp = df.copy()
    df_temp["Churn_bin"] = (df_temp["Churn"] == "Yes").astype(int)
    corr = df_temp[[
        "tenure", "MonthlyCharges", "TotalCharges",
        "SeniorCitizen", "Churn_bin"
    ]].corr()

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        ax=ax4, linewidths=0.5, center=0
    )
    ax4.set_title("Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.markdown(
        '<div class="info-box">'
        "<strong>Insight Korelasi:</strong> "
        "Tenure berkorelasi negatif dengan churn (pelanggan lama lebih loyal). "
        "MonthlyCharges berkorelasi positif dengan churn. "
        "TotalCharges berkorelasi kuat dengan tenure."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── 5. Senior Citizen dan Gender ────────────────────────
    st.subheader("5. Churn Rate Berdasarkan Senior Citizen dan Gender")
    fig5, ax5 = plt.subplots(1, 2, figsize=(12, 4))

    senior_cr = (
        df.groupby("SeniorCitizen")["Churn"]
          .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
          .reset_index()
    )
    senior_cr["SeniorCitizen"] = senior_cr["SeniorCitizen"].map(
        {0: "Non-Senior", 1: "Senior"}
    )
    bars_s = ax5[0].bar(
        senior_cr["SeniorCitizen"], senior_cr["Churn"],
        color=["#2196F3", "#F44336"], edgecolor="white"
    )
    ax5[0].set_title("Churn Rate by Senior Citizen", fontweight="bold")
    ax5[0].set_ylabel("Churn Rate (%)")
    for bar_s, v in zip(bars_s, senior_cr["Churn"]):
        ax5[0].text(
            bar_s.get_x() + bar_s.get_width() / 2,
            v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold"
        )

    gender_cr = (
        df.groupby("gender")["Churn"]
          .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
          .reset_index()
    )
    bars_g = ax5[1].bar(
        gender_cr["gender"], gender_cr["Churn"],
        color=["#9C27B0", "#FF9800"], edgecolor="white"
    )
    ax5[1].set_title("Churn Rate by Gender", fontweight="bold")
    ax5[1].set_ylabel("Churn Rate (%)")
    for bar_g, v in zip(bars_g, gender_cr["Churn"]):
        ax5[1].text(
            bar_g.get_x() + bar_g.get_width() / 2,
            v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()


# ============================================================
# PAGE: PERBANDINGAN MODEL
# ============================================================

elif menu == "Perbandingan Model":
    st.title("Perbandingan Model Klasifikasi")
    st.markdown(
        "Enam model klasifikasi dilatih dan dievaluasi menggunakan "
        "dataset yang sama dengan SMOTE dan StandardScaler."
    )
    st.markdown("---")

    # ── Tabel Metrik ────────────────────────────────────────
    st.subheader("Tabel Metrik Semua Model")
    rows = []
    for name, r in results.items():
        auc_v = r["AUC-ROC"]
        rows.append({
            "Model":      name,
            "Accuracy":   round(r["Accuracy"],   4),
            "Precision":  round(r["Precision"],  4),
            "Recall":     round(r["Recall"],     4),
            "F1-score":   round(r["F1-score"],   4),
            "AUC-ROC":    round(auc_v, 4) if auc_v is not None else "N/A",
            "CV Mean F1": round(r["CV Mean F1"], 4),
            "CV Std F1":  round(r["CV Std F1"],  4),
        })
    df_res = pd.DataFrame(rows)

    def highlight_best(col):
        if col.name in ["Accuracy", "Precision", "Recall",
                        "F1-score", "AUC-ROC", "CV Mean F1"]:
            try:
                max_v = pd.to_numeric(col, errors="coerce").max()
                return [
                    "background-color: #D4EDDA; font-weight: bold"
                    if pd.to_numeric(v, errors="coerce") == max_v else ""
                    for v in col
                ]
            except Exception:
                return [""] * len(col)
        return [""] * len(col)

    st.dataframe(
        df_res.style.apply(highlight_best),
        use_container_width=True
    )

    st.markdown(
        '<div class="info-box">'
        "Sel berwarna hijau menunjukkan nilai terbaik pada setiap metrik. "
        "Regresi Logistik dipilih sebagai model terbaik berdasarkan AUC-ROC tertinggi (0.8397)."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Pilih Metrik ────────────────────────────────────────
    st.subheader("Grafik Perbandingan Metrik")
    metric_sel = st.selectbox(
        "Pilih metrik:",
        ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC", "CV Mean F1"]
    )

    model_names = list(results.keys())
    metric_vals = [
        results[n][metric_sel]
        if results[n].get(metric_sel) is not None else 0
        for n in model_names
    ]
    palette_bar = [
        "#1C7293" if v == max(metric_vals) else "#90CAE6"
        for v in metric_vals
    ]

    fig_bar, ax_bar = plt.subplots(figsize=(11, 5))
    bars_b = ax_bar.bar(
        model_names, metric_vals,
        color=palette_bar, edgecolor="white", linewidth=1.2
    )
    ax_bar.axhline(0.8, color="red", linestyle="--", lw=1.5, label="Threshold 0.8")
    ax_bar.set_title(f"Perbandingan {metric_sel} Antar Model",
                     fontsize=13, fontweight="bold")
    ax_bar.set_ylabel(metric_sel)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend()
    for bar_b, v in zip(bars_b, metric_vals):
        ax_bar.text(
            bar_b.get_x() + bar_b.get_width() / 2,
            v + 0.01, f"{v:.4f}",
            ha="center", fontsize=9, fontweight="bold"
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close()
    st.markdown("---")

    # ── Semua Metrik ────────────────────────────────────────
    st.subheader("Semua Metrik dalam Satu Grafik")
    metrics_all = ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]
    fig_all, ax_all = plt.subplots(figsize=(14, 6))
    x_all = np.arange(len(model_names))
    w_all = 0.14
    clr_all = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
    for i, met in enumerate(metrics_all):
        vals = [
            results[n][met] if results[n].get(met) is not None else 0
            for n in model_names
        ]
        ax_all.bar(x_all + i * w_all, vals, w_all,
                   label=met, color=clr_all[i], alpha=0.85)
    ax_all.set_xticks(x_all + w_all * 2)
    ax_all.set_xticklabels(model_names, rotation=15, ha="right")
    ax_all.axhline(0.8, color="red", linestyle="--", lw=1.5, label="Threshold 0.8")
    ax_all.set_ylim(0.5, 1.05)
    ax_all.set_title("Perbandingan Model - Semua Metrik", fontsize=13, fontweight="bold")
    ax_all.set_ylabel("Score")
    ax_all.legend(loc="lower right", ncol=3)
    plt.tight_layout()
    st.pyplot(fig_all)
    plt.close()
    st.markdown("---")

    # ── Cross-Validation ────────────────────────────────────
    st.subheader("Hasil Cross-Validation (5-Fold F1)")
    cv_means = [results[n]["CV Mean F1"] for n in model_names]
    cv_stds  = [results[n]["CV Std F1"]  for n in model_names]
    pal_cv   = [
        "#1C7293" if v == max(cv_means) else "#90CAE6"
        for v in cv_means
    ]

    fig_cv, ax_cv = plt.subplots(figsize=(11, 4))
    ax_cv.bar(
        model_names, cv_means,
        color=pal_cv, edgecolor="white", linewidth=1.2,
        yerr=cv_stds, capsize=5, error_kw={"color": "#555"}
    )
    ax_cv.set_title("CV Mean F1-score (5-Fold StratifiedKFold)",
                    fontsize=13, fontweight="bold")
    ax_cv.set_ylabel("CV Mean F1")
    ax_cv.set_ylim(0.75, 0.90)
    for j, (v, s) in enumerate(zip(cv_means, cv_stds)):
        ax_cv.text(j, v + s + 0.003, f"{v:.4f}",
                   ha="center", fontsize=9, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig_cv)
    plt.close()
    st.markdown("---")

    # ── ROC Curves ──────────────────────────────────────────
    st.subheader("ROC Curve Semua Model")
    clr_roc = ["#2196F3", "#FF9800", "#4CAF50", "#E53935", "#9C27B0", "#00ACC1"]
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for (name, r), clr_r in zip(results.items(), clr_roc):
        if r["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
            ax_roc.plot(
                fpr, tpr, lw=2, color=clr_r,
                label=f"{name} (AUC = {r['AUC-ROC']:.4f})"
            )
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=11)
    ax_roc.set_title("ROC Curve Semua Model", fontsize=13, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_roc)
    plt.close()


# ============================================================
# PAGE: CONFUSION MATRIX
# ============================================================

elif menu == "Confusion Matrix":
    st.title("Confusion Matrix")
    st.markdown(
        "Confusion matrix menampilkan perbandingan antara label "
        "prediksi dan label aktual untuk setiap model."
    )
    st.markdown(
        "| | Predicted No Churn | Predicted Churn |\n"
        "|---|---|---|\n"
        "| **Actual No Churn** | True Negative (TN) | False Positive (FP) |\n"
        "| **Actual Churn**    | False Negative (FN) | True Positive (TP) |"
    )
    st.markdown("---")

    # ── Pilih Model ─────────────────────────────────────────
    model_options = list(results.keys()) + ["Final Model (Optuna)"]
    sel_model = st.selectbox("Pilih model:", model_options)

    if sel_model == "Final Model (Optuna)":
        y_pred_sel = final_metrics["y_pred"]
        title_sel  = "Final Model - Regresi Logistik (Optuna)"
    else:
        y_pred_sel = results[sel_model]["y_pred"]
        title_sel  = sel_model

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        fig_cm_sel = make_cm_fig(y_test, y_pred_sel, title_sel)
        st.pyplot(fig_cm_sel)
        plt.close()

    with col_cm2:
        cm_arr = confusion_matrix(y_test, y_pred_sel)
        tn, fp, fn, tp = cm_arr.ravel()
        st.markdown("**Detail Confusion Matrix:**")
        st.markdown(f"- True Negative  (TN) : **{tn:,}** - tidak churn, diprediksi benar")
        st.markdown(f"- False Positive (FP) : **{fp:,}** - tidak churn, diprediksi churn")
        st.markdown(f"- False Negative (FN) : **{fn:,}** - churn, diprediksi tidak churn")
        st.markdown(f"- True Positive  (TP) : **{tp:,}** - churn, diprediksi benar")
        st.markdown("---")
        st.metric("Accuracy",  f"{accuracy_score(y_test, y_pred_sel):.4f}")
        st.metric("Precision", f"{precision_score(y_test, y_pred_sel):.4f}")
        st.metric("Recall",    f"{recall_score(y_test, y_pred_sel):.4f}")
        st.metric("F1-score",  f"{f1_score(y_test, y_pred_sel):.4f}")

    st.markdown("---")

    # ── Semua CM ────────────────────────────────────────────
    st.subheader("Confusion Matrix Semua Model")
    model_names_all = list(results.keys())
    cols_grid = st.columns(3)
    for idx, name in enumerate(model_names_all):
        with cols_grid[idx % 3]:
            fig_g = make_cm_fig(y_test, results[name]["y_pred"], name)
            st.pyplot(fig_g)
            plt.close()

    st.markdown("---")

    # ── Final Model CM + ROC ────────────────────────────────
    st.subheader("Confusion Matrix dan ROC - Final Model (Optuna)")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        fig_cf = make_cm_fig(
            y_test, final_metrics["y_pred"],
            "Final Model - Regresi Logistik (Optuna)"
        )
        st.pyplot(fig_cf)
        plt.close()

    with col_f2:
        fpr_f, tpr_f, _ = roc_curve(y_test, final_metrics["y_prob"])
        fig_rf, ax_rf = plt.subplots(figsize=(5, 4))
        ax_rf.plot(
            fpr_f, tpr_f, color="#2196F3", lw=2,
            label=f"AUC = {final_metrics['AUC-ROC']:.4f}"
        )
        ax_rf.plot([0, 1], [0, 1], "k--", lw=1.2)
        ax_rf.set_xlabel("False Positive Rate")
        ax_rf.set_ylabel("True Positive Rate")
        ax_rf.set_title("AUC-ROC Curve - Final Model", fontweight="bold")
        ax_rf.legend(loc="lower right")
        ax_rf.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_rf)
        plt.close()

    st.markdown("---")

    # ── Feature Importance ──────────────────────────────────
    st.subheader("Feature Importance - Final Model (Logistic Regression Coefficient)")
    coef_fi = final_model.coef_[0]
    fi_df = pd.DataFrame({
        "Feature":     feature_cols,
        "Coefficient": coef_fi,
        "Abs":         np.abs(coef_fi)
    }).sort_values("Abs", ascending=True)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
    clr_fi = ["#F44336" if v < 0 else "#4CAF50" for v in fi_df["Coefficient"]]
    ax_fi.barh(
        fi_df["Feature"], fi_df["Coefficient"],
        color=clr_fi, edgecolor="white", linewidth=0.5
    )
    ax_fi.axvline(0, color="black", linewidth=0.8)
    ax_fi.set_xlabel("Coefficient Value", fontsize=11)
    ax_fi.set_title(
        "Feature Importance (Logistic Regression Coefficient)",
        fontsize=12, fontweight="bold"
    )
    ax_fi.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_fi)
    plt.close()

    st.markdown(
        '<div class="info-box">'
        "<strong>Interpretasi:</strong> "
        "Koefisien positif (hijau) berarti fitur tersebut meningkatkan probabilitas churn. "
        "Koefisien negatif (merah) berarti fitur tersebut menurunkan probabilitas churn."
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# PAGE: PREDIKSI PELANGGAN
# ============================================================

elif menu == "Prediksi Pelanggan":
    st.title("Prediksi Churn Pelanggan")
    st.markdown(
        "Masukkan data pelanggan untuk memprediksi apakah pelanggan "
        "tersebut akan melakukan churn atau tidak."
    )
    st.markdown("---")

    with st.form(key="pred_form"):
        st.subheader("Informasi Demografis")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col_d2:
            senior = st.selectbox(
                "Senior Citizen", [0, 1],
                format_func=lambda x: "Ya" if x == 1 else "Tidak"
            )
        with col_d3:
            partner = st.selectbox("Partner", ["Yes", "No"])
        with col_d4:
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.subheader("Informasi Layanan")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            tenure      = st.slider("Lama Berlangganan (bulan)", 0, 72, 12)
            phone_svc   = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
        with col_s2:
            internet   = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_sec = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_bak = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
        with col_s3:
            device_prot = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_sup   = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            stream_tv  = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )

        stream_mov = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

        st.subheader("Informasi Kontrak dan Tagihan")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
        with col_b2:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col_b3:
            payment_mth = st.selectbox(
                "Payment Method",
                [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ]
            )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            monthly_chg = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0, max_value=120.0, value=65.0, step=0.5
            )
        with col_c2:
            default_total = float(monthly_chg * tenure) if tenure > 0 else 65.0
            total_chg = st.number_input(
                "Total Charges ($)",
                min_value=18.0, max_value=9000.0,
                value=default_total, step=1.0
            )

        btn_predict = st.form_submit_button(
            "Prediksi Sekarang", type="primary", use_container_width=True
        )

    if btn_predict:
        row_dict = {
            "gender":           gender,
            "SeniorCitizen":    senior,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_svc,
            "MultipleLines":    multi_lines,
            "InternetService":  internet,
            "OnlineSecurity":   online_sec,
            "OnlineBackup":     online_bak,
            "DeviceProtection": device_prot,
            "TechSupport":      tech_sup,
            "StreamingTV":      stream_tv,
            "StreamingMovies":  stream_mov,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment_mth,
            "MonthlyCharges":   monthly_chg,
            "TotalCharges":     total_chg,
        }

        inp_df = encode_input(row_dict, feature_cols)
        inp_sc = scaler.transform(inp_df)
        prob   = final_model.predict_proba(inp_sc)[0][1]
        pred   = 1 if prob >= 0.5 else 0

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        col_r1, col_r2, col_r3 = st.columns([1.2, 1, 1])
        with col_r1:
            if pred == 1:
                st.markdown(
                    '<div class="pred-churn">CHURN<br>'
                    '<span style="font-size:0.9rem;font-weight:400;">'
                    "Pelanggan diprediksi akan CHURN"
                    "</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="pred-safe">AMAN<br>'
                    '<span style="font-size:0.9rem;font-weight:400;">'
                    "Pelanggan diprediksi TIDAK churn"
                    "</span></div>",
                    unsafe_allow_html=True
                )
        with col_r2:
            st.metric("Probabilitas Churn",      f"{prob * 100:.2f}%")
            st.progress(float(prob))
        with col_r3:
            st.metric("Probabilitas Tidak Churn", f"{(1 - prob) * 100:.2f}%")
            st.progress(float(1 - prob))

        st.markdown("---")

        # ── Probability Bar ────────────────────────────────
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_pb, ax_pb = plt.subplots(figsize=(6, 1.8))
            ax_pb.barh(["Tidak Churn", "Churn"],
                       [1 - prob, prob],
                       color=["#4CAF50", "#F44336"],
                       edgecolor="white", height=0.5)
            ax_pb.set_xlim(0, 1)
            ax_pb.set_title("Probabilitas Prediksi", fontweight="bold")
            ax_pb.set_xlabel("Probabilitas")
            ax_pb.text(1 - prob - 0.02, 0, f"{(1-prob)*100:.1f}%",
                       va="center", ha="right", color="white", fontweight="bold")
            ax_pb.text(prob - 0.02, 1, f"{prob*100:.1f}%",
                       va="center", ha="right", color="white", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_pb)
            plt.close()

        with col_g2:
            st.markdown("**Rekomendasi Tindakan:**")
            if pred == 1:
                if contract == "Month-to-month":
                    st.warning(
                        "Tawarkan upgrade ke kontrak One Year atau Two Year "
                        "dengan diskon khusus."
                    )
                if tenure < 12:
                    st.warning(
                        "Berikan program loyalitas untuk pelanggan baru "
                        "(tenure kurang dari 12 bulan)."
                    )
                if internet == "Fiber optic" and online_sec == "No":
                    st.warning(
                        "Tawarkan paket Online Security sebagai nilai tambah "
                        "layanan Fiber Optic."
                    )
                if payment_mth == "Electronic check":
                    st.warning(
                        "Dorong migrasi ke metode pembayaran otomatis "
                        "untuk mengurangi risiko churn."
                    )
                if tech_sup == "No":
                    st.warning(
                        "Tawarkan paket Tech Support untuk meningkatkan "
                        "kepuasan pelanggan."
                    )
                if not any([
                    contract == "Month-to-month",
                    tenure < 12,
                    payment_mth == "Electronic check"
                ]):
                    st.warning(
                        "Hubungi pelanggan untuk survei kepuasan dan "
                        "penawaran retensi khusus."
                    )
            else:
                st.success(
                    "Pelanggan dalam kondisi stabil. "
                    "Pertahankan dengan program reward loyalitas."
                )
                if tenure >= 24:
                    st.success(
                        "Pelanggan setia (tenure 24 bulan ke atas). "
                        "Pertimbangkan program referral."
                    )

    st.markdown("---")
    st.markdown(
        '<div class="info-box">'
        "<strong>Catatan:</strong> Prediksi menggunakan Regresi Logistik "
        "yang telah dioptimasi dengan Optuna (50 trials). "
        "Threshold klasifikasi: probabilitas lebih dari atau sama dengan 0.50 diklasifikasikan sebagai Churn."
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# PAGE: TENTANG APLIKASI
# ============================================================

elif menu == "Tentang Aplikasi":
    st.title("Tentang Aplikasi")
    st.markdown("---")

    st.subheader("Deskripsi Proyek")
    st.markdown(
        "Aplikasi ini merupakan implementasi Machine Learning untuk memprediksi "
        "customer churn pada perusahaan telekomunikasi, dibangun sebagai proyek akhir "
        "dengan menggunakan dataset IBM Telco Customer Churn."
    )

    st.markdown("---")
    st.subheader("Dataset")
    col_ds1, col_ds2 = st.columns(2)
    with col_ds1:
        st.markdown(
            "- **Sumber**       : IBM Sample Datasets\n"
            "- **Nama File**    : WA_Fn-UseC_-Telco-Customer-Churn.csv\n"
            "- **Jumlah Baris** : 7.043 record\n"
            "- **Jumlah Kolom** : 21 kolom\n"
            "- **Target**       : Churn (Yes / No)"
        )
    with col_ds2:
        st.markdown(
            "- **Fitur Numerik**  : tenure, MonthlyCharges, TotalCharges\n"
            "- **Fitur Kategori** : 17 kolom kategorikal\n"
            "- **Missing Values** : 11 pada TotalCharges, diimputasi dengan median\n"
            "- **Duplikat**       : 22 baris dihapus\n"
            "- **Class Imbalance**: 73.5% No Churn, 26.5% Churn"
        )

    st.markdown("---")
    st.subheader("Pipeline Machine Learning")
    pipeline = [
        ("1.  Load Data",              "Membaca dataset CSV"),
        ("2.  Data Cleaning",          "Konversi tipe data, imputasi missing values, hapus duplikat"),
        ("3.  EDA",                    "Analisis distribusi, korelasi, dan visualisasi pola churn"),
        ("4.  Feature Engineering",    "Binary encoding, label encoding, one-hot encoding"),
        ("5.  Train-Test Split",       "Rasio 80:20 dengan stratified sampling"),
        ("6.  SMOTE",                  "Oversampling kelas minoritas untuk mengatasi class imbalance"),
        ("7.  StandardScaler",         "Normalisasi fitur numerik"),
        ("8.  Pelatihan Model",        "k-NN, Naive Bayes, Decision Tree, Logistic Regression, Random Forest, XGBoost"),
        ("9.  Evaluasi",               "Accuracy, Precision, Recall, F1-score, AUC-ROC, CV 5-Fold"),
        ("10. Hyperparameter Tuning",  "Optuna Bayesian Optimization (50 trials) pada Logistic Regression"),
        ("11. Final Model",            "Logistic Regression dengan parameter terbaik dari Optuna"),
    ]
    for step, desc in pipeline:
        st.markdown(f"**{step}** : {desc}")

    st.markdown("---")
    st.subheader("Hasil Evaluasi Model")
    eval_rows = {
        "Model": [
            "k-NN", "Naive Bayes", "Decision Tree",
            "Regresi Logistik", "Random Forest", "XGBoost",
            "Final Model (Optuna)"
        ],
        "Accuracy":  [0.7502, 0.7587, 0.7167, 0.7886, 0.7801, 0.7786,
                      round(final_metrics["Accuracy"],  4)],
        "Precision": [0.5245, 0.5340, 0.4680, 0.6005, 0.5892, 0.5874,
                      round(final_metrics["Precision"], 4)],
        "Recall":    [0.6048, 0.6962, 0.5108, 0.6022, 0.5591, 0.5511,
                      round(final_metrics["Recall"],    4)],
        "F1-score":  [0.5618, 0.6044, 0.4884, 0.6013, 0.5738, 0.5687,
                      round(final_metrics["F1-score"],  4)],
        "AUC-ROC":   [0.7725, 0.8251, 0.6519, 0.8397, 0.8205, 0.8202,
                      round(final_metrics["AUC-ROC"],   4)],
    }
    st.dataframe(pd.DataFrame(eval_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Tech Stack")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            "**Machine Learning**\n"
            "- scikit-learn\n"
            "- xgboost\n"
            "- imbalanced-learn (SMOTE)\n"
            "- optuna (Bayesian Optimization)"
        )
    with col_t2:
        st.markdown(
            "**Visualisasi dan Web App**\n"
            "- Streamlit\n"
            "- Matplotlib\n"
            "- Seaborn\n"
            "- Pandas / NumPy"
        )

    st.markdown("---")
    st.subheader("Cara Menjalankan Aplikasi")
    st.code(
        "# 1. Install dependencies\n"
        "pip install streamlit scikit-learn xgboost imbalanced-learn optuna\n"
        "pip install matplotlib seaborn pandas numpy\n\n"
        "# 2. Pastikan file dataset ada di direktori yang sama:\n"
        "#    WA_Fn-UseC_-Telco-Customer-Churn.csv\n\n"
        "# 3. Jalankan aplikasi\n"
        "streamlit run streamlit_telco_churn.py",
        language="bash"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<hr style='border:1px solid #D1E8F5; margin-top:40px;'>"
    "<p style='text-align:center; color:#90A4AE; font-size:0.85rem;'>"
    "Telco Customer Churn Prediction | Final Project Machine Learning | 2024"
    "</p>",
    unsafe_allow_html=True
)
