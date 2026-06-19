import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)

st.set_page_config(
    page_title="Company Bankruptcy Prediction",
    page_icon="🏦",
    layout="wide"
)

# ── Train model sekali, cache hasilnya ───────────────────────────
@st.cache_resource(show_spinner="⏳ Melatih model, harap tunggu...")
def load_and_train():
    df = pd.read_csv("data.csv")
    X  = df.drop("Bankrupt?", axis=1)
    y  = df["Bankrupt?"]

    sel     = VarianceThreshold(threshold=0.01)
    X_sel   = sel.fit_transform(X)
    cols    = X.columns[sel.get_support()]
    X       = pd.DataFrame(X_sel, columns=cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler          = StandardScaler()
    X_train_sc      = scaler.fit_transform(X_train)
    X_test_sc       = scaler.transform(X_test)

    smote           = SMOTE(random_state=42)
    X_res, y_res    = smote.fit_resample(X_train_sc, y_train)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_res, y_res)

    return model, scaler, list(cols), X_test_sc, y_test

model, scaler, feature_names, X_test_sc, y_test = load_and_train()

THRESHOLD = 0.7254
TOP_FEATURES = [
    ' Current Ratio',
    ' Tax rate (A)',
    ' Cash Turnover Rate',
    ' Total Asset Growth Rate',
    ' Inventory Turnover Rate (times)',
    ' Cash/Total Assets',
    ' Current Assets/Total Assets',
    ' Research and development expense rate',
    ' Quick Asset Turnover Rate',
    ' Quick Assets/Total Assets',
]

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
st.sidebar.title("🏦 Bankruptcy Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Tentang Aplikasi**
Dashboard prediksi kebangkrutan perusahaan menggunakan
**Gradient Boosting Classifier**.

**Performa Model**
- ROC-AUC : `0.9044`
- Threshold : `0.7254`
- F1-Score (Bangkrut) : `0.42`
""")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "🔍 Prediksi Manual", "📂 Prediksi CSV", "📊 Analisis Fitur"]
)

# ════════════════════════════════════════════════════════════════
# HALAMAN 1 — BERANDA
# ════════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.title("🏦 Company Bankruptcy Prediction Dashboard")
    st.markdown("### Prediksi Kebangkrutan Perusahaan dengan Machine Learning")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 ROC-AUC",    "0.9044", "Excellent")
    c2.metric("📈 Recall",      "57%",    "Kelas Bangkrut")
    c3.metric("🎚️ Threshold", "0.7254",  "Optimal")
    c4.metric("📦 Total Data",  "6,819",  "perusahaan")

    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        st.subheader("📌 Tentang Dataset")
        st.markdown("""
        - **Sumber**: Taiwan Economic Journal (TEJ)
        - **Periode**: 1999–2009
        - **Jumlah Fitur Awal**: 95 fitur keuangan
        - **Fitur setelah seleksi**: 31 fitur
        - **Target**: `Bankrupt?` (0 = Tidak, 1 = Ya)
        - **Class imbalance**: 96.77% sehat vs 3.23% bangkrut
        - **Solusi**: SMOTE oversampling
        """)
    with cr:
        st.subheader("🧠 Alur Pemodelan")
        st.markdown("""
        1. 📥 **Load & EDA** — eksplorasi distribusi & korelasi
        2. 🧹 **Preprocessing** — variance threshold → 31 fitur
        3. ⚖️ **SMOTE** — balancing data training
        4. 🤖 **3 Model** — LR, RF, Gradient Boosting
        5. 🏆 **Best Model** — Gradient Boosting (AUC 0.90)
        6. 🎚️ **Threshold Tuning** — 0.50 → 0.7254
        7. 🌐 **Deploy** — Streamlit Cloud
        """)

    st.markdown("---")
    st.subheader("📊 Distribusi Kelas & Perbandingan Model")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(['Tidak Bangkrut', 'Bangkrut'], [6599, 220],
                color=['#2196F3', '#F44336'], edgecolor='black')
    axes[0].set_title('Distribusi Kelas (Imbalanced)', fontweight='bold')
    axes[0].set_ylabel('Jumlah')
    for i, v in enumerate([6599, 220]):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

    mn  = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
    auc = [0.8368, 0.8817, 0.9044]
    bars = axes[1].bar(mn, auc, color=['#FF9800','#4CAF50','#9C27B0'], edgecolor='black')
    axes[1].set_title('Perbandingan ROC-AUC', fontweight='bold')
    axes[1].set_ylabel('ROC-AUC Score')
    axes[1].set_ylim(0.80, 0.93)
    for b, v in zip(bars, auc):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                     f'{v:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    # ROC Curve live
    st.markdown("---")
    st.subheader("📈 ROC Curve (Model Terbaik)")
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(fpr, tpr, color='#9C27B0', lw=2, label=f'Gradient Boosting (AUC={auc_score:.4f})')
    ax2.plot([0,1],[0,1],'k--', lw=1, label='Random Classifier')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve — Gradient Boosting', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# ════════════════════════════════════════════════════════════════
# HALAMAN 2 — PREDIKSI MANUAL
# ════════════════════════════════════════════════════════════════
elif page == "🔍 Prediksi Manual":
    st.title("🔍 Prediksi Manual")
    st.markdown("Masukkan nilai indikator keuangan perusahaan.")
    st.info("💡 Isi 10 fitur terpenting. Fitur lain diisi nilai 0 secara otomatis.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    user_input = {}
    with col1:
        user_input[' Current Ratio']                         = st.number_input("Current Ratio",               value=1.0,  step=0.01)
        user_input[' Tax rate (A)']                          = st.number_input("Tax Rate (A)",                 value=0.2,  min_value=0.0, max_value=1.0, step=0.01)
        user_input[' Cash Turnover Rate']                    = st.number_input("Cash Turnover Rate",           value=0.5,  step=0.01)
        user_input[' Total Asset Growth Rate']               = st.number_input("Total Asset Growth Rate",      value=0.1,  step=0.01)
        user_input[' Inventory Turnover Rate (times)']       = st.number_input("Inventory Turnover Rate",      value=0.5,  step=0.01)
    with col2:
        user_input[' Cash/Total Assets']                     = st.number_input("Cash / Total Assets",          value=0.1,  min_value=0.0, max_value=1.0, step=0.01)
        user_input[' Current Assets/Total Assets']           = st.number_input("Current Assets/Total Assets",  value=0.5,  min_value=0.0, max_value=1.0, step=0.01)
        user_input[' Research and development expense rate'] = st.number_input("R&D Expense Rate",             value=0.0,  step=0.001)
        user_input[' Quick Asset Turnover Rate']             = st.number_input("Quick Asset Turnover Rate",    value=0.5,  step=0.01)
        user_input[' Quick Assets/Total Assets']             = st.number_input("Quick Assets/Total Assets",    value=0.4,  min_value=0.0, max_value=1.0, step=0.01)

    st.markdown("---")
    if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
        row = pd.DataFrame(columns=feature_names)
        row.loc[0] = 0
        for col, val in user_input.items():
            if col in feature_names:
                row.loc[0, col] = val

        row_sc = scaler.transform(row)
        prob   = model.predict_proba(row_sc)[0][1]
        pred   = int(prob >= THRESHOLD)

        st.markdown("---")
        cr, cg = st.columns(2)
        with cr:
            if pred == 1:
                st.error(f"## ⚠️ BERPOTENSI BANGKRUT\nProbabilitas: **{prob*100:.1f}%**")
                st.markdown("**Rekomendasi:** Audit keuangan segera, evaluasi rasio likuiditas, restrukturisasi utang.")
            else:
                st.success(f"## ✅ KONDISI KEUANGAN SEHAT\nProbabilitas Bangkrut: **{prob*100:.1f}%**")
                st.markdown("**Rekomendasi:** Pertahankan strategi saat ini, monitor pertumbuhan aset.")

        with cg:
            fig_g, ax_g = plt.subplots(figsize=(5, 3))
            color = '#F44336' if pred == 1 else '#4CAF50'
            ax_g.barh(['Risiko'], [prob],       color=color,    height=0.4)
            ax_g.barh(['Risiko'], [1 - prob],   left=[prob],
                      color='#E0E0E0', height=0.4)
            ax_g.axvline(THRESHOLD, color='orange', linestyle='--', lw=2,
                         label=f'Threshold ({THRESHOLD})')
            ax_g.set_xlim(0, 1)
            ax_g.set_title(f'Skor Risiko: {prob*100:.1f}%', fontweight='bold')
            ax_g.legend(fontsize=8)
            st.pyplot(fig_g)

# ════════════════════════════════════════════════════════════════
# HALAMAN 3 — PREDIKSI CSV
# ════════════════════════════════════════════════════════════════
elif page == "📂 Prediksi CSV":
    st.title("📂 Prediksi Batch via CSV")
    st.markdown("Upload file CSV berisi data keuangan banyak perusahaan.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"**Data:** {df_up.shape[0]} baris × {df_up.shape[1]} kolom")
        st.dataframe(df_up.head())

        for f in feature_names:
            if f not in df_up.columns:
                df_up[f] = 0

        X_up     = df_up[feature_names]
        X_up_sc  = scaler.transform(X_up)
        probs    = model.predict_proba(X_up_sc)[:, 1]
        preds    = (probs >= THRESHOLD).astype(int)

        df_up['Probabilitas_Bangkrut_%'] = (probs * 100).round(2)
        df_up['Prediksi']                = preds
        df_up['Status']                  = df_up['Prediksi'].map({0:'✅ Sehat', 1:'⚠️ Bangkrut'})

        st.markdown("---")
        ca, cb, cc = st.columns(3)
        ca.metric("Total Perusahaan",      len(preds))
        cb.metric("⚠️ Prediksi Bangkrut", int(preds.sum()))
        cc.metric("✅ Prediksi Sehat",     int((preds==0).sum()))

        st.dataframe(df_up[['Probabilitas_Bangkrut_%', 'Status']])

        csv_out = df_up[['Probabilitas_Bangkrut_%','Prediksi','Status']].to_csv(index=False)
        st.download_button("⬇️ Download Hasil", csv_out, "hasil_prediksi.csv", "text/csv")

# ════════════════════════════════════════════════════════════════
# HALAMAN 4 — ANALISIS FITUR
# ════════════════════════════════════════════════════════════════
elif page == "📊 Analisis Fitur":
    st.title("📊 Analisis Fitur Penting")
    st.markdown("---")

    feat_imp = {
        'Current Ratio':               0.357019,
        'Tax rate (A)':                0.185623,
        'Cash Turnover Rate':          0.111877,
        'Total Asset Growth Rate':     0.092291,
        'Inventory Turnover Rate':     0.058291,
        'Cash/Total Assets':           0.026022,
        'Current Assets/Total Assets': 0.023332,
        'R&D Expense Rate':            0.022896,
        'Quick Asset Turnover Rate':   0.017965,
        'Quick Assets/Total Assets':   0.016183,
    }
    fi = pd.Series(feat_imp).sort_values()

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(fi)))
    bars   = ax_fi.barh(fi.index, fi.values, color=colors, edgecolor='grey')
    for b, v in zip(bars, fi.values):
        ax_fi.text(b.get_width() + 0.003, b.get_y() + b.get_height()/2,
                   f'{v:.4f}', va='center', fontsize=9)
    ax_fi.set_xlabel('Importance Score')
    ax_fi.set_title('Top 10 Feature Importance — Gradient Boosting', fontweight='bold')
    ax_fi.set_xlim(0, 0.42)
    plt.tight_layout()
    st.pyplot(fig_fi)

    st.markdown("---")
    st.subheader("💡 Interpretasi Fitur Utama")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **🥇 Current Ratio (0.357)**
        Fitur paling dominan. Nilai < 1 → perusahaan tidak mampu bayar kewajiban jangka pendek → sinyal kuat bangkrut.

        **🥈 Tax Rate (0.186)**
        Pajak rendah/nol sering berarti tidak ada laba → indikator kesehatan finansial buruk.

        **🥉 Cash Turnover Rate (0.112)**
        Perputaran kas sangat rendah → likuiditas bermasalah.
        """)
    with c2:
        st.markdown("""
        **4️⃣ Total Asset Growth Rate (0.092)**
        Pertumbuhan aset negatif → bisnis menyusut.

        **5️⃣ Inventory Turnover Rate (0.058)**
        Perputaran lambat → produk tidak terjual → arus kas tersumbat.

        **6️⃣–10️⃣ Rasio Likuiditas Lainnya**
        Cash, Current Assets, Quick Assets — semua mengukur kemampuan penuhi kewajiban jangka pendek.
        """)
