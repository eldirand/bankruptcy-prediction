import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# ── CSS CUSTOM ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Company Bankruptcy Prediction",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
/* Font & background */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { 
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    display: block;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.18); }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(100,180,255,0.25);
    border-radius: 16px;
    padding: 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label { color: #90caf9 !important; font-size: 0.8rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #4ade80 !important; }

/* Info/success/error boxes */
[data-testid="stAlert"] { border-radius: 12px !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.6) !important;
}

/* Card wrapper custom */
.card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(100,180,255,0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.card h3 { color: #90caf9; margin-top: 0; font-size: 1.1rem; }
.card p, .card li { color: #cbd5e1; font-size: 0.9rem; line-height: 1.7; }

/* Section title */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #90caf9;
    border-left: 4px solid #2563eb;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* Tag badge */
.badge {
    display: inline-block;
    background: rgba(37,99,235,0.2);
    border: 1px solid rgba(37,99,235,0.5);
    color: #93c5fd;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px;
    font-weight: 600;
}

/* Step timeline */
.step {
    display: flex;
    align-items: flex-start;
    margin: 10px 0;
    gap: 12px;
}
.step-num {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
    flex-shrink: 0;
}
.step-text { color: #cbd5e1; font-size: 0.9rem; padding-top: 4px; }

/* Input number styling */
[data-testid="stNumberInput"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
}

/* Table styling */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #065f46, #059669) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Page background */
.main { background-color: #0a0f1e; }
</style>
""", unsafe_allow_html=True)


# ── TRAIN MODEL (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Melatih model, harap tunggu...")
def load_and_train():
    df = pd.read_csv("CompanyBankruptcy.csv")
    X  = df.drop("Bankrupt?", axis=1)
    y  = df["Bankrupt?"]

    sel   = VarianceThreshold(threshold=0.01)
    X_sel = sel.fit_transform(X)
    cols  = list(X.columns[sel.get_support()])
    X     = pd.DataFrame(X_sel, columns=cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train)
    X_test_sc    = scaler.transform(X_test)

    smote        = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_sc, y_train)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_res, y_res)

    return model, scaler, cols, X_test_sc, y_test

model, scaler, feature_names, X_test_sc, y_test = load_and_train()

THRESHOLD = 0.7254

# ── SIDEBAR ──────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding: 10px 0 20px 0;'>
    <div style='font-size:3rem;'>🏦</div>
    <div style='font-size:1.3rem; font-weight:700; color:#90caf9;'>Bankruptcy Predictor</div>
    <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>ML Dashboard v1.0</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background:rgba(255,255,255,0.06); border-radius:12px; padding:14px; margin-bottom:12px;'>
    <div style='font-size:0.8rem; color:#90caf9; font-weight:600; margin-bottom:8px;'>⚡ PERFORMA MODEL</div>
    <div style='display:flex; justify-content:space-between; margin:6px 0;'>
        <span style='color:#94a3b8; font-size:0.8rem;'>ROC-AUC</span>
        <span style='color:#4ade80; font-weight:700; font-size:0.85rem;'>0.9044</span>
    </div>
    <div style='display:flex; justify-content:space-between; margin:6px 0;'>
        <span style='color:#94a3b8; font-size:0.8rem;'>Threshold</span>
        <span style='color:#fbbf24; font-weight:700; font-size:0.85rem;'>0.7254</span>
    </div>
    <div style='display:flex; justify-content:space-between; margin:6px 0;'>
        <span style='color:#94a3b8; font-size:0.8rem;'>F1 (Bangkrut)</span>
        <span style='color:#f87171; font-weight:700; font-size:0.85rem;'>0.42</span>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "🏠 Beranda",
    "🔍 Prediksi Manual",
    "📂 Prediksi CSV",
    "📊 Analisis Fitur"
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.72rem; color:#475569; text-align:center; padding-top:8px;'>
    Dataset: Taiwan Economic Journal<br>Periode 1999–2009 | 6.819 perusahaan
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HALAMAN 1 — BERANDA
# ════════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown("""
    <div style='padding: 32px 0 8px 0;'>
        <h1 style='color:#ffffff; font-size:2.2rem; font-weight:800; margin:0;'>
            🏦 Company Bankruptcy Prediction
        </h1>
        <p style='color:#64748b; font-size:1rem; margin-top:8px;'>
            Sistem prediksi kebangkrutan perusahaan berbasis Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 ROC-AUC",      "0.9044", "Excellent")
    c2.metric("📈 Recall",        "57%",    "Bangkrut terdeteksi")
    c3.metric("🎚️ Threshold",   "0.7254",  "Optimal")
    c4.metric("📦 Total Data",    "6,819",  "perusahaan")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div class='card'>
            <h3>📌 Tentang Dataset</h3>
            <table style='width:100%; border-collapse:collapse;'>
                <tr><td style='color:#64748b; padding:5px 0; font-size:0.85rem; width:45%;'>Sumber</td>
                    <td style='color:#e2e8f0; font-size:0.85rem;'>Taiwan Economic Journal</td></tr>
                <tr><td style='color:#64748b; padding:5px 0; font-size:0.85rem;'>Periode</td>
                    <td style='color:#e2e8f0; font-size:0.85rem;'>1999 – 2009</td></tr>
                <tr><td style='color:#64748b; padding:5px 0; font-size:0.85rem;'>Total Data</td>
                    <td style='color:#e2e8f0; font-size:0.85rem;'>6.819 perusahaan</td></tr>
                <tr><td style='color:#64748b; padding:5px 0; font-size:0.85rem;'>Fitur</td>
                    <td style='color:#e2e8f0; font-size:0.85rem;'>95 indikator keuangan</td></tr>
                <tr><td style='color:#64748b; padding:5px 0; font-size:0.85rem;'>Target</td>
                    <td><span class='badge'>Bankrupt? (0/1)</span></td></tr>
                <tr><td style='color:#64748b; padding:5px 0; font-size:0.85rem;'>Imbalance</td>
                    <td><span class='badge'>96.77% vs 3.23%</span></td></tr>
                <tr><td style='color:#64748b; padding:5px 0; font-size:0.85rem;'>Solusi</td>
                    <td><span class='badge'>SMOTE</span></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class='card'>
            <h3>🧠 Alur Pemodelan</h3>
            <div class='step'><div class='step-num'>1</div><div class='step-text'><b style='color:#e2e8f0;'>Load & EDA</b> — Distribusi, korelasi, visualisasi data</div></div>
            <div class='step'><div class='step-num'>2</div><div class='step-text'><b style='color:#e2e8f0;'>Preprocessing</b> — Variance Threshold → 31 fitur</div></div>
            <div class='step'><div class='step-num'>3</div><div class='step-text'><b style='color:#e2e8f0;'>SMOTE</b> — Balancing data training 50:50</div></div>
            <div class='step'><div class='step-num'>4</div><div class='step-text'><b style='color:#e2e8f0;'>Modeling</b> — 3 algoritma dibandingkan</div></div>
            <div class='step'><div class='step-num'>5</div><div class='step-text'><b style='color:#e2e8f0;'>Best Model</b> — Gradient Boosting (AUC 0.90)</div></div>
            <div class='step'><div class='step-num'>6</div><div class='step-text'><b style='color:#e2e8f0;'>Threshold Tuning</b> — 0.50 → 0.7254</div></div>
            <div class='step'><div class='step-num'>7</div><div class='step-text'><b style='color:#e2e8f0;'>Deploy</b> — Streamlit Cloud Dashboard</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📊 Distribusi Data & Perbandingan Model</div>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor='#0f172a')
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values(): spine.set_edgecolor('#334155')

    axes[0].bar(['Tidak Bangkrut', 'Bangkrut'], [6599, 220],
                color=['#3b82f6', '#ef4444'], edgecolor='none', width=0.5)
    axes[0].set_title('Distribusi Kelas (Imbalanced)', color='#e2e8f0', fontweight='bold', pad=12)
    axes[0].set_ylabel('Jumlah', color='#94a3b8')
    for i, v in enumerate([6599, 220]):
        axes[0].text(i, v + 80, str(v), ha='center', fontweight='bold', color='#e2e8f0')

    mn  = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
    auc = [0.8368, 0.8817, 0.9044]
    clr = ['#f59e0b', '#10b981', '#8b5cf6']
    bars = axes[1].bar(mn, auc, color=clr, edgecolor='none', width=0.5)
    axes[1].set_title('Perbandingan ROC-AUC Model', color='#e2e8f0', fontweight='bold', pad=12)
    axes[1].set_ylabel('ROC-AUC Score', color='#94a3b8')
    axes[1].set_ylim(0.80, 0.93)
    for b, v in zip(bars, auc):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                     f'{v:.4f}', ha='center', fontweight='bold', color='#e2e8f0', fontsize=10)

    plt.tight_layout(pad=2)
    st.pyplot(fig)

    # ROC Curve live
    st.markdown("<div class='section-title'>📈 ROC Curve — Model Terbaik</div>", unsafe_allow_html=True)
    y_prob      = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)

    fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor='#0f172a')
    ax2.set_facecolor('#1e293b')
    ax2.tick_params(colors='#94a3b8')
    for spine in ax2.spines.values(): spine.set_edgecolor('#334155')
    ax2.plot(fpr, tpr, color='#8b5cf6', lw=2.5, label=f'Gradient Boosting (AUC = {auc_score:.4f})')
    ax2.fill_between(fpr, tpr, alpha=0.1, color='#8b5cf6')
    ax2.plot([0,1],[0,1],'--', color='#475569', lw=1.5, label='Random Classifier')
    ax2.set_xlabel('False Positive Rate', color='#94a3b8')
    ax2.set_ylabel('True Positive Rate', color='#94a3b8')
    ax2.set_title('ROC Curve — Gradient Boosting', color='#e2e8f0', fontweight='bold')
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    ax2.grid(alpha=0.15, color='#334155')
    plt.tight_layout()
    st.pyplot(fig2)


# ════════════════════════════════════════════════════════════════
# HALAMAN 2 — PREDIKSI MANUAL
# ════════════════════════════════════════════════════════════════
elif page == "🔍 Prediksi Manual":
    st.markdown("""
    <div style='padding: 24px 0 8px 0;'>
        <h1 style='color:#ffffff; font-size:2rem; font-weight:800; margin:0;'>🔍 Prediksi Manual</h1>
        <p style='color:#64748b; margin-top:6px;'>Masukkan indikator keuangan perusahaan untuk mendapatkan prediksi.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(37,99,235,0.12); border:1px solid rgba(37,99,235,0.35);
                border-radius:12px; padding:14px 18px; margin-bottom:20px; color:#93c5fd; font-size:0.9rem;'>
        💡 Isi 10 fitur keuangan terpenting di bawah ini. Fitur lain diisi nilai 0 secara otomatis.
    </div>
    """, unsafe_allow_html=True)

    # Input dalam card
    st.markdown("<div class='card'><h3>📥 Input Fitur Keuangan</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    inputs = {}
    with col1:
        inputs[' Current Ratio']                         = st.number_input("Current Ratio",                    value=1.0,  min_value=0.0, step=0.01, help="Aset Lancar / Kewajiban Lancar. Normal > 1")
        inputs[' Tax rate (A)']                          = st.number_input("Tax Rate (A)",                     value=0.2,  min_value=0.0, max_value=1.0, step=0.01)
        inputs[' Cash Turnover Rate']                    = st.number_input("Cash Turnover Rate",               value=0.5,  min_value=0.0, step=0.01)
        inputs[' Total Asset Growth Rate']               = st.number_input("Total Asset Growth Rate",          value=0.1,  step=0.01)
        inputs[' Inventory Turnover Rate (times)']       = st.number_input("Inventory Turnover Rate",          value=0.5,  min_value=0.0, step=0.01)

    with col2:
        inputs[' Cash/Total Assets']                     = st.number_input("Cash / Total Assets",              value=0.1,  min_value=0.0, max_value=1.0, step=0.01)
        inputs[' Current Assets/Total Assets']           = st.number_input("Current Assets / Total Assets",    value=0.5,  min_value=0.0, max_value=1.0, step=0.01)
        inputs[' Research and development expense rate'] = st.number_input("R&D Expense Rate",                 value=0.0,  min_value=0.0, step=0.001, format="%.3f")
        inputs[' Quick Asset Turnover Rate']             = st.number_input("Quick Asset Turnover Rate",        value=0.5,  min_value=0.0, step=0.01)
        inputs[' Quick Assets/Total Assets']             = st.number_input("Quick Assets / Total Assets",      value=0.4,  min_value=0.0, max_value=1.0, step=0.01)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Prediksi Sekarang", use_container_width=True):
        # FIX: buat row dengan dtype float64 dari awal
        row_data = {col: 0.0 for col in feature_names}
        for col, val in inputs.items():
            if col in row_data:
                row_data[col] = float(val)

        row      = pd.DataFrame([row_data], columns=feature_names).astype(float)
        row_sc   = scaler.transform(row)
        prob     = float(model.predict_proba(row_sc)[0][1])
        pred     = int(prob >= THRESHOLD)

        st.markdown("<br>", unsafe_allow_html=True)
        cr, cg = st.columns([1.1, 1], gap="large")

        with cr:
            if pred == 1:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#450a0a,#7f1d1d);
                            border:1px solid #ef4444; border-radius:16px; padding:24px;'>
                    <div style='font-size:2rem;'>⚠️</div>
                    <div style='font-size:1.4rem; font-weight:800; color:#fca5a5; margin:8px 0;'>BERPOTENSI BANGKRUT</div>
                    <div style='font-size:2.5rem; font-weight:900; color:#ef4444;'>{prob*100:.1f}%</div>
                    <div style='color:#fca5a5; font-size:0.85rem; margin-top:4px;'>Probabilitas Kebangkrutan</div>
                    <hr style='border-color:#7f1d1d; margin:16px 0;'>
                    <div style='color:#fca5a5; font-size:0.85rem;'><b>Rekomendasi:</b></div>
                    <ul style='color:#fca5a5; font-size:0.85rem; margin-top:8px;'>
                        <li>Segera audit laporan keuangan</li>
                        <li>Evaluasi rasio likuiditas perusahaan</li>
                        <li>Pertimbangkan restrukturisasi utang</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#052e16,#14532d);
                            border:1px solid #22c55e; border-radius:16px; padding:24px;'>
                    <div style='font-size:2rem;'>✅</div>
                    <div style='font-size:1.4rem; font-weight:800; color:#86efac; margin:8px 0;'>KONDISI KEUANGAN SEHAT</div>
                    <div style='font-size:2.5rem; font-weight:900; color:#22c55e;'>{prob*100:.1f}%</div>
                    <div style='color:#86efac; font-size:0.85rem; margin-top:4px;'>Probabilitas Kebangkrutan</div>
                    <hr style='border-color:#14532d; margin:16px 0;'>
                    <div style='color:#86efac; font-size:0.85rem;'><b>Rekomendasi:</b></div>
                    <ul style='color:#86efac; font-size:0.85rem; margin-top:8px;'>
                        <li>Pertahankan rasio likuiditas saat ini</li>
                        <li>Monitor pertumbuhan aset secara berkala</li>
                        <li>Lanjutkan strategi keuangan yang ada</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        with cg:
            fig_g, ax_g = plt.subplots(figsize=(5, 3.5), facecolor='#0f172a')
            ax_g.set_facecolor('#1e293b')
            for spine in ax_g.spines.values(): spine.set_edgecolor('#334155')
            ax_g.tick_params(colors='#94a3b8')

            color = '#ef4444' if pred == 1 else '#22c55e'
            ax_g.barh([''], [prob],       color=color,    height=0.4, label='Risiko')
            ax_g.barh([''], [1 - prob],   left=[prob],    color='#1e293b', height=0.4, edgecolor='#334155')
            ax_g.axvline(THRESHOLD, color='#fbbf24', linestyle='--', lw=2,
                         label=f'Threshold ({THRESHOLD})')
            ax_g.set_xlim(0, 1)
            ax_g.set_xlabel('Probabilitas', color='#94a3b8')
            ax_g.set_title(f'Skor Risiko: {prob*100:.1f}%', color='#e2e8f0', fontweight='bold')
            ax_g.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)
            ax_g.text(prob/2, 0, f'{prob*100:.1f}%', ha='center', va='center',
                      color='white', fontweight='bold', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_g)


# ════════════════════════════════════════════════════════════════
# HALAMAN 3 — PREDIKSI CSV
# ════════════════════════════════════════════════════════════════
elif page == "📂 Prediksi CSV":
    st.markdown("""
    <div style='padding: 24px 0 8px 0;'>
        <h1 style='color:#ffffff; font-size:2rem; font-weight:800; margin:0;'>📂 Prediksi Batch via CSV</h1>
        <p style='color:#64748b; margin-top:6px;'>Upload file CSV berisi data keuangan banyak perusahaan sekaligus.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(234,179,8,0.1); border:1px solid rgba(234,179,8,0.35);
                border-radius:12px; padding:14px 18px; margin-bottom:20px; color:#fde68a; font-size:0.85rem;'>
        📋 <b>Format CSV:</b> Pastikan kolom sesuai nama fitur dataset asli (header baris pertama).
        File bisa memiliki kolom <code>Bankrupt?</code> atau tidak — keduanya didukung.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"""
        <div style='background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                    border-radius:10px; padding:12px 16px; color:#6ee7b7; font-size:0.88rem; margin-bottom:12px;'>
            ✅ File berhasil dibaca — <b>{df_up.shape[0]} baris</b> × <b>{df_up.shape[1]} kolom</b>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("👁️ Preview data (5 baris pertama)"):
            st.dataframe(df_up.head(), use_container_width=True)

        # Prediksi
        for f in feature_names:
            if f not in df_up.columns:
                df_up[f] = 0.0

        X_up    = df_up[feature_names].astype(float)
        X_up_sc = scaler.transform(X_up)
        probs   = model.predict_proba(X_up_sc)[:, 1]
        preds   = (probs >= THRESHOLD).astype(int)

        n_bangkrut = int(preds.sum())
        n_sehat    = int((preds == 0).sum())
        pct        = n_bangkrut / len(preds) * 100

        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb, cc, cd = st.columns(4)
        ca.metric("📊 Total",           len(preds))
        cb.metric("⚠️ Bangkrut",        n_bangkrut, f"{pct:.1f}%")
        cc.metric("✅ Sehat",            n_sehat)
        cd.metric("🎚️ Threshold",       THRESHOLD)

        st.markdown("<div class='section-title'>📋 Hasil Prediksi</div>", unsafe_allow_html=True)

        df_result = pd.DataFrame({
            'No': range(1, len(preds)+1),
            'Probabilitas Bangkrut (%)': (probs * 100).round(2),
            'Prediksi': preds,
            'Status': ['⚠️ Bangkrut' if p == 1 else '✅ Sehat' for p in preds]
        })

        st.dataframe(df_result, use_container_width=True, height=400)

        csv_out = df_result.to_csv(index=False)
        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            csv_out, "hasil_prediksi.csv", "text/csv",
            use_container_width=True
        )

        # Mini visualisasi hasil
        st.markdown("<div class='section-title'>📊 Ringkasan Hasil</div>", unsafe_allow_html=True)
        fig_r, axes_r = plt.subplots(1, 2, figsize=(10, 3.5), facecolor='#0f172a')
        for ax in axes_r:
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values(): spine.set_edgecolor('#334155')

        axes_r[0].pie([n_sehat, n_bangkrut], labels=['Sehat', 'Bangkrut'],
                      colors=['#22c55e','#ef4444'], autopct='%1.1f%%',
                      textprops={'color':'#e2e8f0'}, startangle=90)
        axes_r[0].set_title('Proporsi Prediksi', color='#e2e8f0', fontweight='bold')

        axes_r[1].hist(probs * 100, bins=20, color='#8b5cf6', edgecolor='none', alpha=0.8)
        axes_r[1].axvline(THRESHOLD*100, color='#fbbf24', linestyle='--', lw=2,
                          label=f'Threshold ({THRESHOLD*100:.1f}%)')
        axes_r[1].set_xlabel('Probabilitas Bangkrut (%)', color='#94a3b8')
        axes_r[1].set_ylabel('Jumlah Perusahaan', color='#94a3b8')
        axes_r[1].set_title('Distribusi Probabilitas', color='#e2e8f0', fontweight='bold')
        axes_r[1].legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_r)


# ════════════════════════════════════════════════════════════════
# HALAMAN 4 — ANALISIS FITUR
# ════════════════════════════════════════════════════════════════
elif page == "📊 Analisis Fitur":
    st.markdown("""
    <div style='padding: 24px 0 8px 0;'>
        <h1 style='color:#ffffff; font-size:2rem; font-weight:800; margin:0;'>📊 Analisis Fitur Penting</h1>
        <p style='color:#64748b; margin-top:6px;'>Fitur keuangan yang paling berpengaruh dalam prediksi kebangkrutan.</p>
    </div>
    """, unsafe_allow_html=True)

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

    fig_fi, ax_fi = plt.subplots(figsize=(10, 5.5), facecolor='#0f172a')
    ax_fi.set_facecolor('#1e293b')
    ax_fi.tick_params(colors='#94a3b8')
    for spine in ax_fi.spines.values(): spine.set_edgecolor('#334155')

    colors_fi = plt.cm.plasma(np.linspace(0.2, 0.85, len(fi)))
    bars = ax_fi.barh(fi.index, fi.values, color=colors_fi, edgecolor='none', height=0.6)
    for b, v in zip(bars, fi.values):
        ax_fi.text(b.get_width() + 0.004, b.get_y() + b.get_height()/2,
                   f'{v:.4f}', va='center', color='#e2e8f0', fontsize=9, fontweight='600')
    ax_fi.set_xlabel('Importance Score', color='#94a3b8')
    ax_fi.set_title('Top 10 Feature Importance — Gradient Boosting', color='#e2e8f0',
                    fontweight='bold', pad=12)
    ax_fi.set_xlim(0, 0.43)
    plt.tight_layout()
    st.pyplot(fig_fi)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💡 Interpretasi Fitur Utama</div>", unsafe_allow_html=True)

    cards = [
        ("🥇", "Current Ratio", "0.357", "#3b82f6",
         "Fitur paling dominan. Nilai < 1 berarti perusahaan tidak mampu membayar kewajiban jangka pendek — sinyal terkuat risiko kebangkrutan."),
        ("🥈", "Tax Rate (A)", "0.186", "#8b5cf6",
         "Perusahaan tanpa beban pajak sering kali tidak menghasilkan laba. Rendahnya pajak menjadi indikator kesehatan finansial yang buruk."),
        ("🥉", "Cash Turnover Rate", "0.112", "#06b6d4",
         "Perputaran kas yang sangat rendah menandakan perusahaan kesulitan menghasilkan pendapatan dari aset kasnya — tanda likuiditas bermasalah."),
        ("4️⃣", "Total Asset Growth Rate", "0.092", "#10b981",
         "Pertumbuhan aset negatif menunjukkan penyusutan skala bisnis secara keseluruhan, yang memperlemah kemampuan perusahaan membayar utang."),
        ("5️⃣", "Inventory Turnover Rate", "0.058", "#f59e0b",
         "Perputaran inventori yang lambat berarti produk tidak terjual — menyumbat arus kas dan meningkatkan beban penyimpanan."),
    ]

    col_a, col_b = st.columns(2, gap="large")
    for i, (icon, name, score, color, desc) in enumerate(cards):
        col = col_a if i % 2 == 0 else col_b
        col.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border:1px solid {color}44; border-left: 3px solid {color};
                    border-radius:12px; padding:18px; margin-bottom:14px;'>
            <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
                <span style='font-size:1.3rem;'>{icon}</span>
                <span style='font-weight:700; color:#e2e8f0; font-size:0.95rem;'>{name}</span>
                <span style='margin-left:auto; background:{color}22; border:1px solid {color}66;
                             color:{color}; border-radius:20px; padding:2px 10px; font-size:0.78rem;
                             font-weight:700;'>{score}</span>
            </div>
            <p style='color:#94a3b8; font-size:0.85rem; margin:0; line-height:1.6;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    col_b.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:1px solid #33415544; border-left:3px solid #475569;
                border-radius:12px; padding:18px; margin-bottom:14px;'>
        <div style='font-weight:700; color:#e2e8f0; font-size:0.95rem; margin-bottom:8px;'>
            6️⃣–10️⃣ Rasio Likuiditas Lainnya
        </div>
        <p style='color:#94a3b8; font-size:0.85rem; margin:0; line-height:1.6;'>
            Cash/Total Assets, Current Assets/Total Assets, R&D Expense Rate,
            Quick Asset Turnover Rate, dan Quick Assets/Total Assets — semua
            mengukur kemampuan perusahaan memenuhi kewajiban jangka pendek
            dan efisiensi penggunaan aset likuid.
        </p>
    </div>
    """, unsafe_allow_html=True)
