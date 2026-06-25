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

st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    padding-top: 0;
}
[data-testid="stSidebar"] * { color: #1e293b !important; }

[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 2px 0;
    display: block;
    font-size: 0.88rem;
    font-weight: 500;
    color: #475569 !important;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #f1f5f9;
    color: #1e293b !important;
    border-color: #e2e8f0;
}

/* ── METRIC CARDS ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #16a34a !important;
    font-size: 0.78rem !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: #1e40af !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.8rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(30,64,175,0.25) !important;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(30,64,175,0.35) !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    background: #059669 !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #047857 !important;
    transform: translateY(-2px) !important;
}

/* ── CARD ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.card:hover {
    box-shadow: 0 6px 24px rgba(0,0,0,0.09);
    transform: translateY(-2px);
}
.card-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #1e40af;
    margin-bottom: 14px;
}
.card p, .card li { color: #475569; font-size: 0.88rem; line-height: 1.75; }

/* ── STEP ── */
.step {
    display: flex;
    align-items: flex-start;
    margin: 8px 0;
    gap: 12px;
}
.step-num {
    background: #eff6ff;
    color: #1e40af;
    border: 1.5px solid #bfdbfe;
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700;
    flex-shrink: 0;
}
.step-text { color: #475569; font-size: 0.87rem; padding-top: 3px; line-height: 1.5; }

/* ── SECTION TITLE ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}

/* ── BADGE ── */
.badge {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── INFO BOX ── */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #1e40af;
    border-radius: 10px;
    padding: 14px 18px;
    color: #1e3a8a;
    font-size: 0.87rem;
    margin-bottom: 20px;
}
.warn-box {
    background: #fefce8;
    border: 1px solid #fde68a;
    border-left: 4px solid #ca8a04;
    border-radius: 10px;
    padding: 14px 18px;
    color: #713f12;
    font-size: 0.87rem;
    margin-bottom: 20px;
}

/* ── PAGE HEADER ── */
.page-header {
    padding: 28px 0 4px 0;
    margin-bottom: 4px;
}
.page-header h1 {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.page-header p { color: #64748b; font-size: 0.92rem; margin: 0; }

/* ── INPUT ── */
[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 0.9rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #1e40af !important;
    box-shadow: 0 0 0 3px rgba(30,64,175,0.1) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; }

/* ── DIVIDER ── */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

/* ── MAIN BG ── */
.main .block-container { background: #f8fafc; padding-top: 0; }
</style>
""", unsafe_allow_html=True)


# ── TRAIN MODEL ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Melatih model, harap tunggu...")
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
<div style='padding: 28px 20px 20px 20px; border-bottom: 1px solid #e2e8f0;'>
    <div style='font-size: 1.15rem; font-weight: 800; color: #0f172a; letter-spacing:-0.02em;'>
        Bankruptcy Prediction
    </div>
    <div style='font-size: 0.75rem; color: #94a3b8; margin-top: 3px; font-weight:500;'>
        ML Dashboard &nbsp;·&nbsp; v1.0
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='padding: 16px 20px 0 20px;'>
    <div style='font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px;'>
        Model Performance
    </div>
    <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:12px 14px;'>
        <div style='display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f5f9;'>
            <span style='font-size:0.82rem; color:#64748b;'>ROC-AUC</span>
            <span style='font-size:0.82rem; font-weight:700; color:#1e40af;'>0.9044</span>
        </div>
        <div style='display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f5f9;'>
            <span style='font-size:0.82rem; color:#64748b;'>Threshold</span>
            <span style='font-size:0.82rem; font-weight:700; color:#d97706;'>0.7254</span>
        </div>
        <div style='display:flex; justify-content:space-between; padding:5px 0;'>
            <span style='font-size:0.82rem; color:#64748b;'>F1 (Bangkrut)</span>
            <span style='font-size:0.82rem; font-weight:700; color:#dc2626;'>0.42</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='padding: 20px 20px 8px 20px;'>
    <div style='font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 8px;'>
        Navigation
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "Beranda",
    "Prediksi Manual",
    "Prediksi CSV",
    "Analisis Fitur"
], label_visibility="collapsed")

st.sidebar.markdown("""
<div style='position:absolute; bottom:24px; left:0; right:0; padding:0 20px;'>
    <div style='border-top:1px solid #e2e8f0; padding-top:16px;'>
        <div style='font-size:0.72rem; color:#cbd5e1; line-height:1.6;'>
            Dataset: Taiwan Economic Journal<br>
            Periode 1999–2009 &nbsp;|&nbsp; 6.819 perusahaan
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HALAMAN 1 — BERANDA
# ════════════════════════════════════════════════════════════════
if page == "Beranda":
    st.markdown("""
    <div class='page-header'>
        <h1>Company Bankruptcy Prediction</h1>
        <p>Sistem prediksi kebangkrutan perusahaan berbasis Machine Learning — Gradient Boosting Classifier</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",    "0.9044", "Excellent")
    c2.metric("Recall",     "57%",    "Kelas Bangkrut")
    c3.metric("Threshold",  "0.7254", "Optimal")
    c4.metric("Total Data", "6.819",  "perusahaan")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Tentang Dataset</div>
            <table style='width:100%; border-collapse:collapse;'>
                <tr>
                    <td style='color:#94a3b8; padding:6px 0; font-size:0.83rem; width:42%;
                               border-bottom:1px solid #f1f5f9;'>Sumber</td>
                    <td style='color:#334155; font-size:0.83rem; font-weight:500;
                               border-bottom:1px solid #f1f5f9;'>Taiwan Economic Journal</td>
                </tr>
                <tr>
                    <td style='color:#94a3b8; padding:6px 0; font-size:0.83rem;
                               border-bottom:1px solid #f1f5f9;'>Periode</td>
                    <td style='color:#334155; font-size:0.83rem; font-weight:500;
                               border-bottom:1px solid #f1f5f9;'>1999 – 2009</td>
                </tr>
                <tr>
                    <td style='color:#94a3b8; padding:6px 0; font-size:0.83rem;
                               border-bottom:1px solid #f1f5f9;'>Total Data</td>
                    <td style='color:#334155; font-size:0.83rem; font-weight:500;
                               border-bottom:1px solid #f1f5f9;'>6.819 perusahaan</td>
                </tr>
                <tr>
                    <td style='color:#94a3b8; padding:6px 0; font-size:0.83rem;
                               border-bottom:1px solid #f1f5f9;'>Fitur Awal</td>
                    <td style='color:#334155; font-size:0.83rem; font-weight:500;
                               border-bottom:1px solid #f1f5f9;'>95 indikator keuangan</td>
                </tr>
                <tr>
                    <td style='color:#94a3b8; padding:6px 0; font-size:0.83rem;
                               border-bottom:1px solid #f1f5f9;'>Fitur Terpilih</td>
                    <td style='font-size:0.83rem; border-bottom:1px solid #f1f5f9;'>
                        <span class='badge'>31 fitur</span></td>
                </tr>
                <tr>
                    <td style='color:#94a3b8; padding:6px 0; font-size:0.83rem;
                               border-bottom:1px solid #f1f5f9;'>Class Ratio</td>
                    <td style='font-size:0.83rem; border-bottom:1px solid #f1f5f9;'>
                        <span class='badge'>96.77% vs 3.23%</span></td>
                </tr>
                <tr>
                    <td style='color:#94a3b8; padding:6px 0; font-size:0.83rem;'>Solusi Imbalance</td>
                    <td style='font-size:0.83rem;'><span class='badge'>SMOTE</span></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Alur Pemodelan</div>
            <div class='step'>
                <div class='step-num'>1</div>
                <div class='step-text'><b style='color:#1e293b;'>Load & EDA</b> — Eksplorasi distribusi data dan korelasi fitur</div>
            </div>
            <div class='step'>
                <div class='step-num'>2</div>
                <div class='step-text'><b style='color:#1e293b;'>Preprocessing</b> — Variance Threshold, seleksi 31 fitur terbaik</div>
            </div>
            <div class='step'>
                <div class='step-num'>3</div>
                <div class='step-text'><b style='color:#1e293b;'>SMOTE</b> — Balancing data training menjadi 50:50</div>
            </div>
            <div class='step'>
                <div class='step-num'>4</div>
                <div class='step-text'><b style='color:#1e293b;'>Modeling</b> — 3 algoritma dilatih dan dibandingkan</div>
            </div>
            <div class='step'>
                <div class='step-num'>5</div>
                <div class='step-text'><b style='color:#1e293b;'>Best Model</b> — Gradient Boosting terpilih (AUC 0.9044)</div>
            </div>
            <div class='step'>
                <div class='step-num'>6</div>
                <div class='step-text'><b style='color:#1e293b;'>Threshold Tuning</b> — Optimasi dari 0.50 menjadi 0.7254</div>
            </div>
            <div class='step'>
                <div class='step-num'>7</div>
                <div class='step-text'><b style='color:#1e293b;'>Deploy</b> — Dashboard publik via Streamlit Cloud</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Distribusi Data & Perbandingan Model</div>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor='#ffffff')
    for ax in axes:
        ax.set_facecolor('#f8fafc')
        ax.tick_params(colors='#64748b', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#e2e8f0')

    axes[0].bar(['Tidak Bangkrut', 'Bangkrut'], [6599, 220],
                color=['#3b82f6', '#f87171'], edgecolor='none', width=0.45)
    axes[0].set_title('Distribusi Kelas (Imbalanced)', color='#0f172a',
                      fontweight='bold', pad=12, fontsize=10)
    axes[0].set_ylabel('Jumlah', color='#64748b', fontsize=9)
    for i, v in enumerate([6599, 220]):
        axes[0].text(i, v + 60, f'{v:,}', ha='center', fontweight='600',
                     color='#1e293b', fontsize=9)

    mn  = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
    auc = [0.8368, 0.8817, 0.9044]
    clr = ['#94a3b8', '#60a5fa', '#1e40af']
    bars = axes[1].bar(mn, auc, color=clr, edgecolor='none', width=0.45)
    axes[1].set_title('Perbandingan ROC-AUC Model', color='#0f172a',
                      fontweight='bold', pad=12, fontsize=10)
    axes[1].set_ylabel('ROC-AUC Score', color='#64748b', fontsize=9)
    axes[1].set_ylim(0.80, 0.93)
    for b, v in zip(bars, auc):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                     f'{v:.4f}', ha='center', fontweight='600',
                     color='#1e293b', fontsize=9)

    plt.tight_layout(pad=2)
    st.pyplot(fig)

    st.markdown("<div class='section-title'>ROC Curve — Model Terbaik</div>",
                unsafe_allow_html=True)

    y_prob      = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)

    fig2, ax2 = plt.subplots(figsize=(8, 3.8), facecolor='#ffffff')
    ax2.set_facecolor('#f8fafc')
    ax2.tick_params(colors='#64748b', labelsize=9)
    for spine in ax2.spines.values(): spine.set_edgecolor('#e2e8f0')
    ax2.plot(fpr, tpr, color='#1e40af', lw=2.5,
             label=f'Gradient Boosting  (AUC = {auc_score:.4f})')
    ax2.fill_between(fpr, tpr, alpha=0.07, color='#1e40af')
    ax2.plot([0,1],[0,1],'--', color='#cbd5e1', lw=1.5, label='Random Classifier')
    ax2.set_xlabel('False Positive Rate', color='#64748b', fontsize=9)
    ax2.set_ylabel('True Positive Rate', color='#64748b', fontsize=9)
    ax2.set_title('ROC Curve — Gradient Boosting', color='#0f172a',
                  fontweight='bold', fontsize=10)
    ax2.legend(facecolor='#ffffff', edgecolor='#e2e8f0',
               labelcolor='#334155', fontsize=9)
    ax2.grid(alpha=0.4, color='#e2e8f0', linestyle='-')
    plt.tight_layout()
    st.pyplot(fig2)


# ════════════════════════════════════════════════════════════════
# HALAMAN 2 — PREDIKSI MANUAL
# ════════════════════════════════════════════════════════════════
elif page == "Prediksi Manual":
    st.markdown("""
    <div class='page-header'>
        <h1>Prediksi Manual</h1>
        <p>Masukkan indikator keuangan perusahaan untuk mendapatkan hasil prediksi kebangkrutan.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        Isi 10 fitur keuangan terpenting di bawah ini. Fitur lain akan diisi nilai 0 secara otomatis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>Input Fitur Keuangan</div>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    inputs = {}

    with col1:
        inputs[' Current Ratio']                         = st.number_input("Current Ratio",                 value=1.0,  min_value=0.0, step=0.01, help="Aset Lancar / Kewajiban Lancar. Sehat jika > 1")
        inputs[' Tax rate (A)']                          = st.number_input("Tax Rate (A)",                  value=0.2,  min_value=0.0, max_value=1.0, step=0.01)
        inputs[' Cash Turnover Rate']                    = st.number_input("Cash Turnover Rate",            value=0.5,  min_value=0.0, step=0.01)
        inputs[' Total Asset Growth Rate']               = st.number_input("Total Asset Growth Rate",       value=0.1,  step=0.01)
        inputs[' Inventory Turnover Rate (times)']       = st.number_input("Inventory Turnover Rate",       value=0.5,  min_value=0.0, step=0.01)

    with col2:
        inputs[' Cash/Total Assets']                     = st.number_input("Cash / Total Assets",           value=0.1,  min_value=0.0, max_value=1.0, step=0.01)
        inputs[' Current Assets/Total Assets']           = st.number_input("Current Assets / Total Assets", value=0.5,  min_value=0.0, max_value=1.0, step=0.01)
        inputs[' Research and development expense rate'] = st.number_input("R&D Expense Rate",              value=0.0,  min_value=0.0, step=0.001, format="%.3f")
        inputs[' Quick Asset Turnover Rate']             = st.number_input("Quick Asset Turnover Rate",     value=0.5,  min_value=0.0, step=0.01)
        inputs[' Quick Assets/Total Assets']             = st.number_input("Quick Assets / Total Assets",   value=0.4,  min_value=0.0, max_value=1.0, step=0.01)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Jalankan Prediksi", use_container_width=True):
        row_data = {col: 0.0 for col in feature_names}
        for col, val in inputs.items():
            if col in row_data:
                row_data[col] = float(val)

        row    = pd.DataFrame([row_data], columns=feature_names).astype(float)
        row_sc = scaler.transform(row)
        prob   = float(model.predict_proba(row_sc)[0][1])
        pred   = int(prob >= THRESHOLD)

        st.markdown("<br>", unsafe_allow_html=True)
        cr, cg = st.columns([1.1, 1], gap="large")

        with cr:
            if pred == 1:
                st.markdown(f"""
                <div style='background:#fff5f5; border:1px solid #fecaca;
                            border-left:4px solid #dc2626; border-radius:14px; padding:24px;'>
                    <div style='font-size:0.75rem; font-weight:700; text-transform:uppercase;
                                letter-spacing:0.07em; color:#dc2626; margin-bottom:10px;'>
                        Hasil Prediksi
                    </div>
                    <div style='font-size:1.5rem; font-weight:800; color:#991b1b;
                                margin-bottom:4px;'>Berpotensi Bangkrut</div>
                    <div style='font-size:3rem; font-weight:900; color:#dc2626;
                                line-height:1;'>{prob*100:.1f}%</div>
                    <div style='font-size:0.8rem; color:#b91c1c; margin-top:4px;
                                margin-bottom:16px;'>Probabilitas Kebangkrutan</div>
                    <div style='border-top:1px solid #fecaca; padding-top:14px;'>
                        <div style='font-size:0.8rem; font-weight:600; color:#7f1d1d;
                                    margin-bottom:8px;'>Rekomendasi Tindakan:</div>
                        <ul style='color:#991b1b; font-size:0.83rem; margin:0; padding-left:18px; line-height:1.9;'>
                            <li>Segera audit laporan keuangan perusahaan</li>
                            <li>Evaluasi dan perbaiki rasio likuiditas</li>
                            <li>Pertimbangkan restrukturisasi utang</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:#f0fdf4; border:1px solid #bbf7d0;
                            border-left:4px solid #16a34a; border-radius:14px; padding:24px;'>
                    <div style='font-size:0.75rem; font-weight:700; text-transform:uppercase;
                                letter-spacing:0.07em; color:#16a34a; margin-bottom:10px;'>
                        Hasil Prediksi
                    </div>
                    <div style='font-size:1.5rem; font-weight:800; color:#14532d;
                                margin-bottom:4px;'>Kondisi Keuangan Sehat</div>
                    <div style='font-size:3rem; font-weight:900; color:#16a34a;
                                line-height:1;'>{prob*100:.1f}%</div>
                    <div style='font-size:0.8rem; color:#15803d; margin-top:4px;
                                margin-bottom:16px;'>Probabilitas Kebangkrutan</div>
                    <div style='border-top:1px solid #bbf7d0; padding-top:14px;'>
                        <div style='font-size:0.8rem; font-weight:600; color:#14532d;
                                    margin-bottom:8px;'>Rekomendasi:</div>
                        <ul style='color:#166534; font-size:0.83rem; margin:0; padding-left:18px; line-height:1.9;'>
                            <li>Pertahankan rasio likuiditas saat ini</li>
                            <li>Monitor pertumbuhan aset secara berkala</li>
                            <li>Lanjutkan strategi keuangan yang ada</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with cg:
            fig_g, ax_g = plt.subplots(figsize=(5, 3.5), facecolor='#ffffff')
            ax_g.set_facecolor('#f8fafc')
            for spine in ax_g.spines.values(): spine.set_edgecolor('#e2e8f0')
            ax_g.tick_params(colors='#64748b', labelsize=9)

            color  = '#dc2626' if pred == 1 else '#16a34a'
            color2 = '#fee2e2' if pred == 1 else '#dcfce7'
            ax_g.barh([''], [prob],       color=color,  height=0.35, label='Skor Risiko')
            ax_g.barh([''], [1 - prob],   left=[prob],  color=color2, height=0.35)
            ax_g.axvline(THRESHOLD, color='#d97706', linestyle='--',
                         lw=1.8, label=f'Threshold ({THRESHOLD})')
            ax_g.set_xlim(0, 1)
            ax_g.set_xlabel('Probabilitas', color='#64748b', fontsize=9)
            ax_g.set_title(f'Skor Risiko: {prob*100:.1f}%', color='#0f172a',
                           fontweight='bold', fontsize=10)
            ax_g.legend(facecolor='#ffffff', edgecolor='#e2e8f0',
                        labelcolor='#334155', fontsize=8)
            ax_g.text(prob / 2, 0, f'{prob*100:.1f}%', ha='center',
                      va='center', color='white', fontweight='700', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_g)


# ════════════════════════════════════════════════════════════════
# HALAMAN 3 — PREDIKSI CSV
# ════════════════════════════════════════════════════════════════
elif page == "Prediksi CSV":
    st.markdown("""
    <div class='page-header'>
        <h1>Prediksi Batch via CSV</h1>
        <p>Upload file CSV berisi data keuangan banyak perusahaan untuk diprediksi sekaligus.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class='warn-box'>
        <b>Format CSV:</b> Pastikan nama kolom sesuai dengan header dataset asli.
        Kolom <code>Bankrupt?</code> boleh ada atau tidak — keduanya didukung.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"""
        <div class='info-box'>
            File berhasil dibaca — <b>{df_up.shape[0]} baris</b> x <b>{df_up.shape[1]} kolom</b>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Preview data (5 baris pertama)"):
            st.dataframe(df_up.head(), use_container_width=True)

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
        ca.metric("Total Perusahaan", len(preds))
        cb.metric("Prediksi Bangkrut", n_bangkrut, f"{pct:.1f}%")
        cc.metric("Prediksi Sehat",    n_sehat)
        cd.metric("Threshold",         THRESHOLD)

        st.markdown("<div class='section-title'>Hasil Prediksi</div>",
                    unsafe_allow_html=True)

        df_result = pd.DataFrame({
            'No': range(1, len(preds)+1),
            'Probabilitas Bangkrut (%)': (probs * 100).round(2),
            'Prediksi': preds,
            'Status': ['Bangkrut' if p == 1 else 'Sehat' for p in preds]
        })
        st.dataframe(df_result, use_container_width=True, height=380)

        csv_out = df_result.to_csv(index=False)
        st.download_button("Download Hasil Prediksi (CSV)", csv_out,
                           "hasil_prediksi.csv", "text/csv",
                           use_container_width=True)

        st.markdown("<div class='section-title'>Ringkasan Hasil</div>",
                    unsafe_allow_html=True)

        fig_r, axes_r = plt.subplots(1, 2, figsize=(11, 3.8), facecolor='#ffffff')
        for ax in axes_r:
            ax.set_facecolor('#f8fafc')
            ax.tick_params(colors='#64748b', labelsize=9)
            for spine in ax.spines.values(): spine.set_edgecolor('#e2e8f0')

        axes_r[0].pie([n_sehat, n_bangkrut],
                      labels=['Sehat', 'Bangkrut'],
                      colors=['#3b82f6', '#f87171'],
                      autopct='%1.1f%%',
                      textprops={'color':'#1e293b', 'fontsize':9},
                      startangle=90,
                      wedgeprops={'edgecolor':'white', 'linewidth':2})
        axes_r[0].set_title('Proporsi Prediksi', color='#0f172a',
                             fontweight='bold', fontsize=10)

        axes_r[1].hist(probs * 100, bins=20, color='#3b82f6',
                       edgecolor='white', alpha=0.85)
        axes_r[1].axvline(THRESHOLD * 100, color='#d97706',
                          linestyle='--', lw=2,
                          label=f'Threshold ({THRESHOLD*100:.1f}%)')
        axes_r[1].set_xlabel('Probabilitas Bangkrut (%)', color='#64748b', fontsize=9)
        axes_r[1].set_ylabel('Jumlah Perusahaan', color='#64748b', fontsize=9)
        axes_r[1].set_title('Distribusi Probabilitas', color='#0f172a',
                             fontweight='bold', fontsize=10)
        axes_r[1].legend(facecolor='#ffffff', edgecolor='#e2e8f0',
                         labelcolor='#334155', fontsize=8)
        axes_r[1].grid(alpha=0.4, color='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig_r)


# ════════════════════════════════════════════════════════════════
# HALAMAN 4 — ANALISIS FITUR
# ════════════════════════════════════════════════════════════════
elif page == "Analisis Fitur":
    st.markdown("""
    <div class='page-header'>
        <h1>Analisis Fitur Penting</h1>
        <p>Indikator keuangan yang paling berpengaruh terhadap prediksi kebangkrutan perusahaan.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

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

    fig_fi, ax_fi = plt.subplots(figsize=(10, 5), facecolor='#ffffff')
    ax_fi.set_facecolor('#f8fafc')
    ax_fi.tick_params(colors='#64748b', labelsize=9)
    for spine in ax_fi.spines.values(): spine.set_edgecolor('#e2e8f0')

    # Warna gradasi biru
    n = len(fi)
    colors_fi = [plt.cm.Blues(0.35 + 0.55 * i / (n-1)) for i in range(n)]
    bars = ax_fi.barh(fi.index, fi.values, color=colors_fi,
                      edgecolor='none', height=0.55)
    for b, v in zip(bars, fi.values):
        ax_fi.text(b.get_width() + 0.004, b.get_y() + b.get_height()/2,
                   f'{v:.4f}', va='center', color='#334155',
                   fontsize=9, fontweight='600')

    ax_fi.set_xlabel('Importance Score', color='#64748b', fontsize=9)
    ax_fi.set_title('Top 10 Feature Importance — Gradient Boosting',
                    color='#0f172a', fontweight='bold', fontsize=10, pad=12)
    ax_fi.set_xlim(0, 0.43)
    ax_fi.grid(axis='x', alpha=0.4, color='#e2e8f0')
    plt.tight_layout()
    st.pyplot(fig_fi)

    st.markdown("<div class='section-title'>Interpretasi Fitur Utama</div>",
                unsafe_allow_html=True)

    interpretasi = [
        ("01", "Current Ratio", "0.357",
         "#1e40af", "#eff6ff", "#bfdbfe",
         "Fitur paling dominan. Nilai di bawah 1 menandakan perusahaan tidak mampu membayar kewajiban jangka pendek — sinyal terkuat risiko kebangkrutan."),
        ("02", "Tax Rate (A)", "0.186",
         "#7c3aed", "#f5f3ff", "#ddd6fe",
         "Perusahaan tanpa beban pajak sering kali tidak menghasilkan laba. Tax rate rendah menjadi indikator kesehatan finansial yang buruk."),
        ("03", "Cash Turnover Rate", "0.112",
         "#0369a1", "#f0f9ff", "#bae6fd",
         "Perputaran kas yang sangat rendah menandakan perusahaan kesulitan menghasilkan pendapatan dari aset kasnya."),
        ("04", "Total Asset Growth Rate", "0.092",
         "#065f46", "#f0fdf4", "#bbf7d0",
         "Pertumbuhan aset negatif menunjukkan penyusutan skala bisnis, yang memperlemah kemampuan perusahaan membayar kewajiban."),
        ("05", "Inventory Turnover Rate", "0.058",
         "#92400e", "#fffbeb", "#fde68a",
         "Perputaran inventori lambat berarti produk tidak terjual — menyumbat arus kas dan meningkatkan beban penyimpanan."),
        ("06–10", "Rasio Likuiditas Lainnya", "0.016–0.026",
         "#475569", "#f8fafc", "#e2e8f0",
         "Cash/Total Assets, Current Assets/Total Assets, R&D Expense Rate, Quick Asset Turnover Rate, dan Quick Assets/Total Assets — semua mengukur kemampuan memenuhi kewajiban jangka pendek."),
    ]

    col_a, col_b = st.columns(2, gap="large")
    for i, (num, name, score, tc, bg, border_c, desc) in enumerate(interpretasi):
        col = col_a if i % 2 == 0 else col_b
        col.markdown(f"""
        <div style='background:{bg}; border:1px solid {border_c};
                    border-radius:12px; padding:18px 20px; margin-bottom:14px;
                    transition: box-shadow 0.2s;'>
            <div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;'>
                <div style='display:flex; align-items:center; gap:10px;'>
                    <span style='background:{tc}; color:white; border-radius:6px;
                                 padding:2px 8px; font-size:0.7rem; font-weight:700;'>
                        {num}
                    </span>
                    <span style='font-weight:700; color:#1e293b; font-size:0.92rem;'>{name}</span>
                </div>
                <span style='font-size:0.78rem; font-weight:700; color:{tc};'>{score}</span>
            </div>
            <p style='color:#475569; font-size:0.84rem; margin:0; line-height:1.65;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
