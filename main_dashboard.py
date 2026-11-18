import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ─── KONFIGURASI HALAMAN ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluasi Model Depresi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── INJEKSI CSS KUSTOM ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Reset & Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0D1B2A;
    color: #C8D8E8;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── HERO HEADER ── */
.hero-header {
    background: linear-gradient(135deg, #1A2D42 0%, #0D1B2A 60%, #0a2233 100%);
    border: 1px solid #1E3A52;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,180,216,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #E8F4FC;
    letter-spacing: -0.5px;
    margin: 0 0 0.5rem 0;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #8B9EB7;
    margin: 0;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,180,216,0.15);
    border: 1px solid rgba(0,180,216,0.4);
    color: #00B4D8;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── METRIC CARDS ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #1A2D42;
    border: 1px solid #1E3A52;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 12px 12px;
}
.metric-card.teal::after { background: linear-gradient(90deg, #00B4D8, #0090ad); }
.metric-card.amber::after { background: linear-gradient(90deg, #F4A261, #e07c30); }
.metric-card.emerald::after { background: linear-gradient(90deg, #2DD4BF, #14b8a6); }
.metric-card.violet::after { background: linear-gradient(90deg, #A78BFA, #7c3aed); }

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8B9EB7;
    margin-bottom: 0.5rem;
}
.metric-values {
    display: flex;
    align-items: baseline;
    gap: 1rem;
}
.metric-val-knn {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #00B4D8;
}
.metric-val-nb {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 400;
    color: #F4A261;
}
.metric-algo-labels {
    display: flex;
    gap: 0.75rem;
    margin-top: 0.4rem;
}
.chip {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 2px 8px;
    border-radius: 20px;
}
.chip-knn { background: rgba(0,180,216,0.15); color: #00B4D8; }
.chip-nb  { background: rgba(244,162,97,0.15); color: #F4A261; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1A2D42;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1E3A52;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    color: #8B9EB7;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.6rem 1.4rem;
    border: none;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: #0D1B2A !important;
    color: #00B4D8 !important;
    border: 1px solid #1E3A52 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* ── SECTION HEADERS ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.section-icon {
    width: 36px; height: 36px;
    background: rgba(0,180,216,0.12);
    border: 1px solid rgba(0,180,216,0.3);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #E8F4FC;
}
.section-desc {
    font-size: 0.82rem;
    color: #8B9EB7;
    margin-top: 2px;
}

/* ── INSIGHT BOX ── */
.insight-box {
    background: rgba(0,180,216,0.06);
    border: 1px solid rgba(0,180,216,0.2);
    border-left: 3px solid #00B4D8;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 1.5rem;
    font-size: 0.85rem;
    color: #A8C8E0;
    line-height: 1.65;
}
.insight-box strong { color: #00B4D8; }

/* ── CUSTOM TABLE ── */
.custom-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border: 1px solid #1E3A52;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.875rem;
}
.custom-table thead tr {
    background: #142334;
}
.custom-table th {
    padding: 0.75rem 1rem;
    text-align: left;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #8B9EB7;
    border-bottom: 1px solid #1E3A52;
}
.custom-table tbody tr {
    background: #1A2D42;
    transition: background 0.15s ease;
}
.custom-table tbody tr:nth-child(even) {
    background: #162638;
}
.custom-table tbody tr:hover {
    background: #1E3650;
}
.custom-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #1E3A52;
    vertical-align: middle;
}
.custom-table tbody tr:last-child td {
    border-bottom: none;
}
</style>
""", unsafe_allow_html=True)

# FUNGSI ANALISIS DATASET
# Label yang digunakan: 1 = Depresi, 0 = Non-Depresi
# Tema isu dihitung dari keyword pada full_text data label 1.
# ============================================================
ISSUE_KEYWORDS = {
    "Kesedihan dan Tekanan Emosional": [
        r"\bsedih\b", r"\bkecewa\b", r"\bmenangis\b",
        r"\bterluka\b", r"\bsakit hati\b"
    ],
    "Rasa Tidak Berharga dan Kegagalan": [
        r"\btidak berharga\b", r"\bga berguna\b",
        r"\bgak berguna\b", r"\btidak berguna\b",
        r"\bgagal\b", r"\binsecure\b"
    ],
    "Kecemasan dan Tekanan": [
        r"\bcemas\b", r"\btakut\b", r"\bstres\b",
        r"\bstress\b", r"\btertekan\b", r"\bpanik\b"
    ],
    "Kesepian dan Hubungan Sosial": [
        r"\bkesepian\b", r"\bsepi\b", r"\bsendiri\b",
        r"\bditinggalkan\b", r"\bteman\b",
        r"\bkeluarga\b", r"\bputus\b"
    ],
    "Keputusasaan dan Kelelahan Hidup": [
        r"\bputus asa\b", r"\bmenyerah\b",
        r"\blelah hidup\b", r"\bcapek hidup\b",
        r"\bbosan hidup\b", r"\bhopeless\b",
        r"\bdepressed\b"
    ]
}

MONTH_NAMES = {
    "01": "Januari", "02": "Februari", "03": "Maret",
    "04": "April", "05": "Mei", "06": "Juni",
    "07": "Juli", "08": "Agustus", "09": "September",
    "10": "Oktober", "11": "November", "12": "Desember"
}


def build_dataset_analysis(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {csv_path.name}. "
            "Letakkan file CSV satu folder dengan main.py."
        )

    df = pd.read_csv(csv_path)

    required = {"id_str", "created_at", "full_text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Kolom berikut tidak ditemukan pada dataset: "
            + ", ".join(sorted(missing))
        )

    before = len(df)
    df_unique = df.drop_duplicates().reset_index(drop=True).copy()
    duplicate_count = before - len(df_unique)

    df_unique["label"] = pd.to_numeric(
        df_unique["label"], errors="coerce"
    )
    df_unique = df_unique.dropna(subset=["label"]).copy()
    df_unique["label"] = df_unique["label"].astype(int)
    df_unique = df_unique[df_unique["label"].isin([0, 1])].copy()

    # Konversi waktu ke WIB
    df_unique["tanggal_wib"] = pd.to_datetime(
        df_unique["created_at"], utc=True, errors="coerce"
    ).dt.tz_convert("Asia/Jakarta")
    df_unique = df_unique.dropna(subset=["tanggal_wib"]).copy()

    total = int(len(df_unique))
    count_depresi = int((df_unique["label"] == 1).sum())
    count_non = int((df_unique["label"] == 0).sum())

    pct_depresi = count_depresi / total * 100 if total else 0
    pct_non = count_non / total * 100 if total else 0

    # Rekap bulan: jumlah seluruh unggahan dan per label
    df_unique["bulan_kode"] = df_unique["tanggal_wib"].dt.strftime("%Y-%m")
    monthly = (
        df_unique.groupby(["bulan_kode", "label"])
        .size().unstack(fill_value=0)
    )
    monthly = monthly.reindex(columns=[0, 1], fill_value=0)
    monthly = monthly.rename(columns={
        0: "Non-Depresi", 1: "Depresi"
    })
    monthly["Total Unggahan"] = (
        monthly["Non-Depresi"] + monthly["Depresi"]
    )
    monthly = monthly.reset_index().sort_values("bulan_kode")
    monthly["Bulan"] = monthly["bulan_kode"].apply(
        lambda x: MONTH_NAMES[x[5:]] + " " + x[:4]
    )
    monthly = monthly[
        ["bulan_kode", "Bulan", "Depresi", "Non-Depresi", "Total Unggahan"]
    ]

    if len(monthly):
        top_month_row = monthly.loc[monthly["Total Unggahan"].idxmax()]
        top_month = {
            "bulan": str(top_month_row["Bulan"]),
            "jumlah": int(top_month_row["Total Unggahan"])
        }
        top_depression_row = monthly.loc[monthly["Depresi"].idxmax()]
        top_depression_month = {
            "bulan": str(top_depression_row["Bulan"]),
            "jumlah": int(top_depression_row["Depresi"])
        }
    else:
        top_month = {"bulan": "-", "jumlah": 0}
        top_depression_month = {"bulan": "-", "jumlah": 0}

    # Hitung tema dari unggahan berlabel Depresi.
    # Satu unggahan dapat cocok dengan lebih dari satu tema.
    depression_text = (
        df_unique.loc[df_unique["label"] == 1, "full_text"]
        .fillna("").astype(str).str.lower()
    )
    issue_rows = []
    for issue, keywords in ISSUE_KEYWORDS.items():
        pattern = "|".join(keywords)
        number = int(depression_text.str.contains(
            pattern, regex=True, na=False
        ).sum())
        issue_rows.append({
            "Tema Isu": issue,
            "Jumlah Unggahan": number
        })

    issue_df = pd.DataFrame(issue_rows).sort_values(
        "Jumlah Unggahan", ascending=False
    ).reset_index(drop=True)

    if len(issue_df) and int(issue_df.iloc[0]["Jumlah Unggahan"]) > 0:
        top_issue = {
            "tema": str(issue_df.iloc[0]["Tema Isu"]),
            "jumlah": int(issue_df.iloc[0]["Jumlah Unggahan"])
        }
    else:
        top_issue = {"tema": "-", "jumlah": 0}

    return {
        "jumlah_data": {
            "total": total,
            "depresi": count_depresi,
            "non_depresi": count_non,
            "persen_depresi": round(pct_depresi, 2),
            "persen_non_depresi": round(pct_non, 2),
            "duplikat_dihapus": int(duplicate_count)
        },
        "distribusi_bulanan": monthly.to_dict(orient="records"),
        "bulan_terbanyak": top_month,
        "bulan_depresi_terbanyak": top_depression_month,
        "distribusi_isu": issue_df.to_dict(orient="records"),
        "isu_terbanyak": top_issue,
        "metode_analisis_isu": "Pencocokan keyword pada full_text dengan label 1",
        "label_mapping": {"0": "Non-Depresi", "1": "Depresi"}
    }


# ─── MEMUAT HASIL PENGUJIAN DARI PREPROCESSING ─────────────────────────────────
import streamlit as st
import joblib
from pathlib import Path

# Menentukan lokasi folder tempat file Python berada
BASE_DIR = Path(__file__).resolve().parent

# Memanggil file hasil pengujian model
file_hasil = BASE_DIR / "dashboard_results.pkl"

if not file_hasil.exists():
    st.error(
        "File dashboard_results.pkl belum ditemukan. "
        "Simpan hasil pengujian dari Google Colab, "
        "lalu letakkan file tersebut satu folder dengan main.py."
    )
    st.stop()

hasil = joblib.load(file_hasil)

y_true = hasil["y_test"]
y_pred_knn = hasil["y_pred_knn"]
y_proba_knn = hasil["y_prob_knn"]
y_pred_nb = hasil["y_pred_nb"]
y_proba_nb = hasil["y_prob_nb"]

# Memanggil file dataset interpretasi
FILE_DATASET = BASE_DIR / "Dataset_Hasil_NLP (2).csv"

if not FILE_DATASET.exists():
    st.error(
        "File Dataset_Hasil_NLP (2).csv belum ditemukan. "
        "Letakkan dataset satu folder dengan main.py."
    )
    st.stop()
try:
    data_info = build_dataset_analysis(FILE_DATASET)

    # Mengambil ringkasan jumlah data
    counts = data_info["jumlah_data"]

    # Mengubah distribusi bulanan menjadi DataFrame
    monthly_df = pd.DataFrame(
        data_info["distribusi_bulanan"],
        columns=[
            "bulan_kode",
            "Bulan",
            "Depresi",
            "Non-Depresi",
            "Total Unggahan"
        ]
    )

    # Mengubah distribusi tema menjadi DataFrame
    issue_df = pd.DataFrame(
        data_info["distribusi_isu"],
        columns=[
            "Tema Isu",
            "Jumlah Unggahan"
        ]
    )

except Exception as e:
    st.error(f"Gagal menganalisis dataset: {e}")
    st.stop()

# file vectorizer

_candidates_vectorizer = [
    BASE_DIR / "tfidf_vectorizer(1).pkl",
    BASE_DIR / "tfidf_vectorizer.pkl",
]
FILE_VECTORIZER = next(
    (path for path in _candidates_vectorizer if path.exists()),
    _candidates_vectorizer[0]
)
FILE_MODEL_NB = BASE_DIR / "model_nb.pkl"
FILE_MODEL_KNN = BASE_DIR / "model_knn.pkl"



# ─── MATPLOTLIB DARK THEME ─────────────────────────────────────────────────────
def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  '#1A2D42',
        'axes.facecolor':    '#1A2D42',
        'axes.edgecolor':    '#1E3A52',
        'axes.labelcolor':   '#8B9EB7',
        'xtick.color':       '#8B9EB7',
        'ytick.color':       '#8B9EB7',
        'text.color':        '#C8D8E8',
        'grid.color':        '#1E3A52',
        'grid.alpha':        0.1,
        'font.family':       'sans-serif',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

set_dark_style()

KNN_COLOR = '#00B4D8'
NB_COLOR  = '#F4A261' 
BG_COLOR = "#1A2D42"

# ─── HITUNG METRIK ─────────────────────────────────────────────────────────────
def calc(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred)  * 100,
        "Precision": precision_score(y_true, y_pred, average="binary", zero_division=0) * 100,
        "Recall":    recall_score(y_true, y_pred, average="binary", zero_division=0) * 100,
        "F1-Score":  f1_score(y_true, y_pred, average="binary", zero_division=0) * 100,
    }

m_knn = calc(y_true, y_pred_knn)
m_nb  = calc(y_true, y_pred_nb)


# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🧠 Research Dashboard </div>
    <h1 class="hero-title">Evaluasi Komparatif Model Deteksi Depresi</h1>
    <p class="hero-subtitle">
        Perbandingan performa algoritma <strong style="color:#00B4D8">K-Nearest Neighbors (KNN)</strong>
        dan <strong style="color:#F4A261">Naive Bayes</strong> berdasarkan hasil pengujian
        pada dataset klinis. Pilih tab di bawah untuk eksplorasi metrik, confusion matrix,
        dan kurva ROC secara terpisah.
    </p>
</div>
""", unsafe_allow_html=True)


# ─── METRIC CARDS ──────────────────────────────────────────────────────────────
colors = ["teal", "amber", "emerald", "violet"]
metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]

cards_html = '<div class="metric-row">'
for i, metric in enumerate(metrics_list):
    cards_html += f"""
    <div class="metric-card {colors[i]}">
        <div class="metric-label">{metric}</div>
        <div class="metric-values">
            <span class="metric-val-knn">{m_knn[metric]:.2f}<span style="font-size:0.9rem;color:#8B9EB7">%</span></span>
            <span class="metric-val-nb">{m_nb[metric]:.2f}%</span>
        </div>
        <div class="metric-algo-labels">
            <span class="chip chip-knn">KNN</span>
            <span class="chip chip-nb">Naive Bayes</span>
        </div>
    </div>"""
cards_html += '</div>'
st.markdown(cards_html, unsafe_allow_html=True)


# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
     "📊  Classification Report",
     "🔲  Confusion Matrix",
     "📈  Kurva ROC & AUC",
     "📌 Interpretasi Data",
     "🧪 Deteksi Teks"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFICATION REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <div>
            <div class="section-title">Tabel & Grafik Metrik Klasifikasi</div>
            <div class="section-desc">Perbandingan Accuracy, Precision, Recall, dan F1-Score kedua model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_metrics = pd.DataFrame({
        "Metrik":              metrics_list,
        "KNN (%)":             [round(m_knn[m], 2) for m in metrics_list],
        "Naive Bayes (%)":     [round(m_nb[m], 2)  for m in metrics_list],
        "Selisih (KNN − NB)":  [round(m_knn[m] - m_nb[m], 2) for m in metrics_list],
    })

    col_t, col_g = st.columns([1, 1.6], gap="large")

    with col_t:
        # Cari nilai max untuk highlight
        max_knn = max(m_knn[m] for m in metrics_list)
        max_nb  = max(m_nb[m]  for m in metrics_list)

        rows_html = ""
        for m in metrics_list:
            knn_val  = m_knn[m]
            nb_val   = m_nb[m]
            diff     = knn_val - nb_val
            diff_color = '#2DD4BF' if diff > 0 else '#F87171' if diff < 0 else '#8B9EB7'
            knn_bg   = 'background:rgba(0,180,216,0.12);' if knn_val == max_knn else ''
            nb_bg    = 'background:rgba(244,162,97,0.12);' if nb_val  == max_nb  else ''
            rows_html += f"""
            <tr>
                <td style="color:#C8D8E8;font-weight:500">{m}</td>
                <td style="color:#00B4D8;font-family:'JetBrains Mono',monospace;font-weight:600;{knn_bg}">{knn_val:.2f}</td>
                <td style="color:#F4A261;font-family:'JetBrains Mono',monospace;font-weight:600;{nb_bg}">{nb_val:.2f}</td>
                <td style="color:{diff_color};font-family:'JetBrains Mono',monospace;font-weight:700">{diff:+.2f}</td>
            </tr>"""

        st.markdown(f"""
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Metrik</th>
                    <th><span style="color:#00B4D8">■</span> KNN (%)</th>
                    <th><span style="color:#F4A261">■</span> Naive Bayes (%)</th>
                    <th>Selisih (KNN − NB)</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        best_metrics = [m for m in metrics_list if m_nb[m] > m_knn[m]]
        if all(m_nb[m] > m_knn[m] for m in metrics_list):
            conclusion = (
                f"Naive Bayes memperoleh nilai lebih tinggi pada seluruh metrik "
                f"yang ditampilkan. Selisihnya: Accuracy {m_nb['Accuracy']-m_knn['Accuracy']:.2f}, "
                f"Precision {m_nb['Precision']-m_knn['Precision']:.2f}, "
                f"Recall berbobot {m_nb['Recall']-m_knn['Recall']:.2f}, dan "
                f"F1-Score berbobot {m_nb['F1-Score']-m_knn['F1-Score']:.2f} poin persentase. "
                "Berdasarkan metrik klasifikasi ini, Naive Bayes lebih baik pada data uji ini."
            )
        elif all(m_knn[m] > m_nb[m] for m in metrics_list):
            conclusion = "KNN memperoleh nilai lebih tinggi pada seluruh metrik klasifikasi yang ditampilkan."
        else:
            conclusion = (
                "Keunggulan metrik terbagi antara kedua model. Bandingkan F1-Score dan "
                "recall khusus kelas Depresi sesuai tujuan evaluasi, bukan hanya accuracy."
            )
        st.markdown(f"""
        <div class="insight-box">
            <strong>💡 Kesimpulan:</strong><br>
            {conclusion}<br>
            <span style="color:#8B9EB7">Catatan: Recall pada tabel ini adalah recall berbobot untuk semua kelas;
            recall khusus kelas Depresi dijelaskan pada bagian Confusion Matrix.</span>
        </div>
        """, unsafe_allow_html=True)

    with col_g:
        fig, ax = plt.subplots(figsize=(8, 5.95))
        fig.patch.set_facecolor('#1A2D42')

        x = np.arange(len(metrics_list))
        w = 0.35
        bars_knn = ax.bar(x - w/2, [m_knn[m] for m in metrics_list], w,
                          color=KNN_COLOR, alpha=0.9, label='KNN', zorder=3,
                          linewidth=0)
        bars_nb  = ax.bar(x + w/2, [m_nb[m] for m in metrics_list], w,
                          color=NB_COLOR,  alpha=0.9, label='Naive Bayes', zorder=3,
                          linewidth=0)

        for bar in bars_knn:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=8.5, color=KNN_COLOR, fontweight='600')
        for bar in bars_nb:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=8.5, color=NB_COLOR, fontweight='600')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_list, fontsize=10)
        ax.set_ylim(75, 102)
        ax.set_ylabel('Skor (%)', fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        legend = ax.legend(loc='lower right', framealpha=0.2,
                           labelcolor='white', fontsize=9,
                           facecolor='#0D1B2A', edgecolor='#1E3A52')
        st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔲</div>
        <div>
            <div class="section-title">Confusion Matrix — KNN vs Naive Bayes</div>
            <div class="section-desc">Distribusi prediksi benar (TP/TN) dan salah (FP/FN) pada data uji</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    labels = ['Non-Depresi', 'Depresi']
    cm_knn = confusion_matrix(y_true, y_pred_knn)
    cm_nb  = confusion_matrix(y_true, y_pred_nb)

    def plot_cm(cm, title, cmap_start, accent_color):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor('#1A2D42')

        # Custom colormap
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('custom',
            ['#1A2D42', cmap_start], N=256)

        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=2, linecolor='#0D1B2A',
                    annot_kws={"size": 18, "weight": "bold", "color": "white"},
                    cbar_kws={'shrink': 0.8})

        ax.set_xlabel('Prediksi Model', fontsize=10, labelpad=10)
        ax.set_ylabel('Kondisi Aktual', fontsize=10, labelpad=10)
        ax.set_title(title, fontsize=11, fontweight='600', color='#E8F4FC', pad=14)
        ax.tick_params(labelsize=9)

        # Annotate cells
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.72, f'({pct:.1f}%)',
                        ha='center', va='center', fontsize=8,
                        color=(200/255, 216/255, 232/255, 0.7))

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors='#8B9EB7', labelsize=8)

        fig.tight_layout()
        return fig

    col_cm1, col_cm2 = st.columns(2, gap="large")
    with col_cm1:
        fig_knn = plot_cm(cm_knn, "K-Nearest Neighbors (KNN)", '#005F73', KNN_COLOR)
        st.pyplot(fig_knn, use_container_width=True)

    with col_cm2:
        fig_nb = plot_cm(cm_nb, "Naive Bayes", '#7B4B00', NB_COLOR)
        st.pyplot(fig_nb, use_container_width=True)

    # Breakdown tabel TP/TN/FP/FN
    st.markdown("<br>", unsafe_allow_html=True)
    tn_k, fp_k, fn_k, tp_k = cm_knn.ravel()
    tn_n, fp_n, fn_n, tp_n = cm_nb.ravel()

    cm_rows = [
        ("True Positive (TP)",  "Depresi → Depresi",       "✓", tp_k, tp_n, "#2DD4BF"),
        ("True Negative (TN)",  "Non-Depresi → Non-Depresi","✓", tn_k, tn_n, "#2DD4BF"),
        ("False Positive (FP)", "Non-Depresi → Depresi",   "✗", fp_k, fp_n, "#F87171"),
        ("False Negative (FN)", "Depresi → Non-Depresi",   "✗", fn_k, fn_n, "#F87171"),
    ]
    cm_rows_html = ""
    for komponen, desc, mark, val_k, val_n, mark_color in cm_rows:
        cm_rows_html += f"""
        <tr>
            <td style="color:#C8D8E8;font-weight:500">{komponen}</td>
            <td style="color:#8B9EB7">{desc} <span style="color:{mark_color};font-weight:700">{mark}</span></td>
            <td style="color:#00B4D8;font-family:'JetBrains Mono',monospace;font-weight:600;font-size:1.05rem">{val_k}</td>
            <td style="color:#F4A261;font-family:'JetBrains Mono',monospace;font-weight:600;font-size:1.05rem">{val_n}</td>
        </tr>"""

    st.markdown(f"""
    <table class="custom-table">
        <thead>
            <tr>
                <th>Komponen</th>
                <th>Deskripsi</th>
                <th><span style="color:#00B4D8">■</span> KNN</th>
                <th><span style="color:#F4A261">■</span> Naive Bayes</th>
            </tr>
        </thead>
        <tbody>{cm_rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    recall_dep_knn = tp_k / (tp_k + fn_k) * 100 if (tp_k + fn_k) else 0.0
    recall_dep_nb = tp_n / (tp_n + fn_n) * 100 if (tp_n + fn_n) else 0.0
    st.markdown(f"""
    <div class="insight-box">
        <strong>💡 Interpretasi Confusion Matrix:</strong><br>
        <strong>TP</strong> adalah kasus Depresi yang benar terdeteksi, <strong>TN</strong> adalah
        Non-Depresi yang benar dikenali, <strong>FP</strong> adalah Non-Depresi yang salah diprediksi
        Depresi, dan <strong>FN</strong> adalah kasus Depresi yang luput terdeteksi.<br>
        Pada data uji ini, KNN menghasilkan {fn_k} FN dan Naive Bayes {fn_n} FN.
        Recall khusus kelas Depresi = TP/(TP+FN): KNN <strong>{recall_dep_knn:.2f}%</strong>,
        Naive Bayes <strong>{recall_dep_nb:.2f}%</strong>.
        Jadi Naive Bayes lebih baik dalam menangkap kasus Depresi pada hasil ini karena FN lebih sedikit
        dan recall kelas Depresi lebih tinggi. Dalam penggunaan nyata, hasil model bukan diagnosis klinis.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROC CURVE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📈</div>
        <div>
            <div class="section-title">Kurva ROC & Area Under Curve (AUC)</div>
            <div class="section-desc">Kemampuan diskriminasi model pada berbagai nilai threshold keputusan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fpr_knn, tpr_knn, thresh_knn = roc_curve(y_true, y_proba_knn)
    auc_knn = auc(fpr_knn, tpr_knn)
    fpr_nb,  tpr_nb,  thresh_nb  = roc_curve(y_true, y_proba_nb)
    auc_nb  = auc(fpr_nb,  tpr_nb)

    # ==========================================
    # BARIS 1: GRAFIK ROC (Full Width)
    # ==========================================
    # Sedikit melebarkan figsize karena sekarang grafik mengambil lebar penuh
    fig_roc, ax = plt.subplots(figsize=(10, 5.5))
    fig_roc.patch.set_facecolor('#1A2D42')

    # Fill area under curve
    ax.fill_between(fpr_knn, tpr_knn, alpha=0.08, color=KNN_COLOR)
    ax.fill_between(fpr_nb,  tpr_nb,  alpha=0.08, color=NB_COLOR)

    ax.plot(fpr_knn, tpr_knn, color=KNN_COLOR, lw=2.5,
            label=f'KNN  (AUC = {auc_knn:.4f})')
    ax.plot(fpr_nb,  tpr_nb,  color=NB_COLOR,  lw=2.5,
            label=f'Naive Bayes  (AUC = {auc_nb:.4f})')
    ax.plot([0, 1], [0, 1], color='#3D5A73', lw=1.5,
            linestyle='--', label='Random Classifier (AUC = 0.5000)')

    # Optimal threshold point (Youden's J)
    opt_idx = np.argmax(tpr_knn - fpr_knn)
    ax.scatter(fpr_knn[opt_idx], tpr_knn[opt_idx],
               s=80, color=KNN_COLOR, zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate(f'  Optimal\n  thr={thresh_knn[opt_idx]:.2f}',
                xy=(fpr_knn[opt_idx], tpr_knn[opt_idx]-0.08), xycoords='data',
                fontsize=7.5, color=KNN_COLOR)

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.03])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=10, labelpad=8)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=10, labelpad=8)
    ax.set_title('Komparasi Kurva ROC', fontsize=12, fontweight='600',
                 color='#E8F4FC', pad=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', framealpha=0.2, labelcolor='white',
              fontsize=9, facecolor='#0D1B2A', edgecolor='#1E3A52')
    fig_roc.tight_layout()
    
    st.pyplot(fig_roc, use_container_width=True)

    # Memberikan sedikit jarak antara grafik dan cards
    st.write("") 

    # ==========================================
    # BARIS 2: 3 SUMMARY CARDS (Berdampingan)
    # ==========================================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card teal" style="padding:1.5rem; height:100%; border-radius:12px;">
            <div class="metric-label">AUC — KNN</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;
                        font-weight:700;color:#00B4D8;margin:0.3rem 0">{auc_knn:.4f}</div>
            <div style="font-size:0.78rem;color:#8B9EB7">
                {'🟢 Excellent (> 0.90)' if auc_knn > 0.9 else '🟡 Good (> 0.80)' if auc_knn > 0.8 else '🔴 Fair'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card amber" style="padding:1.5rem; height:100%; border-radius:12px;">
            <div class="metric-label">AUC — Naive Bayes</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;
                        font-weight:700;color:#F4A261;margin:0.3rem 0">{auc_nb:.4f}</div>
            <div style="font-size:0.78rem;color:#8B9EB7">
                {'🟢 Excellent (> 0.90)' if auc_nb > 0.9 else '🟡 Good (> 0.80)' if auc_nb > 0.8 else '🔴 Fair'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background:#1A2D42;border:1px solid #1E3A52;border-radius:12px;padding:1.5rem; height:100%;">
            <div class="metric-label">Selisih AUC</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;
                        font-weight:600;color:{'#2DD4BF' if auc_knn >= auc_nb else '#F87171'};margin:0.3rem 0">
                {abs(auc_knn - auc_nb):.4f}
            </div>
            <div style="font-size:0.78rem;color:#8B9EB7">
                {'KNN lebih unggul' if auc_knn >= auc_nb else 'Naive Bayes lebih unggul'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        

# ============================================================
# TAB 4: INTERPRETASI DATASET
# ============================================================

with tab4:

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📌</div>
        <div>
            <div class="section-title">Interpretasi Hasil Dataset</div>
            <div class="section-desc">
                Ringkasan Jumlah Data Depresi dan Nondepresi, Jumlah total depresi paling banyak
                serta tema isu yang ditemukan pada teks.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Data ringkasan yang berasal langsung dari CSV
    total_data = int(counts["total"])
    jumlah_depresi = int(counts["depresi"])
    jumlah_non = int(counts["non_depresi"])
    persen_depresi = float(counts["persen_depresi"])
    persen_non = float(counts["persen_non_depresi"])

    # ========================================================
    # BAGIAN ATAS: JUMLAH DATA DI KIRI, BULAN DI KANAN
    # ========================================================
    col_data, col_bulan = st.columns([1.08, 0.92], gap="large")

    with col_data:
        st.markdown("### Jumlah Data Depresi dan Non-Depresi")

        metric1, metric2 = st.columns(2)
        metric1.metric(
            "Data Depresi", f"{jumlah_depresi:,}",
            f"{persen_depresi:.2f}% dari total"
        )
        metric2.metric(
            "Data Non-Depresi", f"{jumlah_non:,}",
            f"{persen_non:.2f}% dari total"
        )
        st.metric("Total Data", f"{total_data:,}")

        fig, ax = plt.subplots(figsize=(6.2, 3.5))
        labels = ["Depresi", "Non-Depresi"]
        values = [jumlah_depresi, jumlah_non]
        bars = ax.bar(labels, values, color=[KNN_COLOR, NB_COLOR], width=0.55)
        ax.set_ylabel("Jumlah Unggahan")
        ax.set_title("Distribusi Label Dataset", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.18 if values else 1)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                f"{value:,}", ha="center", va="bottom", fontsize=10
            )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_bulan:
        st.markdown("### Bulan dengan Data Depresi Terbanyak")

        if not monthly_df.empty:
            top_dep_month = data_info["bulan_depresi_terbanyak"]
            st.markdown(
                f"""
                <div style="background:#1A2D42;border:1px solid #1E3A52;
                            border-left:4px solid #00B4D8;border-radius:12px;
                            padding:1.15rem 1.25rem;margin:0.35rem 0 1rem 0;">
                    <div style="font-size:0.75rem;color:#8B9EB7;letter-spacing:1px;
                                text-transform:uppercase;font-weight:600;">
                        Bulan dengan unggahan Depresi terbanyak
                    </div>
                    <div style="font-size:1.75rem;font-weight:700;color:#E8F4FC;
                                margin:0.35rem 0;">
                        {top_dep_month['bulan']}
                    </div>
                    <div style="font-size:1rem;color:#00B4D8;font-weight:600;">
                        {top_dep_month['jumlah']:,} unggahan Depresi
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

            month_plot = monthly_df.sort_values("bulan_kode").copy()
            fig, ax = plt.subplots(figsize=(6.2, 3.5))
            bars = ax.bar(
                month_plot["Bulan"], month_plot["Depresi"],
                color=KNN_COLOR, width=0.55
            )
            ax.set_ylabel("Jumlah Unggahan Depresi")
            ax.set_title("Distribusi Data Depresi per Bulan", pad=12)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", rotation=20)
            max_month_value = int(month_plot["Depresi"].max()) if len(month_plot) else 0
            ax.set_ylim(0, max_month_value * 1.18 if max_month_value else 1)
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_month_value * 0.025,
                    f"{int(bar.get_height()):,}",
                    ha="center", va="bottom", fontsize=9
                )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if len(monthly_df) == 1:
                st.caption(
                    "Catatan: dataset hanya mencakup satu bulan. Angka ini "
                    "menunjukkan jumlah pada bulan tersebut, bukan perbandingan "
                    "dengan bulan lain."
                )
            else:
                st.caption(
                    "Bulan terbanyak ditentukan dengan membandingkan jumlah "
                    "unggahan berlabel Depresi pada setiap bulan."
                )
        else:
            st.warning(
                "Informasi bulan belum dapat dihitung karena tanggal unggahan "
                "tidak tersedia atau tidak valid."
            )

    st.divider()

    # ========================================================
    # BAGIAN TENGAH: GRAFIK ISU
    # ========================================================
    st.markdown("###  Isu yang Banyak Ditemukan pada Unggahan Depresi")
    st.markdown(
        "Grafik berikut menunjukkan frekuensi tema berdasarkan pencocokan "
        "kata kunci pada teks unggahan pada data depresi."
    )

    if not issue_df.empty:
        issue_display = issue_df.sort_values(
            "Jumlah Unggahan", ascending=False
        ).copy()
        plot_df = issue_display.sort_values(
            "Jumlah Unggahan", ascending=True
        )

        fig, ax = plt.subplots(figsize=(11, 4.8))
        bars = ax.barh(
            plot_df["Tema Isu"],
            plot_df["Jumlah Unggahan"],
            color=NB_COLOR,
            height=0.68
        )
        ax.set_xlabel("Jumlah Unggahan")
        ax.set_title(
            "Frekuensi Tema Isu pada Unggahan Berlabel Depresi", pad=14
        )
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        max_issue_value = int(plot_df["Jumlah Unggahan"].max()) if len(plot_df) else 0
        ax.set_xlim(0, max_issue_value * 1.13 if max_issue_value else 1)
        for bar in bars:
            ax.text(
                bar.get_width() + max_issue_value * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}",
                va="center", fontsize=10
            )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        top_issue = data_info["isu_terbanyak"]
        st.markdown(
            f"""
            <div class="insight-box">
                <strong>Isu yang paling banyak ditemukan:</strong>
                {top_issue['tema']} dengan
                <strong>{top_issue['jumlah']:,} unggahan</strong>.
                Tema ini merupakan kategori yang paling sering cocok dengan
                kata kunci pada teks berlabel Depresi.
            </div>
            """, unsafe_allow_html=True
        )
        
    else:
        st.info("Belum terdapat data tema isu yang dapat ditampilkan.")
        top_issue = {"tema": "Belum teridentifikasi", "jumlah": 0}

    st.divider()

    # ========================================================
    # BAGIAN BAWAH: KESIMPULAN OTOMATIS
    # ========================================================
    st.markdown("### Kesimpulan Interpretasi Dataset")

    if len(monthly_df) == 1:
        ringkasan_bulan = (
            f"Data yang tersedia berasal dari {top_dep_month['bulan']}, "
            f"dengan {top_dep_month['jumlah']:,} unggahan berlabel Depresi. "
            "Karena dataset hanya mencakup satu bulan, belum dapat ditentukan "
            "apakah jumlah tersebut lebih tinggi dibandingkan bulan lainnya."
        )
    elif len(monthly_df) > 1:
        ringkasan_bulan = (
            f"Jumlah unggahan berlabel Depresi paling banyak ditemukan pada "
            f"{top_dep_month['bulan']}, yaitu {top_dep_month['jumlah']:,} unggahan."
        )
    else:
        ringkasan_bulan = "Distribusi bulanan belum dapat disimpulkan karena tanggal unggahan tidak tersedia."

    if top_issue["jumlah"] > 0:
        ringkasan_isu = (
            f"Tema yang paling sering ditemukan adalah {top_issue['tema']} "
            f"sebanyak {top_issue['jumlah']:,} unggahan."
        )
    else:
        ringkasan_isu = "Tema isu belum dapat disimpulkan dari kata kunci yang ditentukan."

    st.markdown(
        f"""
        <div style="background:#1A2D42;border:1px solid #1E3A52;
                    border-radius:12px;padding:1.25rem 1.5rem;
                    color:#C8D8E8;line-height:1.8;">
            Berdasarkan hasil pengolahan dataset, terdapat
            <strong style="color:#00B4D8">{total_data:,} unggahan</strong>,
            yang terdiri atas <strong>{jumlah_depresi:,} data Depresi
            ({persen_depresi:.2f}%)</strong> dan
            <strong>{jumlah_non:,} data Non-Depresi ({persen_non:.2f}%)</strong>.
            {ringkasan_bulan} {ringkasan_isu}
            
        </div>
        """, unsafe_allow_html=True
    )

    st.info(
        "Catatan metodologis: jumlah unggahan tidak menunjukkan jumlah individu. "
        "Analisis isu dilakukan melalui pencocokan kata kunci pada teks, sehingga "
        "hasilnya menggambarkan tema yang tertulis dan tidak membuktikan penyebab "
        "depresi. Seluruh hasil klasifikasi dan interpretasi bukan merupakan "
        "diagnosis medis atau klinis."
    )

# ============================================================
# TAB 5: DETEKSI TEKS MENGGUNAKAN MODEL KNN DAN NAIVE BAYES
# ============================================================

with tab5:
    import re
    import html

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🧪</div>
        <div>
            <div class="section-title">Deteksi Teks Depresi dan Non-Depresi</div>
            <div class="section-desc">
                Masukkan teks untuk memperoleh prediksi dari model KNN dan Naive Bayes
                yang telah dilatih.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box" style="margin-top:0;margin-bottom:1.2rem;">
        <strong>Petunjuk:</strong> Masukkan kalimat atau teks berbahasa Indonesia.
        Sistem akan melakukan preprocessing, mengubah teks menggunakan TF-IDF,
        kemudian menjalankan model yang dipilih.
    </div>
    """, unsafe_allow_html=True)

    # Preprocessing inferensi harus dibuat konsisten dengan preprocessing
    # yang digunakan ketika model dan TF-IDF dilatih di Google Colab.
    def preprocess_input_text(text_input):
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
                StopWordRemoverFactory
            )
        except ImportError as exc:
            raise ImportError(
                "Library Sastrawi belum terpasang. Jalankan: "
                "pip install Sastrawi"
            ) from exc

        text = html.unescape(str(text_input))
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenisasi sederhana berdasarkan spasi, sesuai representasi teks
        # yang akan diteruskan ke stopword removal dan stemming.
        tokens = text.split()

        
        # STOPWORD REMOVAL
        stop_factory = StopWordRemoverFactory()
        stopwords = set(stop_factory.get_stop_words())

        kata_penting = {
            'tidak', 'bukan', 'belum', 'jangan', 'tak', 'yang'
        }

        awalan_dilindungi = ('se', 'ke', 'me','tak','ber')

        def kata_dilindungi(kata):
            return (
                kata in kata_penting
                or kata.startswith(awalan_dilindungi)
            )

        tokens = [
            kata for kata in tokens
            if kata not in stopwords or kata_dilindungi(kata)
        ]

        # STEMMING
        factory_stemmer = StemmerFactory()
        stemmer = factory_stemmer.create_stemmer()

        hasil_stemming = []

        for word in tokens:
            if kata_dilindungi(word):
                hasil_stemming.append(word)
            else:
                hasil_stemming.append(stemmer.stem(word))

        tokens = hasil_stemming

        return ' '.join(tokens)


    def get_prediction_result(model, vectorized_text):
        prediction = int(model.predict(vectorized_text)[0])

        confidence = None
        probabilities = {}

        if hasattr(model, "predict_proba"):
            probability_array = model.predict_proba(vectorized_text)[0]
            model_classes = list(getattr(model, "classes_", [0, 1]))

            for class_value, probability in zip(model_classes, probability_array):
                class_int = int(class_value)
                class_name = "Depresi" if class_int == 1 else "Non-Depresi"
                probabilities[class_name] = float(probability)

            if prediction in [int(value) for value in model_classes]:
                index = [int(value) for value in model_classes].index(prediction)
                confidence = float(probability_array[index])

        label_name = "Depresi" if prediction == 1 else "Non-Depresi"

        return {
            "label": label_name,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        }

    def render_detection_card(model_name, result, accent):
        label_name = result["label"]
        is_depression = result["prediction"] == 1
        status_color = "#F87171" if is_depression else "#2DD4BF"
        status_bg = "rgba(248,113,113,0.10)" if is_depression else "rgba(45,212,191,0.10)"
        status_icon = "⚠️" if is_depression else "✓"
        confidence = result["confidence"]

        confidence_text = (
            f"{confidence * 100:.2f}%"
            if confidence is not None else "Tidak tersedia"
        )

        st.markdown(f"""
        <div style="
            background:#1A2D42;
            border:1px solid #1E3A52;
            border-top:3px solid {accent};
            border-radius:12px;
            padding:1.25rem 1.4rem;
            height:100%;">
            <div style="font-size:0.75rem;color:#8B9EB7;
                        letter-spacing:1.2px;text-transform:uppercase;
                        font-weight:600;">HASIL {model_name.upper()}</div>
            <div style="margin:0.8rem 0;padding:0.85rem 1rem;
                        background:{status_bg};border-radius:9px;">
                <div style="font-size:1.55rem;font-weight:700;color:{status_color};">
                    {status_icon} {label_name}
                </div>
            </div>
            <div style="font-size:0.82rem;color:#8B9EB7;">Probabilitas kelas prediksi</div>
            <div style="font-size:1.45rem;font-weight:700;color:{accent};
                        font-family:'JetBrains Mono',monospace;margin-top:0.2rem;">
                {confidence_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if result["probabilities"]:
            st.markdown("**Rincian probabilitas kelas**")
            for class_name in ["Depresi", "Non-Depresi"]:
                if class_name in result["probabilities"]:
                    value = result["probabilities"][class_name]
                    st.write(f"{class_name}: **{value * 100:.2f}%**")
                    st.progress(min(max(value, 0.0), 1.0))


    # Input dan pilihan model
    st.markdown("""
    <style>
    .stTextArea textarea {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    .stTextArea textarea::placeholder {
        color: #777777 !important;
        -webkit-text-fill-color: #777777 !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.form("form_deteksi_teks", clear_on_submit=False):
        pilihan_model = st.selectbox(
            "Pilih model klasifikasi",
            [
                "Bandingkan KNN dan Naive Bayes",
                "K-Nearest Neighbors (KNN)",
                "Naive Bayes"
            ],
            help="Pilih satu model atau bandingkan hasil kedua model."
        )

        teks_input = st.text_area(
            "Masukkan teks yang ingin dianalisis",
            placeholder=(
                "Contoh: Aku merasa lelah dengan hidup ini, "
                "semuanya terasa berat," "aku sangat bahagia"
            ),
            height=150,
            max_chars=2000,
            help="Masukkan teks berbahasa Indonesia, maksimal 2.000 karakter."
        )

        tombol_analisis = st.form_submit_button(
            "🔎 Analisis Teks",
            use_container_width=True,
            type="primary"
        )

    if tombol_analisis:
        if not teks_input.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
        else:
            try:
                # Pastikan seluruh artefak model tersedia.
                required_artifacts = {
                    "TF-IDF Vectorizer": FILE_VECTORIZER,
                    "Model KNN": FILE_MODEL_KNN,
                    "Model Naive Bayes": FILE_MODEL_NB
                }

                if pilihan_model == "K-Nearest Neighbors (KNN)":
                    artifacts_to_check = {
                        "TF-IDF Vectorizer": FILE_VECTORIZER,
                        "Model KNN": FILE_MODEL_KNN
                    }
                elif pilihan_model == "Naive Bayes":
                    artifacts_to_check = {
                        "TF-IDF Vectorizer": FILE_VECTORIZER,
                        "Model Naive Bayes": FILE_MODEL_NB
                    }
                else:
                    artifacts_to_check = required_artifacts

                missing_artifacts = [
                    name for name, path in artifacts_to_check.items()
                    if not path.exists()
                ]

                if missing_artifacts:
                    st.error(
                        "File pendukung berikut belum ditemukan: "
                        + ", ".join(missing_artifacts)
                        + ". Letakkan file .pkl tersebut satu folder dengan main.py."
                    )
                else:
                    with st.spinner("Memproses teks dan menjalankan model..."):
                        teks_bersih = preprocess_input_text(teks_input)

                        if not teks_bersih:
                            st.warning(
                                "Teks tidak memiliki kata yang tersisa setelah "
                                "preprocessing. Silakan masukkan kalimat lain."
                            )
                        else:
                            vectorizer = joblib.load(FILE_VECTORIZER)

                            # Validasi bahwa vectorizer benar-benar sudah di-fit
                            # dan jumlah fiturnya cocok dengan model yang digunakan.
                            if not hasattr(vectorizer, "idf_"):
                                raise ValueError(
                                    f"Vectorizer pada {FILE_VECTORIZER.name} "
                                    "belum memiliki atribut idf_. Pastikan file "
                                    "yang dipakai adalah TF-IDF hasil fit dari "
                                    "data training."
                                )

                            teks_tfidf = vectorizer.transform([teks_bersih])

                            if pilihan_model in [
                                "Bandingkan KNN dan Naive Bayes",
                                "K-Nearest Neighbors (KNN)"
                            ]:
                                model_check_knn = joblib.load(FILE_MODEL_KNN)
                                if teks_tfidf.shape[1] != model_check_knn.n_features_in_:
                                    raise ValueError(
                                        "Jumlah fitur TF-IDF tidak cocok dengan KNN: "
                                        f"vectorizer={teks_tfidf.shape[1]}, "
                                        f"KNN={model_check_knn.n_features_in_}."
                                    )

                            if pilihan_model in [
                                "Bandingkan KNN dan Naive Bayes",
                                "Naive Bayes"
                            ]:
                                model_check_nb = joblib.load(FILE_MODEL_NB)
                                if teks_tfidf.shape[1] != model_check_nb.n_features_in_:
                                    raise ValueError(
                                        "Jumlah fitur TF-IDF tidak cocok dengan Naive Bayes: "
                                        f"vectorizer={teks_tfidf.shape[1]}, "
                                        f"Naive Bayes={model_check_nb.n_features_in_}."
                                    )

                            hasil_prediksi = {}

                            if pilihan_model in [
                                "Bandingkan KNN dan Naive Bayes",
                                "K-Nearest Neighbors (KNN)"
                            ]:
                                model_knn = joblib.load(FILE_MODEL_KNN)
                                hasil_prediksi["KNN"] = get_prediction_result(
                                    model_knn, teks_tfidf
                                )

                            if pilihan_model in [
                                "Bandingkan KNN dan Naive Bayes",
                                "Naive Bayes"
                            ]:
                                model_nb = joblib.load(FILE_MODEL_NB)
                                hasil_prediksi["Naive Bayes"] = get_prediction_result(
                                    model_nb, teks_tfidf
                                )

                    if teks_bersih:
                        st.divider()
                        st.markdown("### Hasil Deteksi")

                        col_input, col_clean = st.columns([1, 1], gap="large")
                        with col_input:
                            st.markdown("**Teks yang dimasukkan**")
                            st.info(teks_input)
                        with col_clean:
                            st.markdown("**Hasil preprocessing**")
                            st.code(teks_bersih, language="text")

                        st.markdown("### Perbandingan Hasil Model")

                        result_columns = st.columns(
                            len(hasil_prediksi), gap="large"
                        )

                        for column, (model_name, result) in zip(
                            result_columns, hasil_prediksi.items()
                        ):
                            with column:
                                accent = (
                                    KNN_COLOR if model_name == "KNN" else NB_COLOR
                                )
                                render_detection_card(
                                    model_name, result, accent
                                )

                        # Interpretasi hasil prediksi
                        labels_detected = [
                            result["label"]
                            for result in hasil_prediksi.values()
                        ]

                        st.markdown("### Interpretasi Hasil")

                        if len(set(labels_detected)) == 1:
                            final_label = labels_detected[0]
                            if final_label == "Depresi":
                                st.warning(
                                    "Kedua model mengklasifikasikan teks ini "
                                    "ke dalam kategori **Depresi** berdasarkan "
                                    "pola teks yang dipelajari selama pelatihan."
                                )
                            else:
                                st.success(
                                    "Kedua model mengklasifikasikan teks ini "
                                    "ke dalam kategori **Non-Depresi** berdasarkan "
                                    "pola teks yang dipelajari selama pelatihan."
                                )
                        else:
                            st.warning(
                                "Kedua model memberikan hasil yang berbeda. "
                                "KNN dan Naive Bayes memiliki karakteristik "
                                "klasifikasi yang berbeda, sehingga hasil ini "
                                "perlu ditafsirkan secara hati-hati."
                            )

                        st.markdown("### Informasi Lanjutan")
                        info_col1, info_col2 = st.columns(2)

                        with info_col1:
                            st.markdown("**Tahapan analisis**")
                            st.write("1. Cleaning dan case folding")
                            st.write("2. Tokenizing")
                            st.write("3. Stopword removal")
                            st.write("4. Stemming")
                            st.write("5. Transformasi TF-IDF")
                            st.write("6. Prediksi menggunakan model terpilih")

                        with info_col2:
                            st.markdown("**Ringkasan prediksi**")
                            for model_name, result in hasil_prediksi.items():
                                st.write(
                                    f"- **{model_name}:** {result['label']}"
                                )
                                if result["confidence"] is not None:
                                    st.write(
                                        f"  Probabilitas kelas prediksi: "
                                        f"{result['confidence'] * 100:.2f}%"
                                    )

                        st.markdown("""
                        <div class="insight-box">
                            <strong>Catatan penting:</strong>
                            Hasil ini merupakan klasifikasi teks berdasarkan pola
                            yang dipelajari model dari dataset penelitian. Nilai
                            probabilitas bukan ukuran pasti kondisi psikologis
                            seseorang. Fitur ini tidak digunakan untuk menetapkan
                            diagnosis depresi. Jika mengalami kesulitan emosional
                            atau membutuhkan bantuan, pertimbangkan untuk
                            berkonsultasi dengan tenaga kesehatan mental profesional.
                        </div>
                        """, unsafe_allow_html=True)

            except ImportError as e:
                st.error(str(e))
            except Exception as e:
                st.error(
                    "Terjadi kesalahan saat melakukan deteksi teks. "
                    f"Detail: {e}"
                )
                st.info(
                    "Pastikan model, TF-IDF vectorizer, versi library, serta "
                    "tahapan preprocessing sama dengan yang digunakan saat "
                    "pelatihan di Google Colab."
                )

    else:
        st.markdown("""
        <div style="background:#1A2D42;border:1px dashed #31506B;
                    border-radius:12px;padding:1.3rem;margin-top:1rem;
                    color:#8B9EB7;text-align:center;">
            <div style="font-size:1.8rem;">📝</div>
            <strong style="color:#C8D8E8;">Belum ada teks yang dianalisis</strong>
            <div style="font-size:0.85rem;margin-top:0.35rem;">
                Masukkan teks pada kolom di atas, pilih model, lalu klik Analisis Teks.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Catatan: fitur ini merupakan alat bantu klasifikasi teks untuk keperluan "
        "penelitian, bukan alat diagnosis medis atau klinis."
    )
