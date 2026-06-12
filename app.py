"""
DS4A Final Project — Fundación Amanecer Loan Payment Predictor
Correlation One DS4A Program · Team 84

Gradio demo app. Uses synthetic data that mirrors the real feature schema
(the original dataset is proprietary to Fundación Amanecer).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

# ── Palette ────────────────────────────────────────────────────────────────────
GREEN_DARK   = "#1a6b2e"
GREEN_MID    = "#2ecc71"
GREEN_LIGHT  = "#a9dfbf"
BG_COLOR     = "#f8fdf9"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   "#cccccc",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
})

# ── Categorical option lists ───────────────────────────────────────────────────
SECTORS    = ["urbano", "rural"]
REGIONS    = ["bogotá", "región sur", "región norte", "región oriente", "región occidente"]
ACTIVITIES = ["comercio", "agropecuario", "servicios", "industria", "construcción"]
HOUSING    = ["propia", "arrendada", "familiar", "otro"]
CIVIL      = ["casado", "soltero", "unión libre", "divorciado", "viudo"]
GENDERS    = ["femenino", "masculino"]
EDUCATION  = ["primaria", "secundaria", "técnico", "universitario", "ninguno"]
YES_NO     = ["si", "no"]

CATEGORICAL = [
    "sector", "regional", "actividad_econ", "vivienda",
    "estado_civil", "genero", "educ", "mujer_cabeza", "responsable_hogar"
]
NUMERICAL = ["edad", "monto", "cuotas", "tasa"]


# ── Synthetic dataset (mirrors real feature schema) ────────────────────────────
def _build_dataset(n: int = 6_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    edad   = rng.normal(38, 12, n).clip(18, 75).astype(int)
    monto  = rng.lognormal(15.0, 1.1, n).clip(500_000, 50_000_000)
    cuotas = rng.choice([6, 12, 18, 24, 36, 48, 60], n)
    tasa   = rng.uniform(0.010, 0.045, n)

    sector        = rng.choice(SECTORS, n, p=[0.55, 0.45])
    regional      = rng.choice(REGIONS, n)
    actividad_econ = rng.choice(ACTIVITIES, n, p=[0.35, 0.25, 0.20, 0.12, 0.08])
    vivienda      = rng.choice(HOUSING, n, p=[0.45, 0.30, 0.20, 0.05])
    estado_civil  = rng.choice(CIVIL,   n, p=[0.40, 0.25, 0.25, 0.05, 0.05])
    genero        = rng.choice(GENDERS, n, p=[0.58, 0.42])
    educ          = rng.choice(EDUCATION, n, p=[0.20, 0.35, 0.25, 0.15, 0.05])
    mujer_cabeza  = rng.choice(YES_NO,  n, p=[0.35, 0.65])
    responsable   = rng.choice(YES_NO,  n, p=[0.72, 0.28])

    # Realistic risk signal
    risk = (
        (edad < 25).astype(float)             * 0.30
        + (monto > 20_000_000).astype(float)  * 0.20
        + (cuotas > 36).astype(float)         * 0.15
        + (tasa > 0.035).astype(float)        * 0.10
        + (educ == "ninguno").astype(float)   * 0.15
        + (vivienda == "arrendada").astype(float) * 0.10
        + rng.uniform(0, 0.30, n)
    )
    classification = (risk > 0.50).astype(int)

    return pd.DataFrame({
        "edad": edad, "monto": monto, "cuotas": cuotas, "tasa": tasa,
        "sector": sector, "regional": regional, "actividad_econ": actividad_econ,
        "vivienda": vivienda, "estado_civil": estado_civil, "genero": genero,
        "educ": educ, "mujer_cabeza": mujer_cabeza, "responsable_hogar": responsable,
        "classification": classification,
    })


df = _build_dataset()

scaler  = StandardScaler()
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

X_num = scaler.fit_transform(df[NUMERICAL])
X_cat = encoder.fit_transform(df[CATEGORICAL])
X = np.concatenate([X_num, X_cat], axis=1)
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
ACCURACY = accuracy_score(y_test, y_pred)
AUC      = roc_auc_score(y_test, y_proba)

cat_feature_names   = encoder.get_feature_names_out(CATEGORICAL).tolist()
ALL_FEATURE_NAMES   = NUMERICAL + cat_feature_names


# ── Chart generators ──────────────────────────────────────────────────────────
def _confusion_matrix_fig() -> plt.Figure:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens", ax=ax,
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(
        f"Confusion Matrix\nAccuracy {ACCURACY:.2%}  ·  AUC {AUC:.3f}",
        fontsize=12, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    return fig


def _feature_importance_fig() -> plt.Figure:
    importances = model.feature_importances_
    top_idx  = np.argsort(importances)[-15:]
    top_names = [ALL_FEATURE_NAMES[i] for i in top_idx]
    top_vals  = importances[top_idx]

    palette = [GREEN_DARK if v >= 0.08 else GREEN_MID if v >= 0.04 else GREEN_LIGHT
               for v in top_vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(top_names, top_vals, color=palette, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8, color="#333")
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=10)
    ax.set_title("Top 15 Feature Importances — Random Forest", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


def _distribution_fig() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loan amount by class
    for cls, label, color in [(0, "Low Risk", GREEN_MID), (1, "High Risk", "#e74c3c")]:
        subset = df[df["classification"] == cls]["monto"] / 1_000_000
        axes[0].hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="white")
    axes[0].set_xlabel("Loan Amount (M COP)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("Loan Amount Distribution by Risk", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Age distribution by class
    for cls, label, color in [(0, "Low Risk", GREEN_MID), (1, "High Risk", "#e74c3c")]:
        subset = df[df["classification"] == cls]["edad"]
        axes[1].hist(subset, bins=25, alpha=0.6, label=label, color=color, edgecolor="white")
    axes[1].set_xlabel("Age (years)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("Age Distribution by Risk", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    return fig


# ── Prediction function ────────────────────────────────────────────────────────
def predict_risk(
    edad, monto, cuotas, tasa,
    sector, regional, actividad, vivienda,
    estado_civil, genero, educ,
    mujer_cabeza, responsable_hogar,
):
    num_arr = scaler.transform([[edad, monto, cuotas, tasa]])
    cat_arr = encoder.transform(
        [[sector, regional, actividad, vivienda,
          estado_civil, genero, educ, mujer_cabeza, responsable_hogar]]
    )
    features = np.concatenate([num_arr, cat_arr], axis=1)

    pred  = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    risk_pct = proba[1] * 100

    if pred == 1:
        verdict = f"⚠️ HIGH RISK — Default probability: {risk_pct:.1f}%"
    else:
        verdict = f"✅ LOW RISK — Default probability: {risk_pct:.1f}%"

    confidence = f"Model confidence: {max(proba)*100:.1f}%"

    # Mini risk bar (text-based for simplicity)
    bars_filled = int(risk_pct / 10)
    risk_bar = "█" * bars_filled + "░" * (10 - bars_filled)
    risk_display = f"[{risk_bar}] {risk_pct:.1f}%"

    return verdict, confidence, risk_display


# ── Gradio UI ──────────────────────────────────────────────────────────────────
DESCRIPTION = """
# 💳 DS4A — Loan Payment Risk Predictor
### Fundación Amanecer · Correlation One DS4A Program · Team 84

Predict whether a microcredit borrower is **at risk of default** based on applicant 
and loan characteristics. Powered by a **Random Forest** model trained on portfolio data.

> ℹ️ This demo uses **synthetic data** that mirrors the real feature schema.  
> The original dataset is proprietary to Fundación Amanecer.
"""

ABOUT_TEXT = """
## About This Project

This tool was built as the **Final Project for the DS4A (Data Science for All) program** 
by Correlation One, developed in collaboration with **Fundación Amanecer** — a Colombian 
microfinance institution serving underserved communities.

### 🎯 Problem Statement
Fundación Amanecer grants microcredits to low-income entrepreneurs across Colombia.  
The team built ML models to predict loan payment risk, helping the foundation make 
data-driven lending decisions and reduce default rates.

### 🔬 Features
| Category | Features |
|----------|----------|
| **Numerical** | Age, Loan Amount (COP), Installments, Interest Rate |
| **Socioeconomic** | Gender, Education, Marital Status, Housing Type |
| **Geographic** | Region, Urban/Rural Sector |
| **Household** | Female Head of Household, Household Provider |
| **Economic** | Primary Economic Activity |

### 🤖 Model
- **Algorithm:** Random Forest Classifier (300 trees)  
- **Validation:** Stratified 80/20 train-test split  
- **Metrics:** Accuracy, AUC-ROC

### 👥 Team 84 · DS4A Colombia · 2022
"""

with gr.Blocks(
    title="DS4A — Loan Risk Predictor",
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="gray"),
    css=".gradio-container { max-width: 1100px !important; }",
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1: Prediction ──────────────────────────────────────────────────
        with gr.Tab("🔮 Predict Risk"):
            gr.Markdown("### Enter borrower & loan details, then click **Predict**.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**📊 Loan Details**")
                    edad_in   = gr.Slider(18, 75, value=35, step=1, label="Age (years)")
                    monto_in  = gr.Slider(500_000, 50_000_000, value=5_000_000,
                                          step=100_000, label="Loan Amount (COP)")
                    cuotas_in = gr.Dropdown([6, 12, 18, 24, 36, 48, 60],
                                            value=12, label="Number of Installments")
                    tasa_in   = gr.Slider(0.01, 0.045, value=0.022,
                                          step=0.001, label="Monthly Interest Rate (e.g. 0.022 = 2.2%)")

                with gr.Column(scale=1):
                    gr.Markdown("**📍 Location & Sector**")
                    sector_in   = gr.Dropdown(SECTORS,    value="urbano",    label="Sector")
                    regional_in = gr.Dropdown(REGIONS,    value="bogotá",    label="Region")
                    actividad_in = gr.Dropdown(ACTIVITIES, value="comercio", label="Economic Activity")
                    vivienda_in  = gr.Dropdown(HOUSING,   value="propia",    label="Housing Type")

                with gr.Column(scale=1):
                    gr.Markdown("**👤 Borrower Profile**")
                    civil_in    = gr.Dropdown(CIVIL,     value="casado",     label="Marital Status")
                    genero_in   = gr.Dropdown(GENDERS,   value="femenino",   label="Gender")
                    educ_in     = gr.Dropdown(EDUCATION, value="secundaria", label="Education Level")
                    mujer_in    = gr.Dropdown(YES_NO,    value="no",         label="Female Head of Household")
                    resp_in     = gr.Dropdown(YES_NO,    value="si",         label="Household Provider")

            predict_btn = gr.Button("🔍 Predict Payment Risk", variant="primary", size="lg")

            with gr.Row():
                verdict_out    = gr.Textbox(label="Risk Assessment", interactive=False, scale=3)
                confidence_out = gr.Textbox(label="Confidence",      interactive=False, scale=1)
                bar_out        = gr.Textbox(label="Risk Level",       interactive=False, scale=2)

            predict_btn.click(
                predict_risk,
                inputs=[
                    edad_in, monto_in, cuotas_in, tasa_in,
                    sector_in, regional_in, actividad_in, vivienda_in,
                    civil_in, genero_in, educ_in, mujer_in, resp_in,
                ],
                outputs=[verdict_out, confidence_out, bar_out],
            )

        # ── Tab 2: Model Performance ───────────────────────────────────────────
        with gr.Tab("📊 Model Performance"):
            gr.Markdown(
                f"### Random Forest — Test Set Results\n"
                f"**Accuracy:** `{ACCURACY:.2%}`  ·  "
                f"**AUC-ROC:** `{AUC:.3f}`  ·  "
                f"**Test samples:** `{len(X_test):,}`"
            )
            with gr.Row():
                cm_plot = gr.Plot(label="Confusion Matrix")
                fi_plot = gr.Plot(label="Feature Importances")
            dist_plot = gr.Plot(label="Data Distributions")

            load_btn = gr.Button("📈 Load Charts", variant="secondary")
            load_btn.click(
                lambda: (
                    _confusion_matrix_fig(),
                    _feature_importance_fig(),
                    _distribution_fig(),
                ),
                outputs=[cm_plot, fi_plot, dist_plot],
            )

        # ── Tab 3: About ───────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT_TEXT)

demo.launch()
