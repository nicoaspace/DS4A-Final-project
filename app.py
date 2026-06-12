"""
DS4A Final Project — Fundación Amanecer Loan Payment Risk Predictor
Correlation One DS4A Program · Team 84

Professional microfinance risk assessment tool.
Uses synthetic data calibrated to Colombian microfinance reality.
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import math

# ── Visual theme ────────────────────────────────────────────────────────────────
G_DARK  = "#1a6b2e"
G_MID   = "#2ecc71"
G_LIGHT = "#a9dfbf"
RED     = "#e74c3c"
BG      = "#f8fdf9"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#cccccc",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

# ── Feature schema ──────────────────────────────────────────────────────────────
SECTORS      = ["urbano", "rural"]
REGIONS      = ["bogotá d.c.", "región sur", "región norte",
                "región oriente", "región occidente"]
ACTIVITIES   = ["comercio minorista", "agropecuario", "servicios personales",
                "industria artesanal", "construcción", "transporte", "otro"]
HOUSING      = ["propia sin deuda", "propia con hipoteca", "arrendada", "familiar"]
CIVIL        = ["casado/a", "soltero/a", "unión libre", "divorciado/a", "viudo/a"]
GENDERS      = ["femenino", "masculino"]
EDUCATION    = ["ninguno", "primaria incompleta", "primaria completa",
                "secundaria", "técnico/tecnólogo", "universitario"]
YES_NO       = ["si", "no"]
HISTORIAL    = ["muy bueno", "bueno", "regular", "sin historial", "malo"]

CATEGORICAL = [
    "sector", "regional", "actividad_econ", "vivienda",
    "estado_civil", "genero", "educ", "mujer_cabeza",
    "responsable_hogar", "historial_crediticio",
]
NUMERICAL = [
    "edad", "ingreso_mensual", "monto", "cuotas", "tasa",
    "relacion_cuota_ingreso", "numero_dependientes", "antiguedad_laboral",
]


# ── PMT — monthly installment formula ──────────────────────────────────────────
def calcular_cuota(monto: float, tasa_mensual: float, cuotas: int) -> float:
    """French amortization (equal installment) formula."""
    if tasa_mensual <= 0:
        return monto / cuotas
    r = tasa_mensual
    n = cuotas
    return monto * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def calcular_rci(monto: float, tasa: float, cuotas: int, ingreso: float) -> float:
    cuota = calcular_cuota(monto, tasa, cuotas)
    return cuota / ingreso if ingreso > 0 else 1.0


# ── Synthetic dataset calibrated to Colombian microfinance ─────────────────────
def _build_dataset(n: int = 7_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    # Demographics — Colombian microfinance population profile
    edad     = rng.normal(39, 11, n).clip(20, 70).astype(int)
    genero   = rng.choice(GENDERS, n, p=[0.62, 0.38])   # majority female in microfinance
    educ     = rng.choice(EDUCATION, n, p=[0.04, 0.14, 0.18, 0.38, 0.19, 0.07])
    estado_civil  = rng.choice(CIVIL, n, p=[0.35, 0.22, 0.30, 0.07, 0.06])
    mujer_cabeza  = rng.choice(YES_NO, n, p=[0.42, 0.58])
    resp_hogar    = rng.choice(YES_NO, n, p=[0.68, 0.32])
    num_dep       = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.14, 0.22, 0.28, 0.20, 0.10, 0.06])

    # Income — COP/month, realistic for Colombian microfinance clients
    # Formal minimum wage 2024: ~1.3M; typical clients: 800K – 4M
    educ_income_mult = {"ninguno": 0.70, "primaria incompleta": 0.78,
                        "primaria completa": 0.88, "secundaria": 1.00,
                        "técnico/tecnólogo": 1.30, "universitario": 1.70}
    base_income = np.array([educ_income_mult[e] for e in educ]) * 1_400_000
    ingreso_mensual = (rng.lognormal(0, 0.45, n) * base_income).clip(700_000, 12_000_000)

    # Credit history — correlated with income and education
    historial_weights = []
    for i in range(n):
        if ingreso_mensual[i] > 3_000_000:
            w = [0.20, 0.40, 0.20, 0.12, 0.08]
        elif ingreso_mensual[i] > 1_500_000:
            w = [0.12, 0.33, 0.25, 0.20, 0.10]
        else:
            w = [0.06, 0.22, 0.25, 0.30, 0.17]
        historial_weights.append(w)
    historial_crediticio = np.array([
        rng.choice(HISTORIAL, p=hw) for hw in historial_weights
    ])

    # Work seniority — years at current job
    antiguedad_laboral = rng.exponential(4, n).clip(0, 25).astype(int)

    # Geography
    sector   = rng.choice(SECTORS, n, p=[0.60, 0.40])
    regional = rng.choice(REGIONS, n)
    actividad_econ = rng.choice(ACTIVITIES, n,
                                p=[0.32, 0.22, 0.18, 0.12, 0.07, 0.06, 0.03])
    vivienda = rng.choice(HOUSING, n, p=[0.38, 0.10, 0.32, 0.20])

    # Loan amounts — realistic for microfinance (500K – 30M COP)
    # Larger loans correlate with higher income
    monto_base   = ingreso_mensual * rng.uniform(1.5, 10, n)
    monto        = monto_base.clip(500_000, 30_000_000)
    cuotas       = rng.choice([6, 9, 12, 18, 24, 36, 48], n,
                               p=[0.08, 0.10, 0.28, 0.25, 0.18, 0.08, 0.03])
    tasa         = rng.uniform(0.015, 0.042, n)   # 1.5% – 4.2% monthly, NAMV

    # ── Key risk metric: RCI (Relación Cuota/Ingreso) ──────────────────────────
    cuota_mensual = np.array([calcular_cuota(m, t, int(c))
                               for m, t, c in zip(monto, tasa, cuotas)])
    rci           = cuota_mensual / ingreso_mensual

    # ── Risk signal: professionally calibrated ──────────────────────────────────
    # RCI > 35% is the standard microfinance alert threshold
    hist_score = {
        "muy bueno": -0.40, "bueno": -0.20, "regular": 0.15,
        "sin historial": 0.25, "malo": 0.55,
    }
    risk = (
        # Core financial stress
        np.clip((rci - 0.30) * 3.0, -0.5, 1.5)         # RCI is the #1 predictor
        + (monto / (ingreso_mensual * 12 + 1e-9) - 0.5).clip(-0.3, 0.8) * 0.40
        # Credit history
        + np.array([hist_score[h] for h in historial_crediticio])
        # Demographics
        + np.where(edad < 25, 0.25, 0)
        + np.where(edad > 60, 0.10, 0)
        + (num_dep - 2).clip(0, 4) * 0.05
        + np.where(antiguedad_laboral < 1, 0.20, 0)
        + np.where(antiguedad_laboral > 5, -0.10, 0)
        # Housing & education
        + np.where(vivienda == "arrendada", 0.12, 0)
        + np.where(np.isin(educ, ["ninguno", "primaria incompleta"]), 0.10, 0)
        # Noise
        + rng.normal(0, 0.18, n)
    )

    classification = (risk > 0.50).astype(int)

    return pd.DataFrame({
        "edad": edad, "ingreso_mensual": ingreso_mensual,
        "monto": monto, "cuotas": cuotas, "tasa": tasa,
        "relacion_cuota_ingreso": rci,
        "numero_dependientes": num_dep,
        "antiguedad_laboral": antiguedad_laboral,
        "sector": sector, "regional": regional,
        "actividad_econ": actividad_econ, "vivienda": vivienda,
        "estado_civil": estado_civil, "genero": genero,
        "educ": educ, "mujer_cabeza": mujer_cabeza,
        "responsable_hogar": resp_hogar,
        "historial_crediticio": historial_crediticio,
        "classification": classification,
    })


# ── Build & train ───────────────────────────────────────────────────────────────
print("Building dataset and training model…")
df = _build_dataset()

X = df[NUMERICAL + CATEGORICAL]
y = df["classification"]

preproc = ColumnTransformer([
    ("num", StandardScaler(), NUMERICAL),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
     CATEGORICAL),
])

pipeline = Pipeline([
    ("prep", preproc),
    ("clf",  GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.10,
        subsample=0.8, min_samples_leaf=15, random_state=42,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

METRICS = {
    "accuracy":  accuracy_score(y_test, y_pred),
    "auc":       roc_auc_score(y_test, y_proba),
    "precision": precision_score(y_test, y_pred),
    "recall":    recall_score(y_test, y_pred),
    "f1":        f1_score(y_test, y_pred),
}
print(f"  Accuracy {METRICS['accuracy']:.3f}  AUC {METRICS['auc']:.3f}  "
      f"F1 {METRICS['f1']:.3f}")

# ── Feature importances from the GBM ───────────────────────────────────────────
cat_names = (preproc.named_transformers_["cat"]
             .get_feature_names_out(CATEGORICAL).tolist())
ALL_NAMES = NUMERICAL + cat_names


# ── Chart generators ──────────────────────────────────────────────────────────
def _cm_fig() -> plt.Figure:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                xticklabels=["Bajo Riesgo", "Alto Riesgo"],
                yticklabels=["Bajo Riesgo", "Alto Riesgo"],
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicho", fontsize=11)
    ax.set_ylabel("Real",     fontsize=11)
    ax.set_title(
        f"Matriz de Confusión\n"
        f"Accuracy {METRICS['accuracy']:.2%}  ·  AUC {METRICS['auc']:.3f}  ·  F1 {METRICS['f1']:.2%}",
        fontsize=11, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    return fig


def _fi_fig() -> plt.Figure:
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    top_idx  = np.argsort(importances)[-18:]
    top_names = [ALL_NAMES[i] for i in top_idx]
    top_vals  = importances[top_idx]
    palette   = [G_DARK if v >= 0.10 else G_MID if v >= 0.04 else G_LIGHT
                 for v in top_vals]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.barh(top_names, top_vals, color=palette, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8, color="#333")
    ax.set_xlabel("Importancia (GBM)", fontsize=10)
    ax.set_title("Top 18 Features — Gradient Boosting", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


def _rci_fig() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # RCI distribution by class
    for cls, label, color in [(0, "Bajo Riesgo", G_MID), (1, "Alto Riesgo", RED)]:
        s = df[df["classification"] == cls]["relacion_cuota_ingreso"]
        axes[0].hist(s * 100, bins=35, alpha=0.65, label=label, color=color, edgecolor="white")
    axes[0].axvline(35, color="orange", ls="--", lw=1.8, label="Umbral 35%")
    axes[0].axvline(50, color=RED,      ls="--", lw=1.8, label="Umbral crítico 50%")
    axes[0].set_xlabel("Relación Cuota/Ingreso (%)", fontsize=10)
    axes[0].set_ylabel("Frecuencia", fontsize=10)
    axes[0].set_title("RCI por Clase de Riesgo", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)

    # Default rate by credit history
    dr = df.groupby("historial_crediticio")["classification"].mean().sort_values() * 100
    colors = [RED if v > 40 else G_MID if v < 25 else "orange" for v in dr.values]
    axes[1].barh(dr.index, dr.values, color=colors, edgecolor="white")
    axes[1].bar_label(axes[1].containers[0], fmt="%.1f%%", padding=4, fontsize=9)
    axes[1].set_xlabel("Tasa de Mora (%)", fontsize=10)
    axes[1].set_title("Tasa de Mora por Historial Crediticio", fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


def _income_fig() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Default rate by income decile
    df["ingreso_decil"] = pd.qcut(df["ingreso_mensual"], 5,
                                   labels=["<Q1", "Q1-Q2", "Q2-Q3", "Q3-Q4", ">Q4"])
    dr_inc = df.groupby("ingreso_decil", observed=True)["classification"].mean() * 100
    colors = [RED if v > 40 else G_MID if v < 25 else "orange" for v in dr_inc.values]
    axes[0].bar(dr_inc.index.astype(str), dr_inc.values, color=colors, edgecolor="white")
    axes[0].bar_label(axes[0].containers[0], fmt="%.1f%%", padding=3, fontsize=9)
    axes[0].set_xlabel("Quintil de Ingreso", fontsize=10)
    axes[0].set_ylabel("Tasa de Mora (%)", fontsize=10)
    axes[0].set_title("Mora por Quintil de Ingreso", fontsize=11, fontweight="bold")

    # Loan amount vs income scatter (sample)
    sample = df.sample(800, random_state=1)
    for cls, label, color, alpha in [(0, "Bajo Riesgo", G_MID, 0.4), (1, "Alto Riesgo", RED, 0.5)]:
        s = sample[sample["classification"] == cls]
        axes[1].scatter(s["ingreso_mensual"] / 1e6, s["monto"] / 1e6,
                        c=color, alpha=alpha, s=18, label=label)
    axes[1].axline((0, 0), slope=6,  color="orange", ls="--", lw=1.2, label="6× ingreso")
    axes[1].axline((0, 0), slope=12, color=RED,      ls="--", lw=1.2, label="12× ingreso")
    axes[1].set_xlabel("Ingreso Mensual (M COP)", fontsize=10)
    axes[1].set_ylabel("Monto del Crédito (M COP)", fontsize=10)
    axes[1].set_title("Monto vs Ingreso", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 8); axes[1].set_ylim(0, 32)

    plt.tight_layout()
    return fig


# ── Risk tier config (by language) ─────────────────────────────────────────────
TIERS = {
    "en": [
        (20,  "#16a34a", "#dcfce7", "#15803d", "LOW RISK",
         "Strong profile", "Approval recommended"),
        (40,  "#ca8a04", "#fef9c3", "#a16207", "MODERATE RISK",
         "Review conditions", "Consider collateral or guarantor"),
        (60,  "#ea580c", "#ffedd5", "#c2410c", "HIGH RISK",
         "Elevated risk", "Requires co-signer or real guarantee"),
        (101, "#dc2626", "#fee2e2", "#b91c1c", "VERY HIGH RISK",
         "High-risk profile", "Disbursement not recommended"),
    ],
    "es": [
        (20,  "#16a34a", "#dcfce7", "#15803d", "RIESGO BAJO",
         "Perfil sólido", "Aprobación recomendada"),
        (40,  "#ca8a04", "#fef9c3", "#a16207", "RIESGO MODERADO",
         "Revisar condiciones", "Solicitar aval o garantía adicional"),
        (60,  "#ea580c", "#ffedd5", "#c2410c", "RIESGO ALTO",
         "Riesgo elevado", "Requiere codeudor o garantía real"),
        (101, "#dc2626", "#fee2e2", "#b91c1c", "RIESGO MUY ALTO",
         "Perfil de alto riesgo", "No se recomienda el desembolso"),
    ],
}


def predict_risk(
    edad, ingreso_mensual, monto, cuotas, tasa,
    num_dependientes, antiguedad_laboral,
    sector, regional, actividad, vivienda,
    estado_civil, genero, educ,
    mujer_cabeza, responsable_hogar, historial_crediticio,
    lang="en",
):
    cuota_est = calcular_cuota(monto, tasa, int(cuotas))
    rci       = cuota_est / ingreso_mensual
    capacidad = ingreso_mensual - cuota_est

    row = pd.DataFrame([{
        "edad": edad, "ingreso_mensual": ingreso_mensual,
        "monto": monto, "cuotas": cuotas, "tasa": tasa,
        "relacion_cuota_ingreso": rci,
        "numero_dependientes": num_dependientes,
        "antiguedad_laboral": antiguedad_laboral,
        "sector": sector, "regional": regional,
        "actividad_econ": actividad, "vivienda": vivienda,
        "estado_civil": estado_civil, "genero": genero,
        "educ": educ, "mujer_cabeza": mujer_cabeza,
        "responsable_hogar": responsable_hogar,
        "historial_crediticio": historial_crediticio,
    }])

    proba    = pipeline.predict_proba(row)[0]
    risk_pct = proba[1] * 100

    # Tier lookup
    for limit, color, bg, dark, label, subtitle, action in TIERS[lang]:
        if risk_pct < limit:
            break

    # ── RCI pill ───────────────────────────────────────────────────────────────
    is_en = lang == "en"
    if rci > 0.50:
        rci_bg, rci_color, rci_icon = "#fee2e2", "#b91c1c", "⚠"
        rci_note = "Exceeds critical threshold" if is_en else "Supera umbral crítico"
    elif rci > 0.35:
        rci_bg, rci_color, rci_icon = "#ffedd5", "#c2410c", "⚠"
        rci_note = "Exceeds 35% threshold" if is_en else "Supera umbral del 35%"
    elif rci > 0.25:
        rci_bg, rci_color, rci_icon = "#fef9c3", "#a16207", "●"
        rci_note = "Caution zone" if is_en else "Zona de atención"
    else:
        rci_bg, rci_color, rci_icon = "#dcfce7", "#15803d", "✓"
        rci_note = "Within limit" if is_en else "Dentro del límite"

    cap_color = "#b91c1c" if capacidad < 0 else "#15803d"
    cap_bg    = "#fee2e2" if capacidad < 0 else "#dcfce7"
    cap_icon  = "↓" if capacidad < 0 else "↑"
    cap_label = "COP / month available" if is_en else "COP / mes disponibles"

    bar_w = min(risk_pct, 100)

    lbl_prob      = "Default probability"         if is_en else "Probabilidad de incumplimiento"
    lbl_monthly   = "Estimated Monthly Payment"   if is_en else "Cuota Mensual Estimada"
    lbl_amort     = "COP / month · French amort." if is_en else "COP / mes · Amort. francesa"
    lbl_rci       = "Payment-to-Income Ratio"     if is_en else "Relación Cuota / Ingreso"
    lbl_rci_thr   = "threshold 35%"               if is_en else "umbral 35%"
    lbl_capacity  = "Net Payment Capacity"        if is_en else "Capacidad de Pago Neta"
    lbl_term      = "Loan Term"                   if is_en else "Plazo Pactado"
    lbl_rate      = "monthly N.A.M.V. rate"       if is_en else "tasa N.A.M.V. mensual"
    lbl_total     = "Total loan cost:"            if is_en else "Costo total del crédito:"
    lbl_finance   = "Financing cost:"             if is_en else "Costo financiero:"
    lbl_capital   = "Requested amount:"           if is_en else "Monto solicitado:"
    lbl_install   = "installments"                if is_en else "cuotas"

    html = f"""
<div style="font-family:'Inter','Segoe UI',system-ui,sans-serif;margin-top:8px;color:#0f172a;">

  <div style="background:{bg};border:2px solid {color};border-left:6px solid {color};
    border-radius:12px;padding:20px 24px;margin-bottom:16px;">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
      <div>
        <div style="font-size:1.35em;font-weight:800;color:{dark};letter-spacing:-0.02em;">{label}</div>
        <div style="font-size:0.95em;color:{dark};opacity:0.85;margin-top:2px;">{subtitle} · {action}</div>
      </div>
      <div style="background:{dark};color:white;border-radius:50px;padding:8px 20px;
        font-size:1.4em;font-weight:800;white-space:nowrap;">
        {risk_pct:.1f}%
      </div>
    </div>
    <div style="margin-top:16px;">
      <div style="display:flex;justify-content:space-between;font-size:0.75em;color:{dark};margin-bottom:4px;">
        <span>{lbl_prob}</span><span>{risk_pct:.1f}%</span>
      </div>
      <div style="background:rgba(0,0,0,0.10);border-radius:99px;height:10px;overflow:hidden;">
        <div style="width:{bar_w}%;height:100%;background:{color};border-radius:99px;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.70em;color:{dark};opacity:0.7;margin-top:3px;">
        <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
      </div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:12px;">

    <div style="background:#ffffff;border:1px solid #d0d9e6;border-radius:10px;padding:16px;">
      <div style="font-size:0.72em;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
        {lbl_monthly}</div>
      <div style="font-size:1.30em;font-weight:800;color:#0f172a;">${cuota_est:,.0f}</div>
      <div style="font-size:0.78em;color:#475569;margin-top:2px;">{lbl_amort}</div>
    </div>

    <div style="background:{rci_bg};border:1px solid {rci_color}40;border-radius:10px;padding:16px;">
      <div style="font-size:0.72em;font-weight:600;color:{rci_color};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
        {lbl_rci}</div>
      <div style="font-size:1.30em;font-weight:800;color:{rci_color};">{rci*100:.1f}%</div>
      <div style="font-size:0.78em;color:{rci_color};margin-top:2px;">{rci_icon} {rci_note} · {lbl_rci_thr}</div>
    </div>

    <div style="background:{cap_bg};border:1px solid {cap_color}40;border-radius:10px;padding:16px;">
      <div style="font-size:0.72em;font-weight:600;color:{cap_color};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
        {lbl_capacity}</div>
      <div style="font-size:1.30em;font-weight:800;color:{cap_color};">${capacidad:,.0f}</div>
      <div style="font-size:0.78em;color:{cap_color};margin-top:2px;">{cap_icon} {cap_label}</div>
    </div>

    <div style="background:#ffffff;border:1px solid #d0d9e6;border-radius:10px;padding:16px;">
      <div style="font-size:0.72em;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
        {lbl_term}</div>
      <div style="font-size:1.30em;font-weight:800;color:#0f172a;">{int(cuotas)} {lbl_install}</div>
      <div style="font-size:0.78em;color:#475569;margin-top:2px;">{tasa*100:.2f}% {lbl_rate}</div>
    </div>

  </div>

  <div style="background:#e8edf4;border:1px solid #b8c8dc;border-radius:10px;padding:14px 20px;
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;
    font-size:0.88em;color:#1e3a5f !important;">
    <span style="color:#1e3a5f !important;"><b>{lbl_total}</b>&nbsp; ${cuota_est * int(cuotas):,.0f} COP</span>
    <span style="color:#1e3a5f !important;"><b>{lbl_finance}</b>&nbsp; ${cuota_est * int(cuotas) - monto:,.0f} COP ({(cuota_est * int(cuotas) / monto - 1)*100:.1f}%)</span>
    <span style="color:#1e3a5f !important;"><b>{lbl_capital}</b>&nbsp; ${monto:,.0f} COP</span>
  </div>

</div>
"""
    return html


# ── Translations ───────────────────────────────────────────────────────────────
T = {
    "en": {
        "header_title": "Credit Risk Predictor",
        "header_sub":   "Fundación Amanecer · DS4A Correlation One · Team 84",
        "header_desc":  "Evaluates default risk for microcredit applications using Gradient Boosting "
                        "trained on data calibrated to the real profile of Colombian microfinance clients.",
        "sec_financial": "Financial Information",
        "sec_profile":   "Applicant Profile",
        "sec_location":  "Location &amp; Activity",
        "tab_eval":   "Evaluate Application",
        "tab_perf":   "Model Performance",
        "tab_method": "Methodology",
        "lbl_ingreso":    "Monthly net income (COP)",
        "inf_ingreso":    "Regular income declared by the applicant",
        "lbl_monto":      "Requested amount (COP)",
        "inf_monto":      "Loan capital, excluding interest",
        "lbl_cuotas":     "Number of installments",
        "lbl_tasa":       "Monthly interest rate N.A.M.V. (%)",
        "lbl_historial":  "Credit history",
        "inf_historial":  "DataCrédito / TransUnion",
        "lbl_edad":       "Age (years)",
        "lbl_dep":        "Dependents",
        "lbl_antiguedad": "Work seniority (years)",
        "lbl_genero":     "Gender",
        "lbl_civil":      "Marital status",
        "lbl_educ":       "Education level",
        "lbl_mujer":      "Female head of household",
        "lbl_resp":       "Primary household provider",
        "lbl_sector":     "Sector",
        "lbl_regional":   "Region",
        "lbl_actividad":  "Economic activity",
        "lbl_vivienda":   "Housing type",
        "btn_predict":    "Evaluate Credit Risk",
        "btn_charts":     "Load charts",
        "empty_result":   "Complete the form and click <strong style='color:#1e3a5f;'>Evaluate Credit Risk</strong> to get the analysis.",
        "load_note":      "Click 'Load charts' to render the performance charts.",
        "metrics": ["Accuracy", "AUC-ROC", "Precision", "Recall", "F1-Score"],
    },
    "es": {
        "header_title": "Predictor de Riesgo Crediticio",
        "header_sub":   "Fundación Amanecer · DS4A Correlation One · Equipo 84",
        "header_desc":  "Evalúa el riesgo de mora de solicitudes de microcrédito mediante Gradient Boosting "
                        "entrenado con datos calibrados al perfil real de clientes colombianos.",
        "sec_financial": "Información Financiera",
        "sec_profile":   "Perfil del Solicitante",
        "sec_location":  "Ubicación y Actividad",
        "tab_eval":   "Evaluar Solicitud",
        "tab_perf":   "Desempeño del Modelo",
        "tab_method": "Metodología",
        "lbl_ingreso":    "Ingreso mensual neto (COP)",
        "inf_ingreso":    "Ingresos regulares declarados por el solicitante",
        "lbl_monto":      "Monto solicitado (COP)",
        "inf_monto":      "Capital del crédito, sin intereses",
        "lbl_cuotas":     "Número de cuotas",
        "lbl_tasa":       "Tasa mensual N.A.M.V. (%)",
        "lbl_historial":  "Historial crediticio",
        "inf_historial":  "DataCrédito / TransUnion",
        "lbl_edad":       "Edad (años)",
        "lbl_dep":        "Personas a cargo",
        "lbl_antiguedad": "Antigüedad laboral (años)",
        "lbl_genero":     "Género",
        "lbl_civil":      "Estado civil",
        "lbl_educ":       "Nivel educativo",
        "lbl_mujer":      "Mujer cabeza de hogar",
        "lbl_resp":       "Responsable principal del hogar",
        "lbl_sector":     "Sector",
        "lbl_regional":   "Regional",
        "lbl_actividad":  "Actividad económica",
        "lbl_vivienda":   "Tipo de vivienda",
        "btn_predict":    "Evaluar Riesgo Crediticio",
        "btn_charts":     "Cargar gráficas",
        "empty_result":   "Complete el formulario y haga clic en <strong style='color:#1e3a5f;'>Evaluar Riesgo Crediticio</strong> para obtener el análisis.",
        "load_note":      "Haga clic en 'Cargar gráficas' para ver las métricas del modelo.",
        "metrics": ["Accuracy", "AUC-ROC", "Precisión", "Recall", "F1-Score"],
    },
}

# Dropdown choices: (display_label, model_value) — model always receives Spanish values
CHOICES = {
    "en": {
        "historial":  [("Very good", "muy bueno"), ("Good", "bueno"), ("Fair", "regular"),
                       ("No history", "sin historial"), ("Poor", "malo")],
        "genero":     [("Female", "femenino"), ("Male", "masculino")],
        "civil":      [("Married", "casado/a"), ("Single", "soltero/a"), ("Common-law", "unión libre"),
                       ("Divorced", "divorciado/a"), ("Widowed", "viudo/a")],
        "educ":       [("None", "ninguno"), ("Primary (incomplete)", "primaria incompleta"),
                       ("Primary", "primaria completa"), ("Secondary", "secundaria"),
                       ("Technical", "técnico/tecnólogo"), ("University", "universitario")],
        "yesno":      [("Yes", "si"), ("No", "no")],
        "sector":     [("Urban", "urbano"), ("Rural", "rural")],
        "regional":   [("Bogotá D.C.", "bogotá d.c."), ("South Region", "región sur"),
                       ("North Region", "región norte"), ("East Region", "región oriente"),
                       ("West Region", "región occidente")],
        "actividad":  [("Retail trade", "comercio minorista"), ("Agriculture", "agropecuario"),
                       ("Personal services", "servicios personales"), ("Craft industry", "industria artesanal"),
                       ("Construction", "construcción"), ("Transport", "transporte"), ("Other", "otro")],
        "vivienda":   [("Owned (no mortgage)", "propia sin deuda"), ("Owned (with mortgage)", "propia con hipoteca"),
                       ("Rented", "arrendada"), ("Family housing", "familiar")],
    },
    "es": {
        "historial":  [("Muy bueno", "muy bueno"), ("Bueno", "bueno"), ("Regular", "regular"),
                       ("Sin historial", "sin historial"), ("Malo", "malo")],
        "genero":     [("Femenino", "femenino"), ("Masculino", "masculino")],
        "civil":      [("Casado/a", "casado/a"), ("Soltero/a", "soltero/a"), ("Unión libre", "unión libre"),
                       ("Divorciado/a", "divorciado/a"), ("Viudo/a", "viudo/a")],
        "educ":       [("Ninguno", "ninguno"), ("Primaria incompleta", "primaria incompleta"),
                       ("Primaria completa", "primaria completa"), ("Secundaria", "secundaria"),
                       ("Técnico/tecnólogo", "técnico/tecnólogo"), ("Universitario", "universitario")],
        "yesno":      [("Sí", "si"), ("No", "no")],
        "sector":     [("Urbano", "urbano"), ("Rural", "rural")],
        "regional":   [("Bogotá D.C.", "bogotá d.c."), ("Región Sur", "región sur"),
                       ("Región Norte", "región norte"), ("Región Oriente", "región oriente"),
                       ("Región Occidente", "región occidente")],
        "actividad":  [("Comercio minorista", "comercio minorista"), ("Agropecuario", "agropecuario"),
                       ("Servicios personales", "servicios personales"), ("Industria artesanal", "industria artesanal"),
                       ("Construcción", "construcción"), ("Transporte", "transporte"), ("Otro", "otro")],
        "vivienda":   [("Propia sin deuda", "propia sin deuda"), ("Propia con hipoteca", "propia con hipoteca"),
                       ("Arrendada", "arrendada"), ("Familiar", "familiar")],
    },
}

def _header_html(lang):
    t = T[lang]
    return f"""
<div class="app-header">
  <div style="font-size:1.55em;font-weight:800;margin-bottom:6px;letter-spacing:-0.02em;">{t['header_title']}</div>
  <div style="font-size:0.90em;opacity:0.70;font-weight:400;">{t['header_sub']}</div>
  <div style="margin-top:10px;font-size:0.82em;opacity:0.55;max-width:740px;line-height:1.5;">{t['header_desc']}</div>
</div>"""

def _sec_html(lang, key):
    return (f"<div style='"
            f"background:#1e3a5f;color:#ffffff;"
            f"font-size:0.78em;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.09em;padding:11px 18px;"
            f"margin:-1px -1px 16px -1px;border-radius:11px 11px 0 0;"
            f"'>{T[lang][key]}</div>")

def _empty_result(lang):
    return f"<div style='background:#ffffff;border:1px solid #dde3ec;border-radius:10px;" \
           f"padding:40px 24px;text-align:center;color:#94a3b8;font-family:Inter,sans-serif;" \
           f"font-size:0.95em;'>{T[lang]['empty_result']}</div>"

def _metrics_html(lang):
    names = T[lang]["metrics"]
    vals  = [f"{METRICS['accuracy']:.2%}", f"{METRICS['auc']:.3f}",
             f"{METRICS['precision']:.2%}", f"{METRICS['recall']:.2%}", f"{METRICS['f1']:.2%}"]
    cards = "".join(
        f'<div style="background:#ffffff;border:1px solid #dde3ec;border-radius:10px;'
        f'padding:16px;text-align:center;">'
        f'<div style="font-size:0.72em;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#64748b;margin-bottom:6px;">{n}</div>'
        f'<div style="font-size:1.5em;font-weight:800;color:#1e3a5f;">{v}</div>'
        f'</div>'
        for n, v in zip(names, vals)
    )
    return f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:16px 0;">{cards}</div>'


# ── Gradio UI ──────────────────────────────────────────────────────────────────

METHODOLOGY = f"""
## Metodología del Modelo

### Variables del modelo

| Variable | Descripción | Relevancia |
|----------|-------------|------------|
| **RCI** — Relación Cuota/Ingreso | Cuota mensual estimada ÷ Ingreso mensual | Alta |
| **Ingreso mensual** | Ingresos netos declarados (COP) | Alta |
| **Monto del crédito** | Capital solicitado (COP) | Alta |
| **Historial crediticio** | Comportamiento previo en centrales de riesgo | Alta |
| **Tasa de interés** | Tasa N.A.M.V. mensual pactada | Media |
| **Plazo** | Número de cuotas pactadas | Media |
| **Antigüedad laboral** | Años en el trabajo actual | Media |
| **Edad** | Años cumplidos del solicitante | Media |
| **Personas a cargo** | Número de dependientes económicos | Media |

### Umbrales de alerta — RCI

| RCI | Estado | Acción sugerida |
|-----|--------|-----------------|
| Menos de 25% | Bajo riesgo | Aprobación directa |
| Entre 25% y 35% | Atención | Análisis estándar |
| Entre 35% y 50% | Riesgo alto | Requiere garantía o codeudor |
| Mayor a 50% | Riesgo crítico | Negar o reestructurar condiciones |

### Algoritmo

- **Gradient Boosting (GBM)** — 150 árboles, profundidad 4, tasa de aprendizaje 0.10
- **Preprocesamiento:** StandardScaler (variables numéricas) + OneHotEncoding (variables categóricas)
- **Validación:** Split estratificado 80/20

### Métricas del modelo en conjunto de prueba

| Métrica | Valor |
|---------|-------|
| Accuracy | {METRICS['accuracy']:.2%} |
| AUC-ROC | {METRICS['auc']:.3f} |
| Precision | {METRICS['precision']:.2%} |
| Recall | {METRICS['recall']:.2%} |
| F1-Score | {METRICS['f1']:.2%} |
| Muestras de prueba | {len(X_test):,} |

> Los datos de entrenamiento son sintéticos, calibrados al perfil de clientes de microfinanzas colombianas.
> El modelo original fue entrenado con datos reales de Fundación Amanecer, los cuales son de uso privado.
"""

METHODOLOGY_LANG = {
    "en": f"""
## Model Methodology

### Model variables

| Variable | Description | Relevance |
|----------|-------------|-----------|
| **PTI** — Payment-to-Income Ratio | Estimated monthly payment ÷ Monthly income | High |
| **Monthly income** | Net income declared by applicant (COP) | High |
| **Loan amount** | Requested capital (COP) | High |
| **Credit history** | Prior behavior in credit bureaus | High |
| **Interest rate** | Monthly N.A.M.V. rate agreed | Medium |
| **Loan term** | Number of installments agreed | Medium |
| **Work seniority** | Years at current job | Medium |
| **Age** | Applicant's age | Medium |
| **Dependents** | Number of economic dependents | Medium |

### Alert thresholds — PTI (Payment-to-Income)

| PTI | Status | Suggested action |
|-----|--------|-----------------|
| Below 25% | Low risk | Direct approval |
| 25% – 35% | Caution | Standard analysis |
| 35% – 50% | High risk | Require collateral or co-signer |
| Above 50% | Critical risk | Deny or restructure |

### Algorithm

- **Gradient Boosting (GBM)** — 150 trees, depth 4, learning rate 0.10
- **Preprocessing:** StandardScaler (numerical) + OneHotEncoding (categorical)
- **Validation:** Stratified 80/20 split

### Model metrics on test set

| Metric | Value |
|--------|-------|
| Accuracy | {METRICS['accuracy']:.2%} |
| AUC-ROC | {METRICS['auc']:.3f} |
| Precision | {METRICS['precision']:.2%} |
| Recall | {METRICS['recall']:.2%} |
| F1-Score | {METRICS['f1']:.2%} |
| Test samples | {len(X_test):,} |

> Training data is synthetic, calibrated to the profile of Colombian microfinance clients.
> The original model was trained on proprietary data from Fundación Amanecer.
""",
    "es": f"""
## Metodología del Modelo

### Variables del modelo

| Variable | Descripción | Relevancia |
|----------|-------------|------------|
| **RCI** — Relación Cuota/Ingreso | Cuota mensual estimada ÷ Ingreso mensual | Alta |
| **Ingreso mensual** | Ingresos netos declarados (COP) | Alta |
| **Monto del crédito** | Capital solicitado (COP) | Alta |
| **Historial crediticio** | Comportamiento previo en centrales de riesgo | Alta |
| **Tasa de interés** | Tasa N.A.M.V. mensual pactada | Media |
| **Plazo** | Número de cuotas pactadas | Media |
| **Antigüedad laboral** | Años en el trabajo actual | Media |
| **Edad** | Años cumplidos del solicitante | Media |
| **Personas a cargo** | Número de dependientes económicos | Media |

### Umbrales de alerta — RCI

| RCI | Estado | Acción sugerida |
|-----|--------|-----------------|
| Menos de 25% | Bajo riesgo | Aprobación directa |
| Entre 25% y 35% | Atención | Análisis estándar |
| Entre 35% y 50% | Riesgo alto | Requiere garantía o codeudor |
| Mayor a 50% | Riesgo crítico | Negar o reestructurar condiciones |

### Algoritmo

- **Gradient Boosting (GBM)** — 150 árboles, profundidad 4, tasa de aprendizaje 0.10
- **Preprocesamiento:** StandardScaler (variables numéricas) + OneHotEncoding (variables categóricas)
- **Validación:** Split estratificado 80/20

### Métricas del modelo en conjunto de prueba

| Métrica | Valor |
|---------|-------|
| Accuracy | {METRICS['accuracy']:.2%} |
| AUC-ROC | {METRICS['auc']:.3f} |
| Precisión | {METRICS['precision']:.2%} |
| Recall | {METRICS['recall']:.2%} |
| F1-Score | {METRICS['f1']:.2%} |
| Muestras de prueba | {len(X_test):,} |

> Los datos de entrenamiento son sintéticos, calibrados al perfil de clientes de microfinanzas colombianas.
> El modelo original fue entrenado con datos reales de Fundación Amanecer, los cuales son de uso privado.
""",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
/* ── Global light background ── */
body, .gradio-container, .main, .app,
.block, .gr-panel, .gr-box, .gr-form,
.tabs, .tab-content, .tabitem,
.svelte-phx28p, [class*="panel"] {
    background: #f5f7fa !important;
    color: #0f172a !important;
}
.gradio-container {
    max-width: 1800px !important;
    margin: 0 auto !important;
    padding: 16px !important;
}
footer { display: none !important; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2a5298 100%);
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 0;
    color: white;
    box-shadow: 0 4px 20px rgba(30,58,95,0.25);
}

/* ── Language toggle row ── */
.lang-row {
    display: flex !important;
    justify-content: flex-end !important;
    padding: 10px 0 4px !important;
    gap: 0 !important;
    background: transparent !important;
    border: none !important;
}
.lang-btn {
    min-width: 62px !important;
    padding: 7px 18px !important;
    font-size: 0.82em !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    border-radius: 0 !important;
    border: 1.5px solid #1e3a5f !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
}
.lang-btn:first-child { border-radius: 8px 0 0 8px !important; }
.lang-btn:last-child  { border-radius: 0 8px 8px 0 !important; }
.lang-active {
    background: #1e3a5f !important;
    color: #ffffff !important;
}
.lang-inactive {
    background: #ffffff !important;
    color: #1e3a5f !important;
}
.lang-inactive:hover {
    background: #f0f4f8 !important;
}

/* ── Tabs ── */
.tabs > .tab-nav {
    background: #ffffff !important;
    border-radius: 10px !important;
    padding: 5px !important;
    border: 1px solid #d0d9e6 !important;
    gap: 4px !important;
    margin-bottom: 20px !important;
    display: flex !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
.tabs > .tab-nav > button {
    flex: 1 !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 24px !important;
    font-size: 1.0em !important;
    font-weight: 600 !important;
    color: #64748b !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em !important;
}
.tabs > .tab-nav > button:hover {
    background: #f1f5f9 !important;
    color: #1e3a5f !important;
}
.tabs > .tab-nav > button.selected {
    background: #1e3a5f !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(30,58,95,0.30) !important;
    font-weight: 700 !important;
}

/* ── 3-column row: never wrap ── */
.form-row {
    flex-wrap: nowrap !important;
    gap: 14px !important;
    align-items: flex-start !important;
}
.form-row > * {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}

/* ── Form columns (section cards) ── */
.form-col {
    background: #ffffff !important;
    border: 1px solid #d0d9e6 !important;
    border-radius: 12px !important;
    padding: 0 !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(30,58,95,0.07) !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
/* Strip ALL inner borders/backgrounds */
.form-col .block,
.form-col .gr-form,
.form-col .gr-box,
.form-col .gr-panel,
.form-col > div,
.form-col > div > .block {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    border-radius: 0 !important;
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}
/* ── Input / Number / Textarea ── */
.form-col input[type=number],
.form-col input[type=text],
.form-col textarea {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-weight: 500 !important;
    font-size: 0.97em !important;
}
.form-col input:focus,
.form-col textarea:focus {
    border-color: #1e3a5f !important;
    background: #ffffff !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(30,58,95,0.10) !important;
}
/* ── Dropdowns ── */
.form-col .wrap,
.form-col select,
.form-col ul {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
}
/* ── Label badge pills — match section header ── */
.form-col label > span,
.form-col .label-wrap span,
.form-col .gr-label,
.form-col span.svelte-1gfkn6j {
    background: #1e3a5f !important;
    color: #ffffff !important;
    font-size: 0.76em !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    border-radius: 5px !important;
    padding: 2px 8px !important;
}
/* ── Info/hint text ── */
.form-col .info, .form-col .description {
    color: #64748b !important;
    font-size: 0.78em !important;
    background: transparent !important;
}

/* ── Dropdown: selected value text ── */
.form-col .wrap span,
.form-col .wrap input,
.form-col .wrap > span,
.form-col [data-testid="dropdown"] span,
.form-col select option {
    color: #0f172a !important;
    background: transparent !important;
}
/* The displayed text inside closed dropdown */
.form-col .wrap {
    color: #0f172a !important;
}

/* ── Dropdown OPEN list panel ── */
.form-col ul,
.form-col .options,
.form-col [role="listbox"],
.form-col [data-testid="dropdown-options"] {
    background: #ffffff !important;
    border: 1px solid #d0d9e6 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(30,58,95,0.12) !important;
}
.form-col ul li,
.form-col [role="option"],
.form-col .options > * {
    background: #ffffff !important;
    color: #0f172a !important;
    padding: 8px 14px !important;
}
.form-col ul li:hover,
.form-col [role="option"]:hover,
.form-col ul li.selected,
.form-col [role="option"][aria-selected="true"] {
    background: #eef2f7 !important;
    color: #1e3a5f !important;
}

/* ── Placeholder text ── */
.form-col input::placeholder,
.form-col textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* ── Sliders: track + thumb match dark blue ── */
.form-col input[type=range] {
    accent-color: #1e3a5f !important;
}
.form-col [data-testid="block"] {
    background: transparent !important;
}
/* Slider number box */
.form-col input[type=number].svelte-1b3bq2o {
    background: #1e3a5f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}

/* ── Any remaining dark overrides from theme ── */
.form-col * {
    --input-text-size: 0.95em;
    --color-background-primary: #ffffff;
}
.form-col svg, .form-col .icon {
    color: #1e3a5f !important;
}

/* ── Primary action button (centered, fixed width) ── */
.predict-row {
    display: flex !important;
    justify-content: center !important;
    padding: 8px 0 !important;
}
.predict-btn-col {
    max-width: 480px !important;
    min-width: 280px !important;
    width: 38% !important;
}
button.primary, .btn-primary {
    background: linear-gradient(135deg, #1e3a5f 0%, #2a5298 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    padding: 15px 32px !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 3px 12px rgba(30,58,95,0.30) !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}
button.primary:hover { opacity: 0.9 !important; }

/* ── Secondary button ── */
button.secondary {
    border: 1.5px solid #1e3a5f !important;
    color: #1e3a5f !important;
    background: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
button.secondary:hover { background: #f0f4f8 !important; }

/* ── Sliders global ── */
input[type=range] { accent-color: #1e3a5f !important; }

/* ── Global dropdown list (renders outside column in some Gradio versions) ── */
ul.options, [role="listbox"],
.dropdown-content, .svelte-select-list {
    background: #ffffff !important;
    border: 1px solid #d0d9e6 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(30,58,95,0.12) !important;
}
ul.options li, [role="option"] {
    background: #ffffff !important;
    color: #0f172a !important;
}
ul.options li:hover, [role="option"]:hover {
    background: #eef2f7 !important;
    color: #1e3a5f !important;
}
"""

EMPTY_RESULT = """
<div style="
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-radius: 10px;
    padding: 40px 24px;
    text-align: center;
    color: #94a3b8;
    font-family: Inter, sans-serif;
    font-size: 0.95em;
">
    Complete el formulario y haga clic en <strong style="color:#1e3a5f;">Evaluar Riesgo Crediticio</strong>
    para obtener el análisis.
</div>
"""

with gr.Blocks(
    title="Credit Risk Predictor — Microfinance",
    theme=gr.themes.Default(
        primary_hue="blue",
        neutral_hue="gray",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    lang_state = gr.State("en")

    # ── Top bar: header + language toggle ──────────────────────────────────────
    with gr.Row(elem_classes=["lang-row"]):
        with gr.Column(scale=10):
            header_html = gr.HTML(_header_html("en"))
        with gr.Column(scale=0, min_width=140, elem_classes=["lang-toggle-col"]):
            with gr.Row(elem_classes=["lang-row"]):
                lang_en_btn = gr.Button("ENG", elem_classes=["lang-btn", "lang-active"],  scale=1)
                lang_es_btn = gr.Button("SPA", elem_classes=["lang-btn", "lang-inactive"], scale=1)

    with gr.Tabs():

        # ── Tab 1: Evaluate ────────────────────────────────────────────────────
        with gr.Tab("Evaluate Application", id="tab_eval"):
            with gr.Row(equal_height=False, elem_classes=["form-row"]):

                with gr.Column(scale=1, elem_classes=["form-col"]):
                    sec_fin = gr.HTML(_sec_html("en", "sec_financial"))
                    ingreso_in = gr.Number(
                        value=1_800_000,
                        label=T["en"]["lbl_ingreso"],
                        info=T["en"]["inf_ingreso"])
                    monto_in = gr.Number(
                        value=5_000_000,
                        label=T["en"]["lbl_monto"],
                        info=T["en"]["inf_monto"])
                    cuotas_in = gr.Dropdown(
                        [6, 9, 12, 18, 24, 36, 48], value=12,
                        label=T["en"]["lbl_cuotas"])
                    tasa_in = gr.Slider(
                        1.5, 4.2, value=2.2, step=0.1,
                        label=T["en"]["lbl_tasa"],
                        info="e.g. 2.2 = 2.2% monthly")
                    historial_in = gr.Dropdown(
                        choices=CHOICES["en"]["historial"], value="bueno",
                        label=T["en"]["lbl_historial"],
                        info=T["en"]["inf_historial"])

                with gr.Column(scale=1, elem_classes=["form-col"]):
                    sec_prof = gr.HTML(_sec_html("en", "sec_profile"))
                    edad_in       = gr.Slider(20, 70, value=38, step=1, label=T["en"]["lbl_edad"])
                    num_dep_in    = gr.Slider(0, 6, value=2, step=1,   label=T["en"]["lbl_dep"])
                    antiguedad_in = gr.Slider(0, 25, value=3, step=1,  label=T["en"]["lbl_antiguedad"])
                    genero_in  = gr.Dropdown(choices=CHOICES["en"]["genero"],
                                             value="femenino",    label=T["en"]["lbl_genero"])
                    civil_in   = gr.Dropdown(choices=CHOICES["en"]["civil"],
                                             value="unión libre", label=T["en"]["lbl_civil"])
                    educ_in    = gr.Dropdown(choices=CHOICES["en"]["educ"],
                                             value="secundaria",  label=T["en"]["lbl_educ"])
                    mujer_in   = gr.Dropdown(choices=CHOICES["en"]["yesno"],
                                             value="no",          label=T["en"]["lbl_mujer"])
                    resp_in    = gr.Dropdown(choices=CHOICES["en"]["yesno"],
                                             value="si",          label=T["en"]["lbl_resp"])

                with gr.Column(scale=1, elem_classes=["form-col"]):
                    sec_loc = gr.HTML(_sec_html("en", "sec_location"))
                    sector_in    = gr.Dropdown(choices=CHOICES["en"]["sector"],
                                               value="urbano",             label=T["en"]["lbl_sector"])
                    regional_in  = gr.Dropdown(choices=CHOICES["en"]["regional"],
                                               value="bogotá d.c.",        label=T["en"]["lbl_regional"])
                    actividad_in = gr.Dropdown(choices=CHOICES["en"]["actividad"],
                                               value="comercio minorista", label=T["en"]["lbl_actividad"])
                    vivienda_in  = gr.Dropdown(choices=CHOICES["en"]["vivienda"],
                                               value="propia sin deuda",   label=T["en"]["lbl_vivienda"])

            with gr.Row(elem_classes=["predict-row"]):
                with gr.Column(scale=0, elem_classes=["predict-btn-col"]):
                    predict_btn = gr.Button(T["en"]["btn_predict"], variant="primary", size="lg")

            result_out  = gr.HTML(value=_empty_result("en"))

            predict_btn.click(
                lambda edad, ingreso, monto, cuotas, tasa, dep, ant,
                       sector, regional, actividad, vivienda,
                       civil, genero, educ, mujer, resp, historial, lang: predict_risk(
                    edad, ingreso, monto, cuotas, tasa / 100, dep, ant,
                    sector, regional, actividad, vivienda,
                    civil, genero, educ, mujer, resp, historial, lang,
                ),
                inputs=[
                    edad_in, ingreso_in, monto_in, cuotas_in, tasa_in,
                    num_dep_in, antiguedad_in,
                    sector_in, regional_in, actividad_in, vivienda_in,
                    civil_in, genero_in, educ_in, mujer_in, resp_in,
                    historial_in, lang_state,
                ],
                outputs=[result_out],
            )

        # ── Tab 2: Performance ─────────────────────────────────────────────────
        with gr.Tab("Model Performance", id="tab_perf"):
            metrics_html = gr.HTML(_metrics_html("en"))
            with gr.Row():
                cm_plot = gr.Plot(label="Confusion Matrix")
                fi_plot = gr.Plot(label="Feature Importance")
            with gr.Row():
                rci_plot    = gr.Plot(label="PTI & Credit History")
                income_plot = gr.Plot(label="Income vs Default")

            gr.HTML("<div style='height:8px;'></div>")
            with gr.Row(elem_classes=["predict-row"]):
                with gr.Column(scale=0, elem_classes=["predict-btn-col"]):
                    load_btn = gr.Button(T["en"]["btn_charts"], variant="primary", size="lg")
            load_btn.click(
                lambda: (_cm_fig(), _fi_fig(), _rci_fig(), _income_fig()),
                outputs=[cm_plot, fi_plot, rci_plot, income_plot],
            )

        # ── Tab 3: Methodology ─────────────────────────────────────────────────
        with gr.Tab("Methodology", id="tab_method"):
            method_md = gr.Markdown(METHODOLOGY_LANG["en"])

    # ── Language switch logic ──────────────────────────────────────────────────
    def switch_lang(lang):
        t  = T[lang]
        ch = CHOICES[lang]
        is_en = lang == "en"
        return (
            lang,
            _header_html(lang),
            gr.update(elem_classes=["lang-btn", "lang-active"   if is_en  else "lang-inactive"]),
            gr.update(elem_classes=["lang-btn", "lang-inactive" if is_en  else "lang-active"]),
            _sec_html(lang, "sec_financial"),
            _sec_html(lang, "sec_profile"),
            _sec_html(lang, "sec_location"),
            gr.update(label=t["lbl_ingreso"],    info=t["inf_ingreso"]),
            gr.update(label=t["lbl_monto"],       info=t["inf_monto"]),
            gr.update(label=t["lbl_cuotas"]),
            gr.update(label=t["lbl_tasa"]),
            gr.update(label=t["lbl_historial"],   info=t["inf_historial"], choices=ch["historial"]),
            gr.update(label=t["lbl_edad"]),
            gr.update(label=t["lbl_dep"]),
            gr.update(label=t["lbl_antiguedad"]),
            gr.update(label=t["lbl_genero"],      choices=ch["genero"]),
            gr.update(label=t["lbl_civil"],       choices=ch["civil"]),
            gr.update(label=t["lbl_educ"],        choices=ch["educ"]),
            gr.update(label=t["lbl_mujer"],       choices=ch["yesno"]),
            gr.update(label=t["lbl_resp"],        choices=ch["yesno"]),
            gr.update(label=t["lbl_sector"],      choices=ch["sector"]),
            gr.update(label=t["lbl_regional"],    choices=ch["regional"]),
            gr.update(label=t["lbl_actividad"],   choices=ch["actividad"]),
            gr.update(label=t["lbl_vivienda"],    choices=ch["vivienda"]),
            gr.update(value=t["btn_predict"]),
            _empty_result(lang),
            _metrics_html(lang),
            gr.update(value=t["btn_charts"]),
            METHODOLOGY_LANG[lang],
        )

    lang_outputs = [
        lang_state, header_html,
        lang_en_btn, lang_es_btn,
        sec_fin, sec_prof, sec_loc,
        ingreso_in, monto_in, cuotas_in, tasa_in, historial_in,
        edad_in, num_dep_in, antiguedad_in,
        genero_in, civil_in, educ_in, mujer_in, resp_in,
        sector_in, regional_in, actividad_in, vivienda_in,
        predict_btn, result_out,
        metrics_html, load_btn,
        method_md,
    ]

    lang_en_btn.click(lambda: switch_lang("en"), outputs=lang_outputs)
    lang_es_btn.click(lambda: switch_lang("es"), outputs=lang_outputs)

if __name__ == "__main__":
    demo.launch()
