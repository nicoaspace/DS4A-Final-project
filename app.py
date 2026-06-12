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


# ── Risk tier config ───────────────────────────────────────────────────────────
TIERS = [
    (20,  "#16a34a", "#dcfce7", "#15803d", "RIESGO BAJO",
     "Perfil sólido", "Aprobación recomendada"),
    (40,  "#ca8a04", "#fef9c3", "#a16207", "RIESGO MODERADO",
     "Revisar condiciones", "Solicitar aval o garantía adicional"),
    (60,  "#ea580c", "#ffedd5", "#c2410c", "RIESGO ALTO",
     "Riesgo elevado", "Requiere codeudor o garantía real"),
    (101, "#dc2626", "#fee2e2", "#b91c1c", "RIESGO MUY ALTO",
     "Perfil de alto riesgo", "No se recomienda el desembolso"),
]


def predict_risk(
    edad, ingreso_mensual, monto, cuotas, tasa,
    num_dependientes, antiguedad_laboral,
    sector, regional, actividad, vivienda,
    estado_civil, genero, educ,
    mujer_cabeza, responsable_hogar, historial_crediticio,
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
    for limit, color, bg, dark, label, subtitle, action in TIERS:
        if risk_pct < limit:
            break

    # ── RCI pill ───────────────────────────────────────────────────────────────
    if rci > 0.50:
        rci_bg, rci_color, rci_icon = "#fee2e2", "#b91c1c", "⚠"
        rci_note = "Supera umbral crítico"
    elif rci > 0.35:
        rci_bg, rci_color, rci_icon = "#ffedd5", "#c2410c", "⚠"
        rci_note = "Supera umbral del 35%"
    elif rci > 0.25:
        rci_bg, rci_color, rci_icon = "#fef9c3", "#a16207", "●"
        rci_note = "Zona de atención"
    else:
        rci_bg, rci_color, rci_icon = "#dcfce7", "#15803d", "✓"
        rci_note = "Dentro del límite"

    # ── Capacity pill ──────────────────────────────────────────────────────────
    cap_color = "#b91c1c" if capacidad < 0 else "#15803d"
    cap_bg    = "#fee2e2" if capacidad < 0 else "#dcfce7"
    cap_icon  = "↓" if capacidad < 0 else "↑"

    # ── Progress bar width ─────────────────────────────────────────────────────
    bar_w = min(risk_pct, 100)

    html = f"""
<div style="font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; margin-top: 8px;">

  <!-- VERDICT CARD -->
  <div style="
    background: {bg};
    border: 2px solid {color};
    border-left: 6px solid {color};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  ">
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
      <div>
        <div style="font-size:1.35em; font-weight:800; color:{dark}; letter-spacing:-0.02em;">
          {label}
        </div>
        <div style="font-size:0.95em; color:{dark}; opacity:0.85; margin-top:2px;">
          {subtitle} · {action}
        </div>
      </div>
      <div style="
        background: {dark};
        color: white;
        border-radius: 50px;
        padding: 8px 20px;
        font-size: 1.4em;
        font-weight: 800;
        white-space: nowrap;
      ">
        {risk_pct:.1f}% mora
      </div>
    </div>

    <!-- Risk bar -->
    <div style="margin-top:16px;">
      <div style="display:flex; justify-content:space-between; font-size:0.75em; color:{dark}; margin-bottom:4px;">
        <span>Probabilidad de incumplimiento</span>
        <span>{risk_pct:.1f}%</span>
      </div>
      <div style="background: rgba(0,0,0,0.10); border-radius:99px; height:10px; overflow:hidden;">
        <div style="
          width: {bar_w}%;
          height: 100%;
          background: {color};
          border-radius: 99px;
          transition: width 0.4s ease;
        "></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:0.70em; color:{dark}; opacity:0.7; margin-top:3px;">
        <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
      </div>
    </div>
  </div>

  <!-- METRICS GRID -->
  <div style="
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
  ">

    <!-- Cuota mensual -->
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px;">
      <div style="font-size:0.72em; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
        Cuota Mensual Estimada
      </div>
      <div style="font-size:1.30em; font-weight:800; color:#0f172a;">
        ${cuota_est:,.0f}
      </div>
      <div style="font-size:0.78em; color:#64748b; margin-top:2px;">COP / mes · Amort. francesa</div>
    </div>

    <!-- RCI -->
    <div style="background:{rci_bg}; border:1px solid {rci_color}40; border-radius:10px; padding:16px;">
      <div style="font-size:0.72em; font-weight:600; color:{rci_color}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
        Relación Cuota / Ingreso
      </div>
      <div style="font-size:1.30em; font-weight:800; color:{rci_color};">
        {rci*100:.1f}%
      </div>
      <div style="font-size:0.78em; color:{rci_color}; margin-top:2px;">
        {rci_icon} {rci_note} · umbral 35%
      </div>
    </div>

    <!-- Capacidad neta -->
    <div style="background:{cap_bg}; border:1px solid {cap_color}40; border-radius:10px; padding:16px;">
      <div style="font-size:0.72em; font-weight:600; color:{cap_color}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
        Capacidad de Pago Neta
      </div>
      <div style="font-size:1.30em; font-weight:800; color:{cap_color};">
        ${capacidad:,.0f}
      </div>
      <div style="font-size:0.78em; color:{cap_color}; margin-top:2px;">
        {cap_icon} COP / mes disponibles
      </div>
    </div>

    <!-- Plazo -->
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px;">
      <div style="font-size:0.72em; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
        Plazo Pactado
      </div>
      <div style="font-size:1.30em; font-weight:800; color:#0f172a;">
        {int(cuotas)} cuotas
      </div>
      <div style="font-size:0.78em; color:#64748b; margin-top:2px;">
        Tasa {tasa*100:.2f}% N.A.M.V. mensual
      </div>
    </div>

  </div>

  <!-- TOTAL COST ROW -->
  <div style="
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 12px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.88em;
    color: #475569;
  ">
    <span><b>Costo total del crédito:</b> ${cuota_est * int(cuotas):,.0f} COP</span>
    <span><b>Costo financiero:</b> ${cuota_est * int(cuotas) - monto:,.0f} COP ({(cuota_est * int(cuotas) / monto - 1)*100:.1f}% sobre el capital)</span>
    <span><b>Monto solicitado:</b> ${monto:,.0f} COP</span>
  </div>

</div>
"""
    return html


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

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
body, .gradio-container, .main {
    background: #f0f4f8 !important;
}
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}
footer { display: none !important; }

/* Header bar */
.app-header {
    background: #1e3a5f;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 4px;
    color: white;
}

/* Tab bar */
.tabs > .tab-nav {
    background: #ffffff !important;
    border-radius: 10px !important;
    padding: 6px !important;
    border: 1px solid #dde3ec !important;
    gap: 6px !important;
    margin-bottom: 16px !important;
    display: flex !important;
}
.tabs > .tab-nav > button {
    flex: 1 !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 20px !important;
    font-size: 0.95em !important;
    font-weight: 600 !important;
    color: #64748b !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.tabs > .tab-nav > button:hover {
    background: #f1f5f9 !important;
    color: #1e3a5f !important;
}
.tabs > .tab-nav > button.selected {
    background: #1e3a5f !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(30,58,95,0.25) !important;
}

/* Form panels */
.gr-panel, .gr-box, .gr-form, .block {
    background: #ffffff !important;
    border: 1px solid #dde3ec !important;
    border-radius: 10px !important;
}

/* Labels */
label > span, .gr-label {
    font-size: 0.82em !important;
    font-weight: 600 !important;
    color: #374151 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* Primary button */
button.primary {
    background: #1e3a5f !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1.0em !important;
    padding: 14px 28px !important;
    letter-spacing: 0.02em !important;
}
button.primary:hover {
    background: #16304f !important;
}

/* Secondary button */
button.secondary {
    border: 1.5px solid #1e3a5f !important;
    color: #1e3a5f !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* Sliders accent */
input[type=range]::-webkit-slider-thumb { background: #1e3a5f !important; }
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
    title="Predictor de Riesgo Crediticio — Microfinanzas",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    gr.HTML("""
    <div class="app-header">
        <div style="font-size:1.5em; font-weight:800; margin-bottom:4px; letter-spacing:-0.02em;">
            Predictor de Riesgo Crediticio
        </div>
        <div style="font-size:0.92em; opacity:0.75; font-weight:400;">
            Fundación Amanecer &nbsp;·&nbsp; DS4A Correlation One &nbsp;·&nbsp; Equipo 84
        </div>
        <div style="margin-top:10px; font-size:0.83em; opacity:0.60;">
            Evalúa el riesgo de mora de solicitudes de microcrédito mediante Gradient Boosting
            entrenado con datos calibrados al perfil real de clientes colombianos.
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Evaluar ─────────────────────────────────────────────────────
        with gr.Tab("Evaluar Solicitud"):
            with gr.Row(equal_height=False):

                # Column 1 — Financial
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size:0.78em;font-weight:700;text-transform:uppercase;"
                            "letter-spacing:0.08em;color:#1e3a5f;padding:8px 0 4px;'>Información Financiera</div>")
                    ingreso_in = gr.Number(
                        value=1_800_000, label="Ingreso mensual neto (COP)",
                        info="Ingresos regulares declarados por el solicitante")
                    monto_in = gr.Number(
                        value=5_000_000, label="Monto solicitado (COP)",
                        info="Capital del crédito, sin intereses")
                    cuotas_in = gr.Dropdown(
                        [6, 9, 12, 18, 24, 36, 48], value=12,
                        label="Número de cuotas")
                    tasa_in = gr.Slider(
                        0.015, 0.042, value=0.022, step=0.001,
                        label="Tasa mensual N.A.M.V.")
                    historial_in = gr.Dropdown(
                        HISTORIAL, value="bueno",
                        label="Historial crediticio",
                        info="DataCrédito / TransUnion")

                # Column 2 — Demographics
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size:0.78em;font-weight:700;text-transform:uppercase;"
                            "letter-spacing:0.08em;color:#1e3a5f;padding:8px 0 4px;'>Perfil del Solicitante</div>")
                    edad_in       = gr.Slider(20, 70, value=38, step=1, label="Edad (años)")
                    num_dep_in    = gr.Slider(0, 6, value=2, step=1, label="Personas a cargo")
                    antiguedad_in = gr.Slider(0, 25, value=3, step=1, label="Antigüedad laboral (años)")
                    genero_in     = gr.Dropdown(GENDERS,    value="femenino",    label="Género")
                    civil_in      = gr.Dropdown(CIVIL,      value="unión libre", label="Estado civil")
                    educ_in       = gr.Dropdown(EDUCATION,  value="secundaria",  label="Nivel educativo")
                    mujer_in      = gr.Dropdown(YES_NO, value="no",  label="Mujer cabeza de hogar")
                    resp_in       = gr.Dropdown(YES_NO, value="si",  label="Responsable principal del hogar")

                # Column 3 — Location
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size:0.78em;font-weight:700;text-transform:uppercase;"
                            "letter-spacing:0.08em;color:#1e3a5f;padding:8px 0 4px;'>Ubicación y Actividad</div>")
                    sector_in    = gr.Dropdown(SECTORS,    value="urbano",           label="Sector")
                    regional_in  = gr.Dropdown(REGIONS,    value="bogotá d.c.",      label="Regional")
                    actividad_in = gr.Dropdown(ACTIVITIES, value="comercio minorista", label="Actividad económica")
                    vivienda_in  = gr.Dropdown(HOUSING,    value="propia sin deuda", label="Tipo de vivienda")

            predict_btn = gr.Button(
                "Evaluar Riesgo Crediticio",
                variant="primary", size="lg",
            )
            result_out = gr.HTML(value=EMPTY_RESULT)

            predict_btn.click(
                predict_risk,
                inputs=[
                    edad_in, ingreso_in, monto_in, cuotas_in, tasa_in,
                    num_dep_in, antiguedad_in,
                    sector_in, regional_in, actividad_in, vivienda_in,
                    civil_in, genero_in, educ_in, mujer_in, resp_in,
                    historial_in,
                ],
                outputs=[result_out],
            )

        # ── Tab 2: Model Performance ───────────────────────────────────────────
        with gr.Tab("Desempeño del Modelo"):
            gr.HTML(f"""
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:16px 0;">
                {"".join(
                    f'<div style="background:#ffffff;border:1px solid #dde3ec;border-radius:10px;'
                    f'padding:16px;text-align:center;">'
                    f'<div style="font-size:0.72em;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.06em;color:#64748b;margin-bottom:6px;">{name}</div>'
                    f'<div style="font-size:1.5em;font-weight:800;color:#1e3a5f;">{val}</div>'
                    f'</div>'
                    for name, val in [
                        ("Accuracy",   f"{METRICS['accuracy']:.2%}"),
                        ("AUC-ROC",    f"{METRICS['auc']:.3f}"),
                        ("Precision",  f"{METRICS['precision']:.2%}"),
                        ("Recall",     f"{METRICS['recall']:.2%}"),
                        ("F1-Score",   f"{METRICS['f1']:.2%}"),
                    ]
                )}
            </div>
            """)
            with gr.Row():
                cm_plot = gr.Plot(label="Matriz de Confusión")
                fi_plot = gr.Plot(label="Importancia de Variables")
            with gr.Row():
                rci_plot    = gr.Plot(label="RCI e Historial Crediticio")
                income_plot = gr.Plot(label="Ingreso vs Mora")

            load_btn = gr.Button("Cargar gráficas", variant="secondary")
            load_btn.click(
                lambda: (_cm_fig(), _fi_fig(), _rci_fig(), _income_fig()),
                outputs=[cm_plot, fi_plot, rci_plot, income_plot],
            )

        # ── Tab 3: Methodology ─────────────────────────────────────────────────
        with gr.Tab("Metodología"):
            gr.Markdown(METHODOLOGY)

if __name__ == "__main__":
    demo.launch()
