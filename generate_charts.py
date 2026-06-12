"""Run once locally to generate README chart images."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import os

os.makedirs("docs/images", exist_ok=True)

BG = "#f8fdf9"; G1 = "#1a6b2e"; G2 = "#2ecc71"; G3 = "#a9dfbf"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                     "axes.edgecolor": "#ccc", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "sans-serif"})

SECTORS = ["urbano","rural"]; REGIONS = ["bogotá","región sur","región norte","región oriente","región occidente"]
ACTIVITIES = ["comercio","agropecuario","servicios","industria","construcción"]
HOUSING = ["propia","arrendada","familiar","otro"]; CIVIL = ["casado","soltero","unión libre","divorciado","viudo"]
GENDERS = ["femenino","masculino"]; EDUCATION = ["primaria","secundaria","técnico","universitario","ninguno"]
YES_NO = ["si","no"]
CATEGORICAL = ["sector","regional","actividad_econ","vivienda","estado_civil","genero","educ","mujer_cabeza","responsable_hogar"]
NUMERICAL = ["edad","monto","cuotas","tasa"]

rng = np.random.default_rng(42); N = 6000
edad=rng.normal(38,12,N).clip(18,75).astype(int); monto=rng.lognormal(15,1.1,N).clip(500_000,50_000_000)
cuotas=rng.choice([6,12,18,24,36,48,60],N); tasa=rng.uniform(0.01,0.045,N)
sector=rng.choice(SECTORS,N,p=[.55,.45]); regional=rng.choice(REGIONS,N)
actividad_econ=rng.choice(ACTIVITIES,N,p=[.35,.25,.20,.12,.08])
vivienda=rng.choice(HOUSING,N,p=[.45,.30,.20,.05]); estado_civil=rng.choice(CIVIL,N,p=[.40,.25,.25,.05,.05])
genero=rng.choice(GENDERS,N,p=[.58,.42]); educ=rng.choice(EDUCATION,N,p=[.20,.35,.25,.15,.05])
mujer_cabeza=rng.choice(YES_NO,N,p=[.35,.65]); responsable_hogar=rng.choice(YES_NO,N,p=[.72,.28])
risk=((edad<25).astype(float)*.30+(monto>20_000_000).astype(float)*.20+(cuotas>36).astype(float)*.15
      +(tasa>0.035).astype(float)*.10+(educ=="ninguno").astype(float)*.15
      +(vivienda=="arrendada").astype(float)*.10+rng.uniform(0,.30,N))
classification=(risk>0.50).astype(int)

df=pd.DataFrame({"edad":edad,"monto":monto,"cuotas":cuotas,"tasa":tasa,"sector":sector,"regional":regional,
                 "actividad_econ":actividad_econ,"vivienda":vivienda,"estado_civil":estado_civil,"genero":genero,
                 "educ":educ,"mujer_cabeza":mujer_cabeza,"responsable_hogar":responsable_hogar,"classification":classification})

scaler=StandardScaler(); encoder=OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore")
X_num=scaler.fit_transform(df[NUMERICAL]); X_cat=encoder.fit_transform(df[CATEGORICAL])
X=np.concatenate([X_num,X_cat],axis=1); y=df["classification"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=5,random_state=42,n_jobs=-1)
model.fit(X_train,y_train); y_pred=model.predict(X_test)
y_proba=model.predict_proba(X_test)[:,1]
acc=accuracy_score(y_test,y_pred); auc=roc_auc_score(y_test,y_proba)
print(f"Accuracy: {acc:.4f}  AUC: {auc:.4f}")

cat_names=encoder.get_feature_names_out(CATEGORICAL).tolist()
all_names=NUMERICAL+cat_names

# ── 1. Confusion Matrix ──────────────────────────────────────────────────────
cm=confusion_matrix(y_test,y_pred)
fig,ax=plt.subplots(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Greens",ax=ax,
            xticklabels=["Low Risk","High Risk"],yticklabels=["Low Risk","High Risk"],
            linewidths=0.5,linecolor="white",annot_kws={"size":14,"weight":"bold"})
ax.set_xlabel("Predicted",fontsize=11); ax.set_ylabel("Actual",fontsize=11)
ax.set_title(f"Confusion Matrix\nAccuracy {acc:.2%}  ·  AUC {auc:.3f}",fontsize=12,fontweight="bold",pad=12)
plt.tight_layout(); plt.savefig("docs/images/confusion_matrix.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved confusion_matrix.png")

# ── 2. Feature Importance ─────────────────────────────────────────────────────
importances=model.feature_importances_
top_idx=np.argsort(importances)[-15:]; top_names=[all_names[i] for i in top_idx]; top_vals=importances[top_idx]
palette=[G1 if v>=0.08 else G2 if v>=0.04 else G3 for v in top_vals]
fig,ax=plt.subplots(figsize=(7,5))
bars=ax.barh(top_names,top_vals,color=palette,edgecolor="white")
ax.bar_label(bars,fmt="%.3f",padding=4,fontsize=8,color="#333")
ax.set_xlabel("Mean Decrease in Impurity",fontsize=10)
ax.set_title("Top 15 Feature Importances — Random Forest",fontsize=12,fontweight="bold")
ax.tick_params(axis="y",labelsize=8)
plt.tight_layout(); plt.savefig("docs/images/feature_importance.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved feature_importance.png")

# ── 3. Risk Distribution ──────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(10,4))
for cls,label,color in [(0,"Low Risk",G2),(1,"High Risk","#e74c3c")]:
    axes[0].hist(df[df["classification"]==cls]["monto"]/1_000_000,bins=30,alpha=0.65,label=label,color=color,edgecolor="white")
axes[0].set_xlabel("Loan Amount (M COP)",fontsize=10); axes[0].set_ylabel("Count",fontsize=10)
axes[0].set_title("Loan Amount by Risk Class",fontsize=11,fontweight="bold"); axes[0].legend(fontsize=9)
for cls,label,color in [(0,"Low Risk",G2),(1,"High Risk","#e74c3c")]:
    axes[1].hist(df[df["classification"]==cls]["edad"],bins=25,alpha=0.65,label=label,color=color,edgecolor="white")
axes[1].set_xlabel("Age (years)",fontsize=10); axes[1].set_ylabel("Count",fontsize=10)
axes[1].set_title("Age Distribution by Risk Class",fontsize=11,fontweight="bold"); axes[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("docs/images/distributions.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved distributions.png")

# ── 4. Default rate by category ───────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(11,4))
risk_by_educ=df.groupby("educ")["classification"].mean().sort_values()
axes[0].barh(risk_by_educ.index,risk_by_educ.values*100,color=G1,edgecolor="white")
axes[0].set_xlabel("Default Rate (%)",fontsize=10)
axes[0].set_title("Default Rate by Education Level",fontsize=11,fontweight="bold")
axes[0].bar_label(axes[0].containers[0],fmt="%.1f%%",padding=3,fontsize=8)
risk_by_act=df.groupby("actividad_econ")["classification"].mean().sort_values()
axes[1].barh(risk_by_act.index,risk_by_act.values*100,color=G2,edgecolor="white")
axes[1].set_xlabel("Default Rate (%)",fontsize=10)
axes[1].set_title("Default Rate by Economic Activity",fontsize=11,fontweight="bold")
axes[1].bar_label(axes[1].containers[0],fmt="%.1f%%",padding=3,fontsize=8)
plt.tight_layout(); plt.savefig("docs/images/default_rates.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved default_rates.png")

print("\nAll charts generated in docs/images/")
