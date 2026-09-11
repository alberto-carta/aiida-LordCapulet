# %% --- Configuration ---------------------------------------------------------
MATERIAL_NAME = "UO2"
JSON_FILE = f"../{MATERIAL_NAME}_scan_data_extractor_so_n.json"
OUTPUT_DIR = "."

# Feature groups to include (True/False)
USE_HUBBARD      = True   # hubbard_term
USE_HUND         = True   # hund_term_M2
USE_HEISENBERG   = True   # heisenberg_i_j (binary coupling)
USE_CRYSTAL_FIELD = True  # cf_atom* (occupation matrix diagonals)
USE_CF_OFFDIAGONAL = True  # off-diagonal occupation matrix elements (i < j)

# Train/test split
TEST_SIZE    = 0.90
RANDOM_STATE = 40

# LCB (Lower Confidence Bound) acquisition
LCB_BETA  = 0.5          # exploration-exploitation tradeoff: μ - β·σ
LCB_ETA   = 10          # inverse temperature η for Boltzmann weight: P ∝ exp(-LCB · η)

# ARD pruning threshold (higher = less aggressive, keeps more features)
ARD_THRESHOLD_LAMBDA = 100000  # default=10000; try 1e5, 1e6, 1e7 for milder pruning

# Random Forest (bagged) with uncertainty from tree ensemble
RF_N_ESTIMATORS   = 300    # number of trees
RF_MAX_DEPTH      = None      # max tree depth (None = unlimited)
RF_MAX_LEAF_NODES = None     # max leaf nodes per tree (None = unlimited)
RF_MIN_SAMPLES_LEAF = 2    # min samples per leaf


# %% --- Imports ---------------------------------------------------------------
import numpy as np
import pandas as pd

from data_parser import parse_json_file
from feature_engineering import featurize_all
from sklearn.linear_model import LinearRegression, BayesianRidge, ARDRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# %% --- Step 1: Parse & Featurize ---------------------------------------------
print("=" * 60)
print(f"Material: {MATERIAL_NAME}")
print(f"JSON:     {JSON_FILE}")
print("=" * 60)

records = parse_json_file(JSON_FILE)
fit_df = featurize_all(records, include_cf_offdiagonal=USE_CF_OFFDIAGONAL)

# %% --- Step 2: Select Features -----------------------------------------------
target_col = "total_energy_eV"
all_cols = [c for c in fit_df.columns if c not in ("calculation_id", target_col)]

# Build feature list from toggles
feature_cols = []
if USE_HUBBARD:
    feature_cols += [c for c in all_cols if c == "hubbard_term"]
if USE_HUND:
    feature_cols += [c for c in all_cols if c == "hund_term_M2"]
if USE_HEISENBERG:
    feature_cols += [c for c in all_cols if c.startswith("heisenberg_")]
if USE_CRYSTAL_FIELD:
    if USE_CF_OFFDIAGONAL:
        feature_cols += [c for c in all_cols if c.startswith("cf_")]
    else:
        # Only diagonals: cf_atom{k}_n_{i}_{i} where the two indices match
        import re
        for c in all_cols:
            m = re.match(r"cf_atom\d+_n_(\d+)_(\d+)", c)
            if m and m.group(1) == m.group(2):
                feature_cols.append(c)

print(f"\nTarget:        {target_col}")
print(f"Num features:  {len(feature_cols)}")
print(f"Num samples:   {len(fit_df)}")
print(f"\nActive feature groups:"
     f"\n  hubbard       = {USE_HUBBARD}"
     f"\n  hund          = {USE_HUND}"
     f"\n  heisenberg    = {USE_HEISENBERG}"
     f"\n  crystal_field = {USE_CRYSTAL_FIELD}  (offdiag={USE_CF_OFFDIAGONAL})")
print(f"\nFeatures:")
for i, col in enumerate(feature_cols):
    print(f"  [{i:2d}] {col}")

# %% --- Step 3: Train/Test Split ----------------------------------------------
X = fit_df[feature_cols].values
y = fit_df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nTrain: {len(X_train)} samples,  Test: {len(X_test)} samples")

# %% --- Step 4: Fit Linear Regression -----------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# %% --- Step 5: Results -------------------------------------------------------
coeffs = pd.Series(model.coef_, index=feature_cols)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Physical parameters (only if their features are active)
U_val = coeffs["hubbard_term"] * 2 if USE_HUBBARD else None
J_val = -coeffs["hund_term_M2"] * 4 if USE_HUND else None

print(f"\n--- Linear Regression Results ---")
print(f"Intercept:  {model.intercept_:.4f}")
print(f"U (eff):    {U_val:.4f} eV")
print(f"J (eff):    {J_val:.4f} eV")
print(f"R² (test):  {r2:.4f}")
print(f"RMSE (test):{rmse:.4f} eV")
print(f"\nAll coefficients:")
for name, val in coeffs.items():
    print(f"  {name:40s} {val:+.6e}")

# %% --- Step 6: Plot Predicted vs Actual (Linear) -----------------------------
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})

y_pred_all = model.predict(X)

# Single linear plot (kept for backward compatibility)
plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred_all, color="steelblue", alpha=0.5, edgecolor="black", linewidth=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("DFT Total Energy (eV)")
plt.ylabel("Predicted Total Energy (eV)")
plt.title(f"{MATERIAL_NAME}: Linear Model — R²={r2:.4f}, RMSE={rmse:.4f} eV")
annot_lines = [f"R² = {r2:.4f}", f"RMSE = {rmse:.4f} eV"]
if U_val is not None:
    annot_lines.insert(0, f"U = {U_val:.2f} eV")
if J_val is not None:
    annot_lines.insert(1 if U_val is not None else 0, f"J = {J_val:.2f} eV")
plt.text(0.05, 0.93, "\n".join(annot_lines),
         transform=plt.gca().transAxes, fontsize=11, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{MATERIAL_NAME}_predicted_vs_actual.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to {OUTPUT_DIR}/{MATERIAL_NAME}_predicted_vs_actual.png")

# %% --- Step 7: Bayesian Ridge Regression -------------------------------------
print("\n" + "=" * 60)
print("Bayesian Ridge Regression")
print("=" * 60)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_s = scaler.transform(X)

bayes_model = BayesianRidge(
    n_iter=300, tol=1e-6, alpha_1=1e-6, alpha_2=1e-6,
    lambda_1=1e-6, lambda_2=1e-6, fit_intercept=True,
)
bayes_model.fit(X_train_s, y_train)

y_bayes_pred = bayes_model.predict(X_test_s)
r2_bayes = r2_score(y_test, y_bayes_pred)
rmse_bayes = np.sqrt(mean_squared_error(y_test, y_bayes_pred))

bayes_coeffs = pd.Series(bayes_model.coef_, index=feature_cols)
U_bayes = bayes_coeffs["hubbard_term"] * 2 if USE_HUBBARD else None
J_bayes = -bayes_coeffs["hund_term_M2"] * 4 if USE_HUND else None

print(f"Intercept:      {bayes_model.intercept_:.4f}")
print(f"Alpha (noise):  {bayes_model.alpha_:.4f}")
print(f"Lambda (prior): {bayes_model.lambda_:.4f}")
if USE_HUBBARD:
    print(f"U (eff):        {U_bayes:.4f} eV")
if USE_HUND:
    print(f"J (eff):        {J_bayes:.4f} eV")
print(f"R² (test):      {r2_bayes:.4f}")
print(f"RMSE (test):    {rmse_bayes:.4f} eV")

y_bayes_all, y_bayes_std = bayes_model.predict(X_s, return_std=True)

# %% --- Step 8: ARD Regression (Automatic Relevance Determination) -------------
print("\n" + "=" * 60)
print("ARD Regression (Automatic Relevance Determination)")
print("=" * 60)

ard_model = ARDRegression(
    n_iter=300, tol=1e-6, alpha_1=1e-6, alpha_2=1e-6,
    lambda_1=1e-6, lambda_2=1e-6,
    threshold_lambda=ARD_THRESHOLD_LAMBDA, fit_intercept=True,
)
ard_model.fit(X_train_s, y_train)

y_ard_pred = ard_model.predict(X_test_s)
r2_ard = r2_score(y_test, y_ard_pred)
rmse_ard = np.sqrt(mean_squared_error(y_test, y_ard_pred))

ard_coeffs = pd.Series(ard_model.coef_, index=feature_cols)

# Count how many features were kept (|coef| > small threshold)
n_active = (np.abs(ard_model.coef_) > 1e-6).sum()
n_pruned = len(feature_cols) - n_active

print(f"Intercept:      {ard_model.intercept_:.4f}")
print(f"Alpha (noise):  {ard_model.alpha_:.4f}")
print(f"Features kept:  {n_active} / {len(feature_cols)}  ({n_pruned} pruned to ~zero)")
print(f"R² (test):      {r2_ard:.4f}")
print(f"RMSE (test):    {rmse_ard:.4f} eV")

# Show top retained features by |coefficient|
print(f"\nTop retained features (by |coeff|):")
ard_sorted = ard_coeffs[np.abs(ard_coeffs) > 1e-6].abs().sort_values(ascending=False)
for name in ard_sorted.head(15).index:
    print(f"  {name:40s} {ard_coeffs[name]:+.6e}")

y_ard_all, y_ard_std = ard_model.predict(X_s, return_std=True)

# %% --- Step 9: Random Forest (Bagged) ----------------------------------------
print("\n" + "=" * 60)
print("Random Forest Regression (Bagged, with uncertainty from trees)")
print("=" * 60)

rf_model = RandomForestRegressor(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    max_leaf_nodes=RF_MAX_LEAF_NODES,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)  # RF doesn't need scaling

y_rf_pred = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_rf_pred))

# Uncertainty from individual tree predictions (bagging ensemble)
tree_preds = np.array([tree.predict(X) for tree in rf_model.estimators_])
y_rf_all = tree_preds.mean(axis=0)
y_rf_std = tree_preds.std(axis=0)

# Feature importances
rf_importances = pd.Series(rf_model.feature_importances_, index=feature_cols)

print(f"Trees:          {RF_N_ESTIMATORS}")
print(f"Max depth:      {RF_MAX_DEPTH}")
print(f"Max leaf nodes: {RF_MAX_LEAF_NODES}")
print(f"Min samples/leaf: {RF_MIN_SAMPLES_LEAF}")
print(f"R² (test):      {r2_rf:.4f}")
print(f"RMSE (test):    {rmse_rf:.4f} eV")
print(f"\nTop-10 feature importances:")
for name in rf_importances.nlargest(10).index:
    print(f"  {name:40s} {rf_importances[name]:.4f}")

# %% --- Step 10: Combined Summary ----------------------------------------------
print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)
print(f"{'Model':<25s} {'R²':>8s}  {'RMSE (eV)':>10s}  {'Features':>10s}")
print("-" * 60)
print(f"{'Linear Regression':<25s} {r2:>8.4f}  {rmse:>10.4f}  {len(feature_cols):>10d}")
print(f"{'Bayesian Ridge':<25s} {r2_bayes:>8.4f}  {rmse_bayes:>10.4f}  {len(feature_cols):>10d}")
print(f"{'ARD Regression':<25s} {r2_ard:>8.4f}  {rmse_ard:>10.4f}  {n_active:>10d}")
print(f"{'Random Forest':<25s} {r2_rf:>8.4f}  {rmse_rf:>10.4f}  {len(feature_cols):>10d}")

# %% --- Step 10: Bayesian Ridge Plot ------------------------------------------
plt.figure(figsize=(8, 8))
plt.errorbar(y, y_bayes_all, yerr=y_bayes_std,
             fmt="o", color="darkorange", alpha=0.4, ecolor="gray",
             elinewidth=0.5, capsize=0, markersize=4, markeredgewidth=0.2,
             markeredgecolor="black")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("DFT Total Energy (eV)")
plt.ylabel("Predicted Total Energy (eV)")
plt.title(f"{MATERIAL_NAME}: Bayesian Ridge — R²={r2_bayes:.4f}, RMSE={rmse_bayes:.4f} eV")
plt.text(0.05, 0.93, f"R² = {r2_bayes:.4f}\nRMSE = {rmse_bayes:.4f} eV",
         transform=plt.gca().transAxes, fontsize=11, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{MATERIAL_NAME}_bayesian_ridge.png", dpi=200, bbox_inches="tight")
plt.show()

# %% --- Step 11: ARD Plot -----------------------------------------------------
plt.figure(figsize=(8, 8))
plt.errorbar(y, y_ard_all, yerr=y_ard_std,
             fmt="o", color="seagreen", alpha=0.4, ecolor="gray",
             elinewidth=0.5, capsize=0, markersize=4, markeredgewidth=0.2,
             markeredgecolor="black")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("DFT Total Energy (eV)")
plt.ylabel("Predicted Total Energy (eV)")
plt.title(f"{MATERIAL_NAME}: ARD — R²={r2_ard:.4f}, RMSE={rmse_ard:.4f} eV  ({n_active}/{len(feature_cols)} features)")
plt.text(0.05, 0.93, f"R² = {r2_ard:.4f}\nRMSE = {rmse_ard:.4f} eV\n{n_active} / {len(feature_cols)} features",
         transform=plt.gca().transAxes, fontsize=11, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{MATERIAL_NAME}_ard.png", dpi=200, bbox_inches="tight")
plt.show()

# %% --- Step 12: Random Forest Plot -------------------------------------------
plt.figure(figsize=(8, 8))
plt.errorbar(y, y_rf_all, yerr=y_rf_std,
             fmt="o", color="mediumpurple", alpha=0.4, ecolor="gray",
             elinewidth=0.5, capsize=0, markersize=4, markeredgewidth=0.2,
             markeredgecolor="black")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("DFT Total Energy (eV)")
plt.ylabel("Predicted Total Energy (eV)")
plt.title(f"{MATERIAL_NAME}: Random Forest — R²={r2_rf:.4f}, RMSE={rmse_rf:.4f} eV  ({RF_N_ESTIMATORS} trees)")
plt.text(0.05, 0.93, f"R² = {r2_rf:.4f}\nRMSE = {rmse_rf:.4f} eV\n{RF_N_ESTIMATORS} trees",
         transform=plt.gca().transAxes, fontsize=11, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{MATERIAL_NAME}_random_forest.png", dpi=200, bbox_inches="tight")
plt.show()

# %% --- Step 13: LCB + Boltzmann for All Models -------------------------------
import matplotlib.colors as mcolors

models_lcb = {
    "Bayesian Ridge": (y_bayes_all, y_bayes_std),
    "ARD":            (y_ard_all, y_ard_std),
    "Random Forest":  (y_rf_all, y_rf_std),
}

lcb_data = {}
for name, (mu_i, sigma_i) in models_lcb.items():
    lcb_i = mu_i - LCB_BETA * sigma_i
    lcb_shifted = lcb_i - lcb_i.min()
    boltz_i = np.exp(-lcb_shifted * LCB_ETA)
    boltz_i /= boltz_i.sum()
    lcb_data[name] = {"mu": mu_i, "sigma": sigma_i, "lcb": lcb_i, "boltz": boltz_i}

    print(f"\n--- LCB ({name}, β={LCB_BETA}, η={LCB_ETA}) ---")
    print(f"LCB range:      [{lcb_i.min():.4f}, {lcb_i.max():.4f}] eV")
    print(f"Top-5 LCB calculations:")
    top5 = np.argsort(lcb_i)[:5]
    for rank, idx in enumerate(top5, 1):
        print(f"  #{rank} pk={fit_df.iloc[idx]['calculation_id']:>6s}  "
              f"E_DFT={y[idx]:.4f}  LCB={lcb_i[idx]:.4f}  σ={sigma_i[idx]:.4f}  "
              f"P_boltz={boltz_i[idx]:.4e}")

# 3×2 comparison plot (Bayesian Ridge | ARD | Random Forest)
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
y_min, y_max = y.min(), y.max()

for col, (name, data) in enumerate(lcb_data.items()):
    # LCB plot (top row)
    ax_lcb = axes[0, col]
    sc0 = ax_lcb.scatter(y, data["mu"], c=data["lcb"], cmap="viridis_r",
                         alpha=0.6, edgecolor="black", linewidth=0.15, s=18)
    ax_lcb.plot([y_min, y_max], [y_min, y_max], "r--", lw=1.5)
    ax_lcb.set_xlabel("DFT Total Energy (eV)")
    ax_lcb.set_ylabel("Predicted Mean (eV)")
    ax_lcb.set_title(f"{name}: LCB  (β={LCB_BETA})")
    ax_lcb.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
    plt.colorbar(sc0, ax=ax_lcb, label="LCB (eV)")

    # Boltzmann plot (bottom row)
    ax_boltz = axes[1, col]
    sc1 = ax_boltz.scatter(y, data["mu"], c=data["boltz"], cmap="inferno",
                           alpha=0.6, edgecolor="black", linewidth=0.15, s=18,
                           norm=mcolors.LogNorm())
    ax_boltz.plot([y_min, y_max], [y_min, y_max], "r--", lw=1.5)
    ax_boltz.set_xlabel("DFT Total Energy (eV)")
    ax_boltz.set_ylabel("Predicted Mean (eV)")
    ax_boltz.set_title(f"{name}: Boltzmann  (η={LCB_ETA})")
    ax_boltz.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
    plt.colorbar(sc1, ax=ax_boltz, label="P ∝ exp(−LCB · η)")

fig.suptitle(f"{MATERIAL_NAME}: LCB Acquisition Comparison  (β={LCB_BETA}, η={LCB_ETA})",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{MATERIAL_NAME}_lcb_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"\nLCB comparison plot saved to {OUTPUT_DIR}/{MATERIAL_NAME}_lcb_comparison.png")

# %%
