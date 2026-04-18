"""
explain.py
SHAP explainability for the Home Credit LightGBM model.
Generates:
  - models/shap_values.npy         (full OOF SHAP matrix, sampled)
  - models/shap_expected_value.npy (base rate)
  - models/shap_summary.png        (beeswarm plot)
  - models/shap_top20.csv          (mean |SHAP| per feature)
  - models/shap_feature_desc.json  (human-readable feature descriptions)
"""

import json
import warnings
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")

# ── Human-readable descriptions for top features shown in dashboard ──────────
FEATURE_DESCRIPTIONS = {
    # Application features
    "EXT_SOURCE_MEAN":           "Average external credit score (3 bureaus)",
    "EXT_SOURCE_1":              "External credit score — source 1",
    "EXT_SOURCE_2":              "External credit score — source 2",
    "EXT_SOURCE_3":              "External credit score — source 3",
    "EXT_SOURCE_PRODUCT":        "Combined product of all 3 external scores",
    "EXT_SOURCE_MIN":            "Lowest of the 3 external credit scores",
    "CREDIT_TERM":               "Monthly repayment as fraction of total credit",
    "CREDIT_INCOME_RATIO":       "Total credit amount relative to annual income",
    "ANNUITY_INCOME_RATIO":      "Monthly annuity payment relative to income",
    "GOODS_CREDIT_RATIO":        "Goods price relative to credit amount",
    "INCOME_PER_PERSON":         "Income divided by number of family members",
    "AGE_YEARS":                 "Applicant age in years",
    "EMPLOYED_YEARS":            "Years at current employer",
    "EMPLOYED_TO_AGE_RATIO":     "Employment length relative to age",
    "IS_UNEMPLOYED":             "Whether applicant is currently unemployed",
    "DAYS_BIRTH":                "Days since applicant birth (age proxy)",
    "DAYS_EMPLOYED":             "Days since employment started",
    "DAYS_REGISTRATION":         "Days since registration document change",
    "DAYS_ID_PUBLISH":           "Days since ID was last changed",
    "DAYS_LAST_PHONE_CHANGE":    "Days since phone number was changed",
    "AMT_CREDIT":                "Total loan credit amount applied for",
    "AMT_ANNUITY":               "Monthly loan annuity payment",
    "AMT_INCOME_TOTAL":          "Total annual income of applicant",
    "AMT_GOODS_PRICE":           "Price of goods for the loan",
    "CNT_CHILDREN":              "Number of children",
    "CNT_FAM_MEMBERS":           "Total family members",
    "REGION_RATING_CLIENT":      "Rating of region where client lives",
    "CREDIT_ENQUIRY_TOTAL":      "Total credit bureau enquiries (all periods)",
    "DOCUMENT_COUNT":            "Number of documents submitted",
    # Bureau features
    "BURO_CREDIT_COUNT":         "Total number of previous credit bureau records",
    "BURO_ACTIVE_COUNT":         "Number of currently active bureau credits",
    "BURO_AMT_CREDIT_SUM_SUM":   "Total outstanding debt across all bureau credits",
    "BURO_AMT_CREDIT_SUM_DEBT_SUM": "Total current debt in bureau records",
    "BURO_DAYS_CREDIT_MIN":      "Most recent bureau credit (days ago)",
    "BURO_CREDIT_REPAID_MEAN":   "Fraction of bureau credits fully repaid",
    "BURO_OVERDUE_RATIO_MAX":    "Worst overdue ratio across bureau credits",
    "BURO_BB_STATUS_MAX":        "Worst delinquency status in bureau balance",
    "BURO_BB_DPD_MONTHS_SUM":    "Total months with days-past-due in bureau",
    # Previous application features
    "PREV_COUNT":                "Number of previous loan applications",
    "PREV_APPROVED_RATIO":       "Fraction of previous applications approved",
    "PREV_REFUSED_COUNT":        "Number of previously refused applications",
    "PREV_AMT_CREDIT_MAX":       "Highest previous loan amount",
    "PREV_LOAN_DIFF_MEAN":       "Average gap between applied and approved amount",
    "PREV_DAYS_DECISION_MIN":    "Days since most recent previous application",
    # Installment features
    "INS_PAID_LATE_RATIO":       "Fraction of installments paid late",
    "INS_DAYS_LATE_MAX":         "Worst late payment (days)",
    "INS_DAYS_LATE_MEAN":        "Average days late on installment payments",
    "INS_PAYMENT_RATIO_MIN":     "Worst payment ratio (paid vs owed)",
    "INS_PAYMENT_DIFF_MAX":      "Largest underpayment on installments",
    # POS Cash features
    "POS_SK_DPD_MAX":            "Worst days-past-due on POS/cash loans",
    "POS_SK_DPD_MEAN":           "Average days-past-due on POS/cash loans",
    "POS_DPD_POSITIVE_RATIO":    "Fraction of months with any DPD on POS loans",
    # Credit card features
    "CC_UTILIZATION_MAX":        "Peak credit card utilization rate",
    "CC_UTILIZATION_MEAN":       "Average credit card utilization rate",
    "CC_SK_DPD_MAX":             "Worst days-past-due on credit card",
    "CC_DPD_RATIO":              "Fraction of months with credit card DPD",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL + DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_data():
    print("[1/4] Loading model and data ...")

    model = joblib.load(MODELS_DIR / "lgbm_best.pkl")

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # Load a stratified sample for SHAP (full 307k is too slow)
    train = pd.read_csv(PROCESSED_DIR / "train_processed.csv")

    # Drop string columns same as in train.py
    obj_cols = train[feature_cols].select_dtypes(
        include=["object", "category"]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in obj_cols]

    # Sample: 3000 defaults + 3000 non-defaults = 6000 rows
    # This gives representative SHAP without taking 30 mins
    df_pos = train[train["TARGET"] == 1].sample(
        n=min(3000, (train["TARGET"] == 1).sum()), random_state=42)
    df_neg = train[train["TARGET"] == 0].sample(
        n=min(3000, (train["TARGET"] == 0).sum()), random_state=42)
    df_sample = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42)

    X_sample = df_sample[feature_cols].values.astype(np.float32)
    y_sample = df_sample["TARGET"].values

    print(f"   Model loaded: best fold")
    print(f"   Features    : {len(feature_cols)}")
    print(f"   SHAP sample : {X_sample.shape}  "
          f"(defaults: {y_sample.sum()}  non-defaults: {(y_sample==0).sum()})")

    return model, feature_cols, X_sample, y_sample, df_sample


# ═══════════════════════════════════════════════════════════════════════════
# 2. COMPUTE SHAP VALUES
# ═══════════════════════════════════════════════════════════════════════════

def compute_shap(model, X_sample, feature_cols):
    print("\n[2/4] Computing SHAP values ...")
    print("   Using TreeExplainer (fast, exact for tree models) ...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary classification returns list [neg_class, pos_class]
    # We want the positive class (default = 1)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[1])
    else:
        expected_value = float(expected_value)

    print(f"   SHAP matrix shape : {shap_vals.shape}")
    print(f"   Expected value    : {expected_value:.4f}  "
          f"(base log-odds of default)")

    # Save raw SHAP values
    np.save(MODELS_DIR / "shap_values.npy",          shap_vals)
    np.save(MODELS_DIR / "shap_expected_value.npy",  np.array([expected_value]))

    return shap_vals, expected_value


# ═══════════════════════════════════════════════════════════════════════════
# 3. SHAP SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

def build_shap_summary(shap_vals, feature_cols):
    """
    Build a ranked DataFrame of mean |SHAP| per feature.
    This is what the Streamlit dashboard uses to show
    'Why was this applicant flagged?'
    """
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":        feature_cols,
        "mean_abs_shap":  mean_abs_shap,
        "description":    [FEATURE_DESCRIPTIONS.get(f, f.replace("_", " ").title())
                           for f in feature_cols],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_df.to_csv(MODELS_DIR / "shap_top20.csv", index=False)
    print(f"\n   Top 10 SHAP features:")
    for _, row in shap_df.head(10).iterrows():
        print(f"   {row['feature']:45s}  {row['mean_abs_shap']:.5f}  "
              f"— {row['description']}")
    return shap_df


# ═══════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(shap_vals, X_sample, feature_cols, shap_df):
    print("\n[3/4] Generating SHAP plots ...")

    # ── 4a. Beeswarm summary plot ──────────────────────────────────────────
    top_n     = 20
    top_feats = shap_df["feature"].head(top_n).tolist()
    top_idx   = [feature_cols.index(f) for f in top_feats]

    shap_top   = shap_vals[:, top_idx]
    X_top      = X_sample[:, top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_top, X_top,
        feature_names=top_feats,
        show=False, plot_size=None,
        color_bar_label="Feature Value",
    )
    plt.title("SHAP Summary — Top 20 Features\n"
              "Red = high feature value  |  Blue = low feature value",
              fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Beeswarm plot  → models/shap_summary.png")

    # ── 4b. Mean |SHAP| bar chart ──────────────────────────────────────────
    top20 = shap_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top20["feature"][::-1],
                   top20["mean_abs_shap"][::-1],
                   color="#e63946")
    ax.set_xlabel("Mean |SHAP Value| (average impact on model output)")
    ax.set_title("Top 20 Features by SHAP Importance\n"
                 "Home Credit Default Risk Model", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    # Add value labels
    for bar, val in zip(bars, top20["mean_abs_shap"][::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Bar chart      → models/shap_bar.png")

    # ── 4c. EXT_SOURCE_MEAN dependence plot ───────────────────────────────
    if "EXT_SOURCE_MEAN" in feature_cols:
        idx = feature_cols.index("EXT_SOURCE_MEAN")
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            X_sample[:, idx],
            shap_vals[:, idx],
            c=X_sample[:, idx],
            cmap="RdYlGn", alpha=0.4, s=8,
        )
        plt.colorbar(sc, ax=ax, label="EXT_SOURCE_MEAN value")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("EXT_SOURCE_MEAN (average external credit score)")
        ax.set_ylabel("SHAP value (impact on default probability)")
        ax.set_title("SHAP Dependence — External Credit Score\n"
                     "Higher score = lower default risk (negative SHAP = good)",
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "shap_dependence_ext_source.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("   Dependence plot → models/shap_dependence_ext_source.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. SAVE FEATURE DESCRIPTIONS JSON
# ═══════════════════════════════════════════════════════════════════════════

def save_feature_descriptions(feature_cols):
    desc = {f: FEATURE_DESCRIPTIONS.get(f, f.replace("_", " ").title())
            for f in feature_cols}
    with open(MODELS_DIR / "shap_feature_desc.json", "w") as f:
        json.dump(desc, f, indent=2)
    print("   Feature descriptions → models/shap_feature_desc.json")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_explain():
    print("\n" + "="*60)
    print("  HOME CREDIT — SHAP EXPLAINABILITY PIPELINE")
    print("="*60)

    model, feature_cols, X_sample, y_sample, df_sample = load_model_and_data()
    shap_vals, expected_value = compute_shap(model, X_sample, feature_cols)
    shap_df = build_shap_summary(shap_vals, feature_cols)

    generate_plots(shap_vals, X_sample, feature_cols, shap_df)

    print("\n[4/4] Saving feature descriptions ...")
    save_feature_descriptions(feature_cols)

    print("\n" + "="*60)
    print("  SHAP COMPLETE")
    print(f"  Top feature by SHAP : {shap_df.iloc[0]['feature']}")
    print(f"  Description         : {shap_df.iloc[0]['description']}")
    print("="*60 + "\n")

    return shap_vals, shap_df, feature_cols


if __name__ == "__main__":
    run_explain()