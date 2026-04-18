"""
predict.py
Loads model + metadata once at startup.
Handles single prediction and batch scoring with SHAP.
"""

import json
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path

# ── Paths — resolve relative to project root regardless of where uvicorn runs ──
_HERE         = Path(__file__).resolve().parent        # .../backend/
ROOT          = _HERE.parent                           # .../finexcore-loan-default/
MODELS_DIR    = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Load everything once at startup ───────────────────────────────────────────
print("Loading model...")
MODEL = joblib.load(MODELS_DIR / "lgbm_best.pkl")

print("Loading metadata...")
with open(MODELS_DIR / "feature_cols.json")       as f: FEATURE_COLS  = json.load(f)
with open(MODELS_DIR / "threshold.json")           as f: THRESHOLD     = json.load(f)["threshold"]
with open(MODELS_DIR / "metrics.json")             as f: METRICS       = json.load(f)
with open(MODELS_DIR / "shap_feature_desc.json")   as f: FEAT_DESC     = json.load(f)
with open(PROCESSED_DIR / "pipeline_meta.json")    as f: PIPELINE_META = json.load(f)

SHAP_TOP = pd.read_csv(MODELS_DIR / "shap_top20.csv").to_dict(orient="records")

print("Loading SHAP explainer...")
EXPLAINER = shap.TreeExplainer(MODEL)

print("All loaded. Ready.")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(input_dict: dict) -> np.ndarray:
    """
    Takes a raw input dict, fills missing with training medians,
    clips with training bounds, returns float32 array.
    """
    row = {feat: input_dict.get(feat, np.nan) for feat in FEATURE_COLS}
    df  = pd.DataFrame([row])

    # Impute missing
    for col, med in PIPELINE_META["medians"].items():
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(med)

    # Clip outliers
    for col, (lo, hi) in PIPELINE_META["clip_bounds"].items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # Final fillna safety
    df = df.fillna(0)

    return df[FEATURE_COLS].values.astype(np.float32)


def risk_tier(prob: float) -> str:
    if prob >= THRESHOLD:
        return "HIGH"
    elif prob >= THRESHOLD * 0.6:
        return "MEDIUM"
    return "LOW"


def decision(prob: float) -> str:
    if prob >= THRESHOLD:             return "DECLINE"
    elif prob >= THRESHOLD * 0.6:     return "REVIEW"
    return "APPROVE"


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(input_dict: dict) -> dict:
    X    = build_feature_vector(input_dict)
    prob = float(MODEL.predict_proba(X)[0, 1])

    # SHAP
    sv = EXPLAINER.shap_values(X)
    sv = sv[1][0] if isinstance(sv, list) else sv[0]

    shap_factors = []
    indices = np.argsort(np.abs(sv))[::-1][:10]
    for i in indices:
        feat = FEATURE_COLS[i]
        shap_factors.append({
            "feature":     feat,
            "description": FEAT_DESC.get(feat, feat.replace("_", " ").title()),
            "value":       round(float(X[0, i]), 4),
            "shap":        round(float(sv[i]), 6),
            "direction":   "increases_risk" if sv[i] > 0 else "reduces_risk",
        })

    return {
        "probability":   round(prob, 6),
        "probability_pct": round(prob * 100, 2),
        "risk_tier":     risk_tier(prob),
        "decision":      decision(prob),
        "threshold":     round(THRESHOLD, 4),
        "credit_score":  max(300, min(850, int(850 - prob * 600))),
        "shap_factors":  shap_factors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_batch(records: list[dict]) -> dict:
    """
    Takes a list of applicant dicts.
    Returns scored results + portfolio summary.
    """
    if not records:
        return {"error": "No records provided"}

    # Build feature matrix
    rows = []
    for rec in records:
        row = {feat: rec.get(feat, np.nan) for feat in FEATURE_COLS}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Impute
    for col, med in PIPELINE_META["medians"].items():
        if col in df.columns:
            df[col] = df[col].fillna(med)

    # Clip
    for col, (lo, hi) in PIPELINE_META["clip_bounds"].items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # Drop string cols, fill remaining NaN
    obj_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    valid    = [c for c in FEATURE_COLS if c in df.columns and c not in obj_cols]
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            df[feat] = PIPELINE_META["medians"].get(feat, 0)
    df = df.fillna(0)

    X     = df[FEATURE_COLS].values.astype(np.float32)
    probs = MODEL.predict_proba(X)[:, 1]

    # Build results
    results = []
    for i, (rec, prob) in enumerate(zip(records, probs)):
        prob = float(prob)
        results.append({
            "id":          rec.get("SK_ID_CURR", i + 1),
            "probability": round(prob, 4),
            "risk_tier":   risk_tier(prob),
            "decision":    decision(prob),
        })

    # Portfolio summary
    probs_arr   = np.array([r["probability"] for r in results])
    approve_n   = sum(1 for r in results if r["decision"] == "APPROVE")
    review_n    = sum(1 for r in results if r["decision"] == "REVIEW")
    decline_n   = sum(1 for r in results if r["decision"] == "DECLINE")
    high_n      = sum(1 for r in results if r["risk_tier"] == "HIGH")
    medium_n    = sum(1 for r in results if r["risk_tier"] == "MEDIUM")
    low_n       = sum(1 for r in results if r["risk_tier"] == "LOW")

    return {
        "total":        len(results),
        "summary": {
            "approve_count":   approve_n,
            "review_count":    review_n,
            "decline_count":   decline_n,
            "approve_pct":     round(approve_n / len(results) * 100, 1),
            "decline_pct":     round(decline_n / len(results) * 100, 1),
            "mean_probability":round(float(probs_arr.mean()), 4),
            "high_risk_count": high_n,
            "medium_risk_count": medium_n,
            "low_risk_count":  low_n,
        },
        "results": results,
    }