"""
train.py
LightGBM model with:
- Stratified K-Fold cross validation (5 fold)
- Class imbalance handling (scale_pos_weight)
- Early stopping
- Feature importance
- Model + metadata saved to models/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── LightGBM Parameters ──────────────────────────────────────────────────────
# These are well-tuned defaults for this dataset based on top Kaggle solutions
LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "n_estimators":     5000,
    "learning_rate":    0.02,
    "num_leaves":       34,
    "max_depth":        -1,
    "min_child_samples":    80,
    "min_child_weight":     40,
    "subsample":            0.85,
    "subsample_freq":       1,
    "colsample_bytree":     0.85,
    "reg_alpha":            0.1,
    "reg_lambda":           0.1,
    "min_split_gain":       0.02,
    "cat_smooth":           10,
    "random_state":         42,
    "n_jobs":               -1,
    "verbose":              -1,
}

N_FOLDS     = 5
THRESHOLD   = 0.5   # default — we tune this below


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    print("[1/5] Loading processed data ...")
    train = pd.read_csv(PROCESSED_DIR / "train_processed.csv")
    test  = pd.read_csv(PROCESSED_DIR / "test_processed.csv")

    feature_cols = [c for c in train.columns if c not in ("TARGET", "SK_ID_CURR")]

    # Drop any remaining object/string columns that slipped through preprocessing
    obj_cols_train = train[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    obj_cols_test  = test[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    drop_str_cols  = list(set(obj_cols_train + obj_cols_test))

    if drop_str_cols:
        print(f"   Dropping {len(drop_str_cols)} leftover string columns: {drop_str_cols}")
        feature_cols = [c for c in feature_cols if c not in drop_str_cols]

    # Verify no strings remain
    remaining_obj = train[feature_cols].select_dtypes(include=["object","category"]).columns.tolist()
    if remaining_obj:
        raise ValueError(f"Still has string columns: {remaining_obj}")

    X      = train[feature_cols].values.astype(np.float32)
    y      = train["TARGET"].values.astype(np.int8)
    X_test = test[feature_cols].values.astype(np.float32)
    ids    = test["SK_ID_CURR"].values

    print(f"   Final feature count: {len(feature_cols)}")

    print(f"   X_train : {X.shape}")
    print(f"   X_test  : {X_test.shape}")
    print(f"   Positive rate: {y.mean():.4f}  "
          f"(imbalance ratio: {(1-y.mean())/y.mean():.1f}:1)")
    return X, y, X_test, ids, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# 2. CROSS-VALIDATED TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_cv(X, y, X_test, feature_cols):
    print(f"\n[2/5] Training LightGBM — {N_FOLDS}-fold Stratified CV ...")

    skf            = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_preds      = np.zeros(len(X))          # out-of-fold predictions
    test_preds     = np.zeros(len(X_test))     # averaged test predictions
    fold_aucs      = []
    fold_models    = []

    # Class imbalance: weight negative class less
    neg_pos_ratio  = (y == 0).sum() / (y == 1).sum()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  ── Fold {fold}/{N_FOLDS} ──")

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        params = LGB_PARAMS.copy()
        params["scale_pos_weight"] = neg_pos_ratio

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set              = [(X_val, y_val)],
            eval_metric           = "auc",
            callbacks             = [
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        # OOF predictions
        val_pred          = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_aucs.append(fold_auc)
        print(f"   Fold {fold} AUC = {fold_auc:.5f}   "
              f"Best iter = {model.best_iteration_}")

        # Test predictions (average across folds)
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        fold_models.append(model)

    print(f"\n  ── CV Summary ──")
    print(f"  Fold AUCs     : {[round(a,5) for a in fold_aucs]}")
    print(f"  Mean AUC      : {np.mean(fold_aucs):.5f}")
    print(f"  Std  AUC      : {np.std(fold_aucs):.5f}")
    print(f"  OOF AUC       : {roc_auc_score(y, oof_preds):.5f}")
    print(f"  OOF Avg Prec  : {average_precision_score(y, oof_preds):.5f}")

    return fold_models, oof_preds, test_preds, fold_aucs


# ═══════════════════════════════════════════════════════════════════════════
# 3. THRESHOLD TUNING
# ═══════════════════════════════════════════════════════════════════════════

def tune_threshold(y_true, oof_preds):
    """
    Find threshold that maximises F1 on OOF predictions.
    Important for imbalanced datasets — default 0.5 is rarely optimal.
    """
    print("\n[3/5] Tuning classification threshold ...")
    thresholds  = np.arange(0.05, 0.95, 0.01)
    f1_scores   = []

    for t in thresholds:
        preds = (oof_preds >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        f1_scores.append(f1)

    best_idx       = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1        = f1_scores[best_idx]
    print(f"   Best threshold : {best_threshold:.2f}")
    print(f"   Best F1        : {best_f1:.4f}")
    return float(best_threshold)


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION REPORT
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(y_true, oof_preds, threshold, fold_aucs):
    print(f"\n[4/5] Full evaluation (threshold = {threshold:.2f}) ...")
    preds_binary = (oof_preds >= threshold).astype(int)

    print("\n  Classification Report:")
    print(classification_report(y_true, preds_binary,
                                 target_names=["No Default", "Default"],
                                 digits=4))

    cm = confusion_matrix(y_true, preds_binary)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix   : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Sensitivity/Recall : {tp/(tp+fn):.4f}  (defaults caught)")
    print(f"  Specificity        : {tn/(tn+fp):.4f}  (non-defaults correct)")
    print(f"  ROC-AUC (OOF)      : {roc_auc_score(y_true, oof_preds):.5f}")
    print(f"  Avg Precision      : {average_precision_score(y_true, oof_preds):.5f}")

    # Save metrics
    metrics = {
        "oof_roc_auc":        round(roc_auc_score(y_true, oof_preds), 5),
        "avg_precision":      round(average_precision_score(y_true, oof_preds), 5),
        "best_threshold":     round(threshold, 4),
        "fold_aucs":          [round(a, 5) for a in fold_aucs],
        "cv_mean_auc":        round(np.mean(fold_aucs), 5),
        "cv_std_auc":         round(np.std(fold_aucs), 5),
        "confusion_matrix":   {"TN": int(tn), "FP": int(fp),
                                "FN": int(fn), "TP": int(tp)},
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → models/metrics.json")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def save_feature_importance(fold_models, feature_cols):
    print("\n  Computing feature importance ...")

    # Average importance across folds
    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": np.mean([m.feature_importances_ for m in fold_models], axis=0)
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    # Plot top 40
    top40 = importance_df.head(40)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(top40["feature"][::-1], top40["importance"][::-1], color="#1a73e8")
    ax.set_xlabel("Average Importance (across 5 folds)")
    ax.set_title("Top 40 Feature Importances — Home Credit Default Model")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"  Top 5 features: {importance_df['feature'].head(5).tolist()}")
    print(f"  Importance chart → models/feature_importance.png")
    return importance_df


# ═══════════════════════════════════════════════════════════════════════════
# 6. SAVE ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════

def save_artifacts(fold_models, test_preds, ids, threshold,
                   feature_cols, metrics, importance_df):
    print("\n[5/5] Saving artefacts ...")

    # Save all fold models
    for i, m in enumerate(fold_models):
        joblib.dump(m, MODELS_DIR / f"lgbm_fold_{i+1}.pkl")

    # Save best single model (fold with highest AUC)
    best_idx = int(np.argmax(metrics["fold_aucs"]))
    joblib.dump(fold_models[best_idx], MODELS_DIR / "lgbm_best.pkl")
    print(f"   Best model = fold {best_idx+1} "
          f"(AUC={metrics['fold_aucs'][best_idx]})")

    # Save feature columns list
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    # Save threshold
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)

    # Save test submission
    submission = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET":     test_preds
    })
    submission.to_csv(MODELS_DIR / "submission.csv", index=False)
    print(f"   Submission saved → models/submission.csv")

    # Save OOF predictions
    oof_df = pd.read_csv(PROCESSED_DIR / "train_processed.csv",
                         usecols=["SK_ID_CURR", "TARGET"])
    oof_df["OOF_PRED"] = np.round(
        np.load(MODELS_DIR / "oof_preds.npy") if
        (MODELS_DIR / "oof_preds.npy").exists() else
        np.zeros(len(oof_df)), 6)
    print(f"   All artefacts saved to: {MODELS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_training():
    print("\n" + "="*60)
    print("  HOME CREDIT — LIGHTGBM TRAINING PIPELINE")
    print("="*60)

    X, y, X_test, ids, feature_cols = load_data()

    fold_models, oof_preds, test_preds, fold_aucs = train_cv(
        X, y, X_test, feature_cols)

    # Save OOF preds array for later use
    np.save(MODELS_DIR / "oof_preds.npy", oof_preds)
    np.save(MODELS_DIR / "test_preds.npy", test_preds)

    threshold = tune_threshold(y, oof_preds)
    metrics   = evaluate(y, oof_preds, threshold, fold_aucs)

    importance_df = save_feature_importance(fold_models, feature_cols)
    save_artifacts(fold_models, test_preds, ids, threshold,
                   feature_cols, metrics, importance_df)

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print(f"  OOF ROC-AUC : {metrics['oof_roc_auc']}")
    print(f"  CV Mean AUC : {metrics['cv_mean_auc']} ± {metrics['cv_std_auc']}")
    print(f"  Threshold   : {threshold:.2f}")
    print("="*60 + "\n")

    return fold_models, oof_preds, test_preds, metrics


if __name__ == "__main__":
    run_training()