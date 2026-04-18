"""
preprocess.py
Full preprocessing pipeline for Home Credit Default Risk dataset.
Handles: application_train/test, bureau, bureau_balance,
         previous_application, installments_payments,
         POS_CASH_balance, credit_card_balance
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def reduce_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Cast columns to smallest possible dtypes to save RAM."""
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object:
            df[col] = df[col].astype("category")
        elif col_type.kind in ("i", "u"):
            c_min, c_max = df[col].min(), df[col].max()
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
        elif col_type.kind == "f":
            for dtype in [np.float16, np.float32, np.float64]:
                if df[col].astype(dtype).max() < np.finfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print(f"   Memory reduced: {start_mem:.1f} MB → {end_mem:.1f} MB "
              f"({100*(start_mem-end_mem)/start_mem:.1f}% saved)")
    return df


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """Division that returns NaN instead of inf/0-div errors."""
    return np.where(b == 0, np.nan, a / b)


# ═══════════════════════════════════════════════════════════════════════════
# 2. SUPPLEMENTARY TABLE AGGREGATIONS
# ═══════════════════════════════════════════════════════════════════════════

def process_bureau(raw_dir: Path) -> pd.DataFrame:
    """
    Aggregate bureau.csv + bureau_balance.csv per SK_ID_CURR.
    Returns one row per applicant.
    """
    print("  Processing bureau + bureau_balance ...")
    buro = pd.read_csv(raw_dir / "bureau.csv")
    bb   = pd.read_csv(raw_dir / "bureau_balance.csv")

    # ── bureau_balance → per SK_ID_BUREAU aggregates ──────────────────────
    bb["STATUS_NUM"] = bb["STATUS"].map(
        {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    ).fillna(0)

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT  = ("MONTHS_BALANCE", "count"),
        BB_STATUS_MAX    = ("STATUS_NUM",      "max"),
        BB_STATUS_MEAN   = ("STATUS_NUM",      "mean"),
        BB_DPD_MONTHS    = ("STATUS_NUM",      lambda x: (x > 0).sum()),
    ).reset_index()

    # ── merge into bureau ──────────────────────────────────────────────────
    buro = buro.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # Fix sentinel: 365243 in DAYS columns → NaN
    for col in ["DAYS_CREDIT", "DAYS_CREDIT_ENDDATE",
                "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"]:
        buro[col] = buro[col].replace(365243, np.nan)

    # Derived
    buro["CREDIT_DURATION"] = buro["DAYS_CREDIT_ENDDATE"] - buro["DAYS_CREDIT"]
    buro["CREDIT_REPAID"]   = (buro["CREDIT_ACTIVE"] == "Closed").astype(int)
    buro["OVERDUE_RATIO"]   = safe_divide(
        buro["AMT_CREDIT_SUM_OVERDUE"], buro["AMT_CREDIT_SUM"])

    # ── aggregate to applicant level ───────────────────────────────────────
    num_aggs = {
        "DAYS_CREDIT":           ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE":    ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE":["max", "mean"],
        "AMT_CREDIT_SUM":        ["sum", "mean"],
        "AMT_CREDIT_SUM_DEBT":   ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE":["sum"],
        "AMT_ANNUITY":           ["sum", "mean"],
        "CNT_CREDIT_PROLONG":    ["sum"],
        "CREDIT_DURATION":       ["mean", "max"],
        "CREDIT_REPAID":         ["mean", "sum"],
        "OVERDUE_RATIO":         ["mean", "max"],
        "BB_MONTHS_COUNT":       ["sum"],
        "BB_STATUS_MAX":         ["max"],
        "BB_STATUS_MEAN":        ["mean"],
        "BB_DPD_MONTHS":         ["sum"],
    }
    buro_agg = buro.groupby("SK_ID_CURR").agg(num_aggs)
    buro_agg.columns = ["BURO_" + "_".join(c).upper()
                         for c in buro_agg.columns]
    buro_agg["BURO_CREDIT_COUNT"] = buro.groupby("SK_ID_CURR").size()
    buro_agg["BURO_ACTIVE_COUNT"] = (
        buro[buro["CREDIT_ACTIVE"] == "Active"]
        .groupby("SK_ID_CURR").size()
    )
    return reduce_memory(buro_agg.reset_index())


def process_previous_application(raw_dir: Path) -> pd.DataFrame:
    """Aggregate previous_application.csv per SK_ID_CURR."""
    print("  Processing previous_application ...")
    prev = pd.read_csv(raw_dir / "previous_application.csv")

    # Sentinel fix
    for col in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        prev[col] = prev[col].replace(365243, np.nan)

    prev["APP_CREDIT_RATIO"] = safe_divide(prev["AMT_APPLICATION"],
                                            prev["AMT_CREDIT"])
    prev["CREDIT_DOWN_RATIO"] = safe_divide(prev["AMT_DOWN_PAYMENT"],
                                             prev["AMT_CREDIT"])
    prev["APPROVED"]  = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["REFUSED"]   = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["LOAN_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_COUNT           = ("SK_ID_PREV",        "count"),
        PREV_APPROVED_COUNT  = ("APPROVED",           "sum"),
        PREV_REFUSED_COUNT   = ("REFUSED",            "sum"),
        PREV_APPROVED_RATIO  = ("APPROVED",           "mean"),
        PREV_AMT_CREDIT_MEAN = ("AMT_CREDIT",         "mean"),
        PREV_AMT_CREDIT_MAX  = ("AMT_CREDIT",         "max"),
        PREV_AMT_ANNUITY_MEAN= ("AMT_ANNUITY",        "mean"),
        PREV_APP_CREDIT_RATIO= ("APP_CREDIT_RATIO",   "mean"),
        PREV_LOAN_DIFF_MEAN  = ("LOAN_DIFF",          "mean"),
        PREV_DOWN_RATIO_MEAN = ("CREDIT_DOWN_RATIO",  "mean"),
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION",      "min"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION",     "mean"),
        PREV_RATE_DOWN_MEAN  = ("RATE_DOWN_PAYMENT",  "mean"),
        PREV_CNT_PAYMENT_MEAN= ("CNT_PAYMENT",        "mean"),
    ).reset_index()
    return reduce_memory(agg)


def process_installments(raw_dir: Path) -> pd.DataFrame:
    """Aggregate installments_payments.csv per SK_ID_CURR."""
    print("  Processing installments_payments ...")
    ins = pd.read_csv(raw_dir / "installments_payments.csv")

    ins["PAYMENT_DIFF"]  = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["PAYMENT_RATIO"] = safe_divide(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    ins["DAYS_LATE"]     = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["PAID_LATE"]     = (ins["DAYS_LATE"] > 0).astype(int)
    ins["PAID_FULL"]     = (ins["PAYMENT_DIFF"] <= 0).astype(int)

    agg = ins.groupby("SK_ID_CURR").agg(
        INS_COUNT            = ("NUM_INSTALMENT_NUMBER", "count"),
        INS_AMT_PAYMENT_SUM  = ("AMT_PAYMENT",           "sum"),
        INS_AMT_PAYMENT_MEAN = ("AMT_PAYMENT",           "mean"),
        INS_PAYMENT_DIFF_MEAN= ("PAYMENT_DIFF",          "mean"),
        INS_PAYMENT_DIFF_MAX = ("PAYMENT_DIFF",          "max"),
        INS_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO",         "mean"),
        INS_PAYMENT_RATIO_MIN= ("PAYMENT_RATIO",         "min"),
        INS_DAYS_LATE_MEAN   = ("DAYS_LATE",             "mean"),
        INS_DAYS_LATE_MAX    = ("DAYS_LATE",             "max"),
        INS_PAID_LATE_RATIO  = ("PAID_LATE",             "mean"),
        INS_PAID_FULL_RATIO  = ("PAID_FULL",             "mean"),
    ).reset_index()
    return reduce_memory(agg)


def process_pos_cash(raw_dir: Path) -> pd.DataFrame:
    """Aggregate POS_CASH_balance.csv per SK_ID_CURR."""
    print("  Processing POS_CASH_balance ...")
    pos = pd.read_csv(raw_dir / "POS_CASH_balance.csv")

    pos["DPD_POSITIVE"] = (pos["SK_DPD"] > 0).astype(int)

    agg = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT             = ("MONTHS_BALANCE",       "count"),
        POS_MONTHS_BALANCE_MAX= ("MONTHS_BALANCE",       "max"),
        POS_SK_DPD_MEAN       = ("SK_DPD",               "mean"),
        POS_SK_DPD_MAX        = ("SK_DPD",               "max"),
        POS_SK_DPD_DEF_MEAN   = ("SK_DPD_DEF",           "mean"),
        POS_SK_DPD_DEF_MAX    = ("SK_DPD_DEF",           "max"),
        POS_DPD_POSITIVE_RATIO= ("DPD_POSITIVE",         "mean"),
        POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT",       "mean"),
    ).reset_index()
    return reduce_memory(agg)


def process_credit_card(raw_dir: Path) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv per SK_ID_CURR."""
    print("  Processing credit_card_balance ...")
    cc = pd.read_csv(raw_dir / "credit_card_balance.csv")

    cc["UTILIZATION"] = safe_divide(cc["AMT_BALANCE"],
                                    cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["PAYMENT_RATIO"] = safe_divide(cc["AMT_PAYMENT_TOTAL_CURRENT"],
                                      cc["AMT_TOTAL_RECEIVABLE"])
    cc["DPD_POSITIVE"] = (cc["SK_DPD"] > 0).astype(int)

    agg = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT              = ("MONTHS_BALANCE",            "count"),
        CC_AMT_BALANCE_MEAN   = ("AMT_BALANCE",               "mean"),
        CC_AMT_BALANCE_MAX    = ("AMT_BALANCE",               "max"),
        CC_LIMIT_MEAN         = ("AMT_CREDIT_LIMIT_ACTUAL",   "mean"),
        CC_DRAWINGS_MEAN      = ("AMT_DRAWINGS_CURRENT",      "mean"),
        CC_DRAWINGS_MAX       = ("AMT_DRAWINGS_CURRENT",      "max"),
        CC_UTILIZATION_MEAN   = ("UTILIZATION",               "mean"),
        CC_UTILIZATION_MAX    = ("UTILIZATION",               "max"),
        CC_PAYMENT_RATIO_MEAN = ("PAYMENT_RATIO",             "mean"),
        CC_SK_DPD_MEAN        = ("SK_DPD",                    "mean"),
        CC_SK_DPD_MAX         = ("SK_DPD",                    "max"),
        CC_DPD_RATIO          = ("DPD_POSITIVE",              "mean"),
    ).reset_index()
    return reduce_memory(agg)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MAIN APPLICATION TABLE
# ═══════════════════════════════════════════════════════════════════════════

def process_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean + feature-engineer the main application table.
    Works on both train and test (TARGET column optional).
    """

    # ── 3a. Sentinel value fixes ───────────────────────────────────────────
    # DAYS_EMPLOYED = 365243 means "unemployed" — replace with NaN
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Negative day values are relative to application date — convert to abs
    for col in ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
                "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]:
        if col in df.columns:
            df[col] = df[col].abs()

    # ── 3b. Binary encode simple categoricals ─────────────────────────────
    df["CODE_GENDER"]     = df["CODE_GENDER"].map({"M": 0, "F": 1, "XNA": np.nan})
    df["FLAG_OWN_CAR"]    = df["FLAG_OWN_CAR"].map({"N": 0, "Y": 1})
    df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map({"N": 0, "Y": 1})

    # ── 3c. Feature engineering ────────────────────────────────────────────
    # Age in years
    df["AGE_YEARS"] = df["DAYS_BIRTH"] / 365

    # Employment length in years (NaN for unemployed)
    df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"] / 365

    # Credit burden
    df["CREDIT_INCOME_RATIO"]  = safe_divide(df["AMT_CREDIT"],
                                              df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = safe_divide(df["AMT_ANNUITY"],
                                              df["AMT_INCOME_TOTAL"])
    df["CREDIT_TERM"]          = safe_divide(df["AMT_ANNUITY"],
                                              df["AMT_CREDIT"])
    df["GOODS_CREDIT_RATIO"]   = safe_divide(df["AMT_GOODS_PRICE"],
                                              df["AMT_CREDIT"])
    df["INCOME_PER_PERSON"]    = safe_divide(df["AMT_INCOME_TOTAL"],
                                              df["CNT_FAM_MEMBERS"])

    # External source interactions (top predictors in this dataset)
    df["EXT_SOURCE_MEAN"]    = df[["EXT_SOURCE_1",
                                    "EXT_SOURCE_2",
                                    "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_STD"]     = df[["EXT_SOURCE_1",
                                    "EXT_SOURCE_2",
                                    "EXT_SOURCE_3"]].std(axis=1)
    df["EXT_SOURCE_PRODUCT"] = (df["EXT_SOURCE_1"].fillna(0)
                                 * df["EXT_SOURCE_2"].fillna(0)
                                 * df["EXT_SOURCE_3"].fillna(0))
    df["EXT_SOURCE_MIN"]     = df[["EXT_SOURCE_1",
                                    "EXT_SOURCE_2",
                                    "EXT_SOURCE_3"]].min(axis=1)

    # Employment stability
    df["EMPLOYED_TO_AGE_RATIO"] = safe_divide(df["EMPLOYED_YEARS"],
                                               df["AGE_YEARS"])
    df["IS_UNEMPLOYED"] = df["DAYS_EMPLOYED"].isna().astype(int)

    # Document submission score (how many docs provided)
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)

    # Social circle risk
    df["DEF_30_60_DIFF"] = (df["DEF_60_CNT_SOCIAL_CIRCLE"]
                             - df["DEF_30_CNT_SOCIAL_CIRCLE"]).clip(lower=0)

    # Credit bureau enquiry recency
    for col in ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
                "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
                "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
        df[col] = df[col].fillna(0)

    df["CREDIT_ENQUIRY_TOTAL"] = (
        df["AMT_REQ_CREDIT_BUREAU_HOUR"]
        + df["AMT_REQ_CREDIT_BUREAU_DAY"]
        + df["AMT_REQ_CREDIT_BUREAU_WEEK"]
        + df["AMT_REQ_CREDIT_BUREAU_MON"]
        + df["AMT_REQ_CREDIT_BUREAU_QRT"]
        + df["AMT_REQ_CREDIT_BUREAU_YEAR"]
    )

    # ── 3d. One-hot encode remaining categoricals ──────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Drop low-value ID-like categoricals
    drop_cats = ["WEEKDAY_APPR_PROCESS_START"]
    cat_cols  = [c for c in cat_cols if c not in drop_cats]

    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. ALIGN TRAIN / TEST COLUMNS
# ═══════════════════════════════════════════════════════════════════════════

def align_columns(train: pd.DataFrame,
                  test:  pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    After one-hot encoding, train and test may have different columns
    (some categories appear only in one split).
    This aligns them: adds missing columns as 0, drops test-only columns.
    TARGET is excluded from alignment.
    """
    target = train["TARGET"].copy()
    train  = train.drop(columns=["TARGET"])

    # Add columns missing in test
    for col in train.columns:
        if col not in test.columns:
            test[col] = 0

    # Add columns missing in train
    for col in test.columns:
        if col not in train.columns:
            train[col] = 0

    # Keep same column order
    test   = test[train.columns]
    train["TARGET"] = target
    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# 5. MISSING VALUE TREATMENT
# ═══════════════════════════════════════════════════════════════════════════

def handle_missing(df: pd.DataFrame,
                   train_medians: dict | None = None,
                   is_train:      bool        = True
                   ) -> tuple[pd.DataFrame, dict]:
    """
    Numerical: fill with MEDIAN (computed on train, applied to test).
    Binary/Flag columns (0/1 only): fill with 0.
    Returns (df_filled, medians_dict).
    """
    skip_cols = {"SK_ID_CURR", "TARGET"}
    medians   = {} if train_medians is None else train_medians

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in skip_cols]

    for col in num_cols:
        if df[col].isna().sum() == 0:
            continue

        # Flag columns — fill 0
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            df[col] = df[col].fillna(0)
            continue

        # Numeric — fill median
        if is_train:
            medians[col] = df[col].median()
        fill_val = medians.get(col, df[col].median())
        df[col] = df[col].fillna(fill_val)

    return df, medians


# ═══════════════════════════════════════════════════════════════════════════
# 6. OUTLIER CLIPPING
# ═══════════════════════════════════════════════════════════════════════════

def clip_outliers(df:            pd.DataFrame,
                  clip_bounds:   dict | None = None,
                  is_train:      bool        = True,
                  percentile:    float       = 99.5
                  ) -> tuple[pd.DataFrame, dict]:
    """
    Clip numerical features at (0.5th, 99.5th) percentile to remove
    extreme outliers. Bounds computed on train, applied to test.
    Skips TARGET, IDs, and binary columns.
    """
    skip_cols = {"SK_ID_CURR", "TARGET"}
    bounds    = {} if clip_bounds is None else clip_bounds

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in skip_cols]

    for col in num_cols:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            continue  # skip binary

        if is_train:
            lo = df[col].quantile(1 - percentile / 100)
            hi = df[col].quantile(percentile / 100)
            bounds[col] = (lo, hi)

        lo, hi = bounds.get(col, (df[col].min(), df[col].max()))
        df[col] = df[col].clip(lower=lo, upper=hi)

    return df, bounds


# ═══════════════════════════════════════════════════════════════════════════
# 7. DROP HIGH-MISSING-RATE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def drop_high_missing(df:          pd.DataFrame,
                      threshold:   float = 0.60,
                      keep_cols:   list  = None,
                      drop_list:   list  = None
                      ) -> tuple[pd.DataFrame, list]:
    """
    Drop columns where >threshold fraction of values are missing.
    keep_cols : columns to NEVER drop (e.g. TARGET, SK_ID_CURR).
    Returns (df, list_of_dropped_cols).
    If drop_list provided, use that list directly (for test set).
    """
    keep_cols = keep_cols or []
    if drop_list is not None:
        cols_to_drop = [c for c in drop_list if c in df.columns]
        return df.drop(columns=cols_to_drop), drop_list

    missing_ratio = df.isnull().mean()
    cols_to_drop  = missing_ratio[missing_ratio > threshold].index.tolist()
    cols_to_drop  = [c for c in cols_to_drop if c not in keep_cols]
    print(f"   Dropping {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing")
    return df.drop(columns=cols_to_drop), cols_to_drop


# ═══════════════════════════════════════════════════════════════════════════
# 8. FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("\n" + "="*60)
    print("  HOME CREDIT DEFAULT RISK — PREPROCESSING PIPELINE")
    print("="*60)

    # ── Load main tables ───────────────────────────────────────────────────
    print("\n[1/7] Loading application tables ...")
    train = pd.read_csv(RAW_DIR / "application_train.csv")
    test  = pd.read_csv(RAW_DIR / "application_test.csv")
    print(f"   Train shape: {train.shape}   Test shape: {test.shape}")
    print(f"   Target distribution:\n{train['TARGET'].value_counts(normalize=True).round(4)}")

    # ── Process supplementary tables ──────────────────────────────────────
    print("\n[2/7] Aggregating supplementary tables ...")
    buro_agg  = process_bureau(RAW_DIR)
    prev_agg  = process_previous_application(RAW_DIR)
    ins_agg   = process_installments(RAW_DIR)
    pos_agg   = process_pos_cash(RAW_DIR)
    cc_agg    = process_credit_card(RAW_DIR)

    # ── Merge all into train/test ──────────────────────────────────────────
    print("\n[3/7] Merging all tables ...")
    for agg_df in [buro_agg, prev_agg, ins_agg, pos_agg, cc_agg]:
        train = train.merge(agg_df, on="SK_ID_CURR", how="left")
        test  = test.merge(agg_df,  on="SK_ID_CURR", how="left")
    print(f"   Train after merge: {train.shape}   Test: {test.shape}")

    # ── Feature engineer application table ────────────────────────────────
    print("\n[4/7] Feature engineering application table ...")
    train = process_application(train)
    test  = process_application(test)
    print(f"   Train after FE: {train.shape}   Test: {test.shape}")

    # ── Align one-hot columns ─────────────────────────────────────────────
    print("\n[5/7] Aligning train/test columns ...")
    train, test = align_columns(train, test)
    print(f"   Final aligned shapes — Train: {train.shape}   Test: {test.shape}")

    # ── Drop high-missing columns ─────────────────────────────────────────
    print("\n[6/7] Dropping high-missing columns ...")
    train, dropped_cols = drop_high_missing(
        train, threshold=0.60, keep_cols=["TARGET", "SK_ID_CURR"])
    test,  _            = drop_high_missing(
        test,  threshold=0.60, keep_cols=["SK_ID_CURR"],
        drop_list=dropped_cols)

    # ── Replace any inf/-inf with NaN before imputation ───────────────────
    print("\n[6a/7] Replacing inf/-inf with NaN ...")
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace( [np.inf, -np.inf], np.nan, inplace=True)
    inf_check_train = np.isinf(train.select_dtypes(include=np.number)).sum().sum()
    inf_check_test  = np.isinf(test.select_dtypes( include=np.number)).sum().sum()
    print(f"   Remaining inf after replace — Train: {inf_check_train}  Test: {inf_check_test}")

    # ── Missing value imputation ───────────────────────────────────────────
    print("\n[6b/7] Imputing missing values ...")
    train, medians  = handle_missing(train, is_train=True)
    test,  _        = handle_missing(test,  train_medians=medians, is_train=False)

    # ── Outlier clipping ───────────────────────────────────────────────────
    print("\n[6c/7] Clipping outliers ...")
    train, clip_bounds = clip_outliers(train, is_train=True)
    test,  _           = clip_outliers(test,  clip_bounds=clip_bounds, is_train=False)

    print("\n[7/7] Reducing memory ...")
    train = reduce_memory(train)
    test  = reduce_memory(test)

    # Final safety — catch any inf reintroduced by dtype casting
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace( [np.inf, -np.inf], np.nan, inplace=True)
    # Re-impute the tiny number of NaNs that just appeared
    for col in train.select_dtypes(include=np.number).columns:
        if col in ("TARGET", "SK_ID_CURR"):
            continue
        if train[col].isna().any():
            fill = train[col].median()
            train[col] = train[col].fillna(fill)
    for col in test.select_dtypes(include=np.number).columns:
        if col == "SK_ID_CURR":
            continue
        if test[col].isna().any():
            fill = test[col].median()
            test[col] = test[col].fillna(fill)

    # ── Sanity checks ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL CHECKS")
    print("="*60)
    print(f"  Train shape      : {train.shape}")
    print(f"  Test shape       : {test.shape}")
    print(f"  Train NaN total  : {train.drop(columns=['TARGET','SK_ID_CURR']).isna().sum().sum()}")
    print(f"  Test  NaN total  : {test.drop(columns=['SK_ID_CURR']).isna().sum().sum()}")
    print(f"  Target balance   : {train['TARGET'].value_counts(normalize=True).round(4).to_dict()}")
    print(f"  Infinite values  : {np.isinf(train.select_dtypes(include=np.number)).sum().sum()}")

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n  Saving processed files ...")
    train.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    test.to_csv( PROCESSED_DIR / "test_processed.csv",  index=False)

    # Save medians + clip bounds for later use by Streamlit app
    import json
    meta = {
        "medians":      {k: float(v) for k, v in medians.items()},
        "clip_bounds":  {k: [float(lo), float(hi)]
                         for k, (lo, hi) in clip_bounds.items()},
        "dropped_cols": dropped_cols,
        "feature_cols": [c for c in train.columns
                         if c not in ("TARGET", "SK_ID_CURR")],
    }
    with open(PROCESSED_DIR / "pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to: {PROCESSED_DIR}")
    print("  DONE.\n")

    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train, test = run_pipeline()