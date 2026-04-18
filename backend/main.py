"""
main.py
FastAPI backend for Finexcore AI Lending Intelligence.
Endpoints:
  GET  /health          → health check
  GET  /model/info      → model metrics + top SHAP features
  POST /predict         → single applicant scoring
  POST /predict/batch   → batch portfolio scoring
  POST /predict/csv     → upload CSV for batch scoring
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import pandas as pd
import io

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent   # .../backend/
FRONTEND_FILE = _HERE.parent / "index.html"       # project root/index.html

from predict import (
    predict_single, predict_batch,
    METRICS, SHAP_TOP, FEAT_DESC,
    THRESHOLD, FEATURE_COLS,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Finexcore AI Lending Intelligence API",
    description = "Loan default prediction powered by LightGBM + SHAP",
    version     = "1.0.0",
    docs_url    = "/docs",
)

# ── CORS — allow frontend (any origin for now) ────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ApplicantInput(BaseModel):
    # Personal
    AGE_YEARS:            Optional[float] = Field(35,    description="Applicant age in years")
    CODE_GENDER:          Optional[float] = Field(0,     description="0=Male 1=Female")
    CNT_CHILDREN:         Optional[float] = Field(0,     description="Number of children")
    CNT_FAM_MEMBERS:      Optional[float] = Field(2,     description="Total family members")
    FLAG_OWN_CAR:         Optional[float] = Field(0,     description="1=owns car")
    FLAG_OWN_REALTY:      Optional[float] = Field(0,     description="1=owns realty")

    # Financial
    AMT_INCOME_TOTAL:     Optional[float] = Field(200000, description="Annual income")
    AMT_CREDIT:           Optional[float] = Field(500000, description="Loan amount")
    AMT_ANNUITY:          Optional[float] = Field(25000,  description="Monthly annuity")
    AMT_GOODS_PRICE:      Optional[float] = Field(450000, description="Goods price")
    EMPLOYED_YEARS:       Optional[float] = Field(5,      description="Years employed")
    IS_UNEMPLOYED:        Optional[float] = Field(0,      description="1=unemployed")

    # External credit scores
    EXT_SOURCE_1:         Optional[float] = Field(0.5,   description="External score 1 (0-1)")
    EXT_SOURCE_2:         Optional[float] = Field(0.5,   description="External score 2 (0-1)")
    EXT_SOURCE_3:         Optional[float] = Field(0.5,   description="External score 3 (0-1)")

    # Credit history
    INS_PAID_LATE_RATIO:  Optional[float] = Field(0.05,  description="Fraction of late installments")
    INS_DAYS_LATE_MAX:    Optional[float] = Field(5,     description="Max days late")
    PREV_APPROVED_RATIO:  Optional[float] = Field(0.7,   description="Previous approval rate")
    PREV_REFUSED_COUNT:   Optional[float] = Field(0,     description="Previous refusals")

    class Config:
        extra = "allow"   # accept any extra features too


class BatchInput(BaseModel):
    applicants: list[dict]


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def serve_frontend():
    """Serve the frontend index.html from the project root."""
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    raise HTTPException(status_code=404, detail="Frontend index.html not found")


@app.get("/health")
def health():
    return {"status": "ok", "model": "lgbm_best", "version": "1.0.0"}


@app.get("/model/info")
def model_info():
    """Returns model metrics, top SHAP features, threshold."""
    return {
        "metrics":       METRICS,
        "threshold":     THRESHOLD,
        "feature_count": len(FEATURE_COLS),
        "top_features":  SHAP_TOP[:15],
        "feature_descriptions": {
            k: v for k, v in list(FEAT_DESC.items())[:30]
        },
    }


@app.post("/predict")
def predict(applicant: ApplicantInput):
    """
    Score a single loan applicant.
    Returns probability, risk tier, decision, and top 10 SHAP factors.
    """
    try:
        input_dict = applicant.model_dump()

        # Auto-compute derived features from raw inputs
        income  = input_dict.get("AMT_INCOME_TOTAL", 1)
        credit  = input_dict.get("AMT_CREDIT", 1)
        annuity = input_dict.get("AMT_ANNUITY", 1)
        goods   = input_dict.get("AMT_GOODS_PRICE", 0)
        age     = input_dict.get("AGE_YEARS", 35)
        emp     = input_dict.get("EMPLOYED_YEARS", 0)
        unemp   = input_dict.get("IS_UNEMPLOYED", 0)
        ext1    = input_dict.get("EXT_SOURCE_1", 0.5)
        ext2    = input_dict.get("EXT_SOURCE_2", 0.5)
        ext3    = input_dict.get("EXT_SOURCE_3", 0.5)
        fam     = max(input_dict.get("CNT_FAM_MEMBERS", 1), 1)

        input_dict.update({
            "DAYS_BIRTH":            age * 365,
            "DAYS_EMPLOYED":         0 if unemp else emp * 365,
            "CREDIT_INCOME_RATIO":   credit / max(income, 1),
            "ANNUITY_INCOME_RATIO":  annuity / max(income, 1),
            "CREDIT_TERM":           annuity / max(credit, 1),
            "GOODS_CREDIT_RATIO":    goods / max(credit, 1),
            "INCOME_PER_PERSON":     income / fam,
            "EMPLOYED_TO_AGE_RATIO": 0 if unemp else emp / max(age, 1),
            "EXT_SOURCE_MEAN":       (ext1 + ext2 + ext3) / 3,
            "EXT_SOURCE_STD":        float(__import__("numpy").std([ext1, ext2, ext3])),
            "EXT_SOURCE_PRODUCT":    ext1 * ext2 * ext3,
            "EXT_SOURCE_MIN":        min(ext1, ext2, ext3),
            "INS_DAYS_LATE_MEAN":    input_dict.get("INS_DAYS_LATE_MAX", 0) * 0.3,
        })

        return predict_single(input_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def batch_json(payload: BatchInput):
    """
    Score multiple applicants from JSON array.
    Returns per-applicant scores + portfolio summary.
    """
    try:
        if len(payload.applicants) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Max 5000 applicants per batch request."
            )
        return predict_batch(payload.applicants)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv")
async def batch_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file of applicants.
    Returns per-applicant scores + portfolio summary.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted."
        )
    try:
        contents = await file.read()
        df       = pd.read_csv(io.BytesIO(contents))

        if len(df) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Max 10,000 rows per CSV upload."
            )

        records = df.to_dict(orient="records")
        return predict_batch(records)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))