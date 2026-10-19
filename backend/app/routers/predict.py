"""Prediction endpoints: bioactivity, toxicity, drug-likeness."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.druglikeness import assess_druglikeness
from app.ml.explain import explain
from app.ml.llm_explain import build_context, generate_explanation
from app.ml.predict_bioactivity import load_bundle as load_bioactivity
from app.ml.predict_bioactivity import predict_bioactivity
from app.ml.predict_toxicity import FLAG_THRESHOLD, predict_toxicity
from app.ml.predict_toxicity import load_bundle as load_toxicity

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=500, examples=["CC(=O)Oc1ccccc1C(=O)O"])
    explain_top_n: int = Field(3, ge=0, le=13, description="SHAP breakdown for the N highest-scoring endpoints")
    top_k_features: int = Field(8, ge=1, le=25, description="Features listed per SHAP breakdown")
    include_llm_explanation: bool = Field(False, description="Opt in to sending derived model features to Gemini for a plain-English explanation")


class SmilesRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=500)


def _run(kind: str, req: PredictRequest, predict_fn, load_fn, positive_label: str, negative_label: str):
    try:
        scores = predict_fn(req.smiles)
        bundle = load_fn()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"{kind} model file missing; run its training script")

    metrics = bundle["metrics"].get("per_target", bundle["metrics"])
    predictions = []
    for ep, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        auc = metrics.get(ep, {}).get("roc_auc")
        predictions.append({
            "endpoint": ep,
            "description": bundle.get("descriptions", {}).get(ep),
            "model_score": round(p, 4),
            "prediction": positive_label if p >= FLAG_THRESHOLD else negative_label,
            "model_test_auc": round(auc, 3) if auc is not None else None,
        })
    explanations = [explain(kind, pr["endpoint"], req.smiles, req.top_k_features)
                    for pr in predictions[:req.explain_top_n]]
    result = {"smiles": req.smiles, "predictions": predictions, "explanations": explanations}
    if req.include_llm_explanation:
        top = predictions[0]
        top_expl = explanations[0] if explanations else explain(kind, top["endpoint"], req.smiles, req.top_k_features)
        others = [p["endpoint"] for p in predictions[1:] if p["prediction"] == positive_label]
        ctx = build_context(kind, top, top_expl, other_flagged=others)
        result["plain_english_explanation"] = generate_explanation(ctx)
    return result


@router.post("/bioactivity")
def bioactivity(req: PredictRequest):
    res = _run("bioactivity", req, predict_bioactivity, load_bioactivity, "active", "inactive")
    res["threshold"] = "active = IC50 <= 1 uM (pIC50 >= 6)"
    return res


@router.post("/toxicity")
def toxicity(req: PredictRequest):
    res = _run("toxicity", req, predict_toxicity, load_toxicity, "toxic", "non-toxic")
    flagged = [p["endpoint"] for p in res["predictions"] if p["prediction"] == "toxic"]
    res["flagged_endpoints"] = flagged
    res["overall"] = "toxicity alerts" if flagged else "no toxicity alerts"
    return res


@router.post("/druglikeness")
def druglikeness(req: SmilesRequest):
    try:
        return {"smiles": req.smiles, **assess_druglikeness(req.smiles)}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
