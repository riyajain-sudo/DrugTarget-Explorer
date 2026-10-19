"""Plain-English explanations of predictions, grounded in the SHAP output.

The LLM (Gemini) only receives structured data - probability, model reliability and
the top SHAP features - and is instructed to use nothing else. If the API key is
missing or the call fails, a deterministic template explanation is returned instead,
so the API never breaks because of the LLM.

Run: python -m app.ml.llm_explain "<SMILES>" [bioactivity|toxicity]
"""
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TIMEOUT_MS = 30_000

SYSTEM_PROMPT = """You explain machine-learning predictions about molecules to a non-expert.
You are given structured data from a model and its SHAP feature contributions. Write 2-3 plain-English sentences.

Strict rules:
- Use ONLY the data provided. Do not add biological mechanisms, drug facts, disease links or
  chemistry knowledge from memory. If the data does not say why a substructure matters, do not guess.
- Describe features as "the model associates X with Y", never as proven causes.
- "increases" means the feature pushed the prediction toward the positive class; "decreases" means away from it.
- Refer to substructures by their SMILES fragment, e.g. "a c-N-c linkage (cNc)", and to descriptors by value.
- Call the 0-1 output an uncalibrated model score, not a probability of correctness. A test ROC AUC
  describes ranking discrimination only; do not describe it as case-level certainty or reliability.
- The FINAL sentence must always be exactly: "This is a computational prediction, not a lab result."
- No headings, bullet points or markdown. 2-3 sentences total, including that final sentence."""


def build_context(kind: str, prediction: dict, explanation: dict, other_flagged: list[str] | None = None,
                  druglikeness: dict | None = None) -> dict:
    """The only data the LLM sees."""
    ctx = {
        "prediction_type": kind,
        "endpoint": prediction["endpoint"],
        "endpoint_description": prediction.get("description"),
        "predicted_class": prediction["prediction"],
        "uncalibrated_model_score": prediction["model_score"],
        "shap_base_logit": round(explanation["base_logit"], 3),
        "explanation_scope": explanation["contribution_view"],
        "omitted_absent_bit_shap_mass": round(explanation["omitted_abs_shap_mass"], 3),
        "model_test_auc": prediction.get("model_test_auc"),
        "top_shap_features": [
            {"feature": c["label"], "effect": c["direction"], "shap_log_odds": round(c["shap"], 3)}
            for c in explanation["contributions"]
        ],
    }
    if other_flagged:
        ctx["other_endpoints_also_flagged"] = other_flagged
    if druglikeness:
        ctx["drug_likeness"] = druglikeness["summary"]
    return ctx


def template_explanation(ctx: dict) -> str:
    """Deterministic fallback built purely from the same structured data."""
    feats = ctx["top_shap_features"]
    ups = [f["feature"] for f in feats if f["effect"] == "increases"][:2]
    downs = [f["feature"] for f in feats if f["effect"] == "decreases"][:1]
    s = (f"The model labels this molecule {ctx['predicted_class']} for {ctx['endpoint']} "
         f"(uncalibrated model score {ctx['uncalibrated_model_score']:.0%}; SHAP base logit "
         f"{ctx['shap_base_logit']:.3f}, not an average-molecule probability).")
    if ups:
        s += f" The strongest features raising this score were: {'; '.join(ups)}."
    if downs:
        s += f" Lowering it: {downs[0]}."
    return s + " This is a computational prediction, not a lab result."


def _client():
    from google import genai
    from google.genai import types
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS)), types


def generate_explanation(ctx: dict) -> dict:
    """Return {"text": ..., "source": "gemini" | "template"}."""
    try:
        client, types = _client()
        resp = client.models.generate_content(
            model=MODEL,
            contents="Explain this prediction:\n" + json.dumps(ctx, indent=2),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.2, max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("empty response")
        return {"text": text, "source": "gemini", "model": MODEL}
    except Exception as e:  # never let the LLM break the prediction API
        log.warning("Gemini explanation failed (%s: %s); using template", type(e).__name__, e)
        return {"text": template_explanation(ctx), "source": "template", "model": None}


def main():
    from app.ml.explain import explain
    from app.ml.predict_bioactivity import predict_bioactivity
    from app.ml.predict_toxicity import predict_toxicity

    smiles = sys.argv[1] if len(sys.argv) > 1 else "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
    kind = sys.argv[2] if len(sys.argv) > 2 else "bioactivity"
    try:
        scores = (predict_bioactivity if kind == "bioactivity" else predict_toxicity)(smiles)
    except ValueError as e:
        sys.exit(str(e))
    ep, p = max(scores.items(), key=lambda kv: kv[1])
    pos, neg = ("active", "inactive") if kind == "bioactivity" else ("toxic", "non-toxic")
    pred = {"endpoint": ep, "model_score": round(p, 4), "prediction": pos if p >= 0.5 else neg}
    ctx = build_context(kind, pred, explain(kind, ep, smiles, 6))
    print("--- data sent to the LLM ---\n" + json.dumps(ctx, indent=2))
    out = generate_explanation(ctx)
    print(f"\n--- explanation ({out['source']}) ---\n{out['text']}")


if __name__ == "__main__":
    main()
