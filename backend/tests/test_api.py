from fastapi.testclient import TestClient
from app.ml.llm_explain import build_context, template_explanation
from app.routers.predict import PredictRequest

from app.main import app

client = TestClient(app)


def test_llm_is_opt_in_and_baseline_context_is_raw_logit():
    assert PredictRequest(smiles="CCO").include_llm_explanation is False
    context = build_context(
        "bioactivity",
        {"endpoint": "EGFR", "model_score": 0.8, "prediction": "active"},
        {"base_logit": 0.3, "contribution_view": "filtered heuristic",
         "omitted_abs_shap_mass": 1.2, "contributions": []},
    )
    assert context["shap_base_logit"] == 0.3
    assert "baseline_probability_for_average_molecule" not in context
    text = template_explanation(context)
    assert "uncalibrated model score" in text
    assert "not an average-molecule probability" in text


def test_readiness_status_tracks_file_presence_only(monkeypatch, tmp_path):
    paths = [tmp_path / "bio.pkl", tmp_path / "tox.pkl", tmp_path / "repurpose.db"]
    monkeypatch.setattr("app.ml.predict_bioactivity.MODEL_PATH", paths[0])
    monkeypatch.setattr("app.ml.predict_toxicity.MODEL_PATH", paths[1])
    monkeypatch.setattr("app.db.models.DB_PATH", paths[2])

    response = client.get("/ready")
    assert response.status_code == 503
    report = response.json()
    assert set(report) == {"ready", "artifacts_present"}
    assert set(report["artifacts_present"]) == {
        "bioactivity_model_file", "toxicity_model_file", "repurposing_database_file"
    }
    assert all(isinstance(value, bool) for value in report["artifacts_present"].values())
    assert report["ready"] is False
    assert report["artifacts_present"] == {
        "bioactivity_model_file": False,
        "toxicity_model_file": False,
        "repurposing_database_file": False,
    }

    for path in paths:
        path.write_text("presence fixture only")
    response = client.get("/ready")
    assert response.status_code == 200
    report = response.json()
    assert report["ready"] is True
    assert all(report["artifacts_present"].values())
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ERLOTINIB = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
BISPHENOL_A = "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_bioactivity():
    r = client.post("/predict/bioactivity", json={"smiles": ERLOTINIB, "include_llm_explanation": False})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 6
    top = body["predictions"][0]
    assert top["endpoint"] == "EGFR" and top["prediction"] == "active"
    assert 0 <= top["model_score"] <= 1 and "confidence" not in top
    assert len(body["explanations"]) == 3 and body["explanations"][0]["contributions"]


def test_toxicity():
    r = client.post("/predict/toxicity", json={"smiles": BISPHENOL_A, "explain_top_n": 1, "include_llm_explanation": False})
    assert r.status_code == 200
    body = r.json()
    assert "NR-ER" in body["flagged_endpoints"] and len(body["explanations"]) == 1


def test_druglikeness():
    body = client.post("/predict/druglikeness", json={"smiles": ASPIRIN}).json()
    assert body["drug_like"] is True and 0 <= body["qed"] <= 1


def test_invalid_smiles_is_422():
    for path in ("bioactivity", "toxicity", "druglikeness"):
        r = client.post(f"/predict/{path}", json={"smiles": "not_a_smiles"})
        assert r.status_code == 422 and "Invalid SMILES" in r.json()["detail"]


def test_validation_errors():
    assert client.post("/predict/toxicity", json={}).status_code == 422
    assert client.post("/predict/toxicity", json={"smiles": ""}).status_code == 422


def test_llm_explanation_is_grounded_and_included(monkeypatch):
    seen = {}

    def fake_generate(ctx):
        seen["ctx"] = ctx
        return {"text": "stub explanation", "source": "gemini", "model": "stub"}

    monkeypatch.setattr("app.routers.predict.generate_explanation", fake_generate)
    body = client.post("/predict/bioactivity", json={"smiles": ERLOTINIB, "explain_top_n": 0,
                                                       "include_llm_explanation": True}).json()
    assert body["plain_english_explanation"]["text"] == "stub explanation"
    ctx = seen["ctx"]  # the LLM is given only structured model output
    assert ctx["endpoint"] == "EGFR" and ctx["model_test_auc"] is not None
    assert ctx["top_shap_features"] and set(ctx["top_shap_features"][0]) == {"feature", "effect", "shap_log_odds"}


def test_llm_can_be_disabled():
    body = client.post("/predict/toxicity", json={"smiles": ASPIRIN, "include_llm_explanation": False}).json()
    assert "plain_english_explanation" not in body


def test_falls_back_to_template_when_llm_fails(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    body = client.post("/predict/toxicity", json={"smiles": BISPHENOL_A,
                                                  "include_llm_explanation": True}).json()
    expl = body["plain_english_explanation"]
    assert expl["source"] == "template" and "computational prediction" in expl["text"]
