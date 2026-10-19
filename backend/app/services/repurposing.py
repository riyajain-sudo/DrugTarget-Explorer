"""Drug repurposing: disease -> known treatments -> their targets -> other approved drugs on those targets.

Pipeline for a disease D:
  1. Match D to disease names in the approved-indications table.
  2. K = approved drugs already indicated for the matched disease(s).
  3. T = single-protein targets of the drugs in K.
  4. Candidates = approved drugs (not in K) that hit a target in T, plus drugs whose
     structure is close to a drug in K (Tanimoto >= SIMILARITY_ONLY_MIN).
  5. Score each candidate from up to three signals and attach human-readable reasons:
       target   - how central the shared target is among D's known treatments
       similarity - max Tanimoto (Morgan r=2, 2048 bits) to any drug in K
       model    - uncalibrated bioactivity-model score on the shared target (kinases only)

These are hypotheses for further research, not clinical recommendations.
"""
import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sqlalchemy import select

from app.db.models import Drug, DrugTarget, Indication, Target, get_session
from app.ml.featurize import FP_BITS

SIMILARITY_ONLY_MIN = 0.55   # candidates with no shared target must be at least this similar
SIMILARITY_REASON_MIN = 0.40  # only cite similarity in the reasons above this value
WEIGHTS = {"target": 0.5, "similarity": 0.3, "model": 0.2}

# ChEMBL target id -> kinase name used by the bioactivity models
MODEL_TARGETS = {"CHEMBL203": "EGFR", "CHEMBL1862": "ABL1", "CHEMBL301": "CDK2",
                 "CHEMBL267": "SRC", "CHEMBL279": "VEGFR2", "CHEMBL5145": "BRAF"}

PHRASE_SYNONYMS = {"chronic myeloid leukemia": "leukemia myelogenous chronic bcr-abl positive",
                   "chronic myelogenous leukemia": "leukemia myelogenous chronic bcr-abl positive",
                   "cml": "leukemia myelogenous chronic bcr-abl positive",
                   "acute myeloid leukemia": "leukemia myeloid acute", "aml": "leukemia myeloid acute",
                   "high blood pressure": "hypertension", "heart attack": "myocardial infarction",
                   "type 2 diabetes": "diabetes mellitus type 2", "type 1 diabetes": "diabetes mellitus type 1",
                   "high cholesterol": "hypercholesterolemia", "cholesterol": "hypercholesterolemia",
                   "aids": "acquired immunodeficiency syndrome", "flu": "influenza", "cold": "common cold",
                   "depression": "depressive disorder", "anxiety": "anxiety disorders",
                   "epilepsy": "epilepsy", "schizophrenia": "schizophrenia"}
TOKEN_SYNONYMS = {"cancer": "neoplasm", "cancers": "neoplasm", "tumor": "neoplasm", "tumour": "neoplasm",
                   "tumors": "neoplasm", "tumours": "neoplasm"}


def _stems(text: str) -> tuple[str, ...]:
    text = text.lower().strip()
    for phrase, repl in PHRASE_SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(phrase)}\b", repl, text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return tuple(TOKEN_SYNONYMS.get(t, t).rstrip("s") for t in tokens)


@dataclass
class Index:
    """Everything the engine needs, loaded from SQLite once."""
    drug_ids: list[str]
    names: dict[str, str]
    smiles: dict[str, str]
    row: dict[str, int]
    fps: np.ndarray                        # (n_drugs, FP_BITS) float32
    fp_counts: np.ndarray
    drug_targets: dict[str, set[str]]      # single-protein targets only
    target_drugs: dict[str, set[str]]
    target_info: dict[str, dict]
    diseases: dict[str, set[str]]          # disease name -> drug ids
    drug_diseases: dict[str, list[str]]
    disease_stems: dict[str, tuple[str, ...]] = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_index() -> Index:
    with get_session() as s:
        drugs = s.execute(select(Drug)).scalars().all()
        if not drugs:
            raise FileNotFoundError("repurposing database is empty; run scripts/download_repurposing.py and app.db.seed_data")
        targets = {t.id: t for t in s.execute(select(Target)).scalars()}
        links = s.execute(select(DrugTarget)).scalars().all()
        inds = s.execute(select(Indication)).scalars().all()

    fps = np.stack([np.unpackbits(np.frombuffer(d.fingerprint, dtype=np.uint8))[:FP_BITS] for d in drugs]).astype(np.float32)
    drug_targets: dict[str, set[str]] = {}
    target_drugs: dict[str, set[str]] = {}
    for l in links:
        if targets[l.target_id].target_type != "SINGLE PROTEIN":
            continue  # skip DNA, complexes, organisms, ... - not meaningful "shared protein" reasons
        drug_targets.setdefault(l.drug_id, set()).add(l.target_id)
        target_drugs.setdefault(l.target_id, set()).add(l.drug_id)
    diseases: dict[str, set[str]] = {}
    drug_diseases: dict[str, list[str]] = {}
    for i in inds:
        diseases.setdefault(i.disease, set()).add(i.drug_id)
        drug_diseases.setdefault(i.drug_id, []).append(i.disease)
    return Index(
        drug_ids=[d.id for d in drugs], names={d.id: d.name or d.id for d in drugs},
        smiles={d.id: d.smiles for d in drugs}, row={d.id: k for k, d in enumerate(drugs)},
        fps=fps, fp_counts=fps.sum(axis=1), drug_targets=drug_targets, target_drugs=target_drugs,
        target_info={t.id: {"name": t.name, "uniprot": t.uniprot} for t in targets.values()},
        diseases=diseases, drug_diseases=drug_diseases,
        disease_stems={name: _stems(name) for name in diseases},
    )


def match_disease(query: str, idx: Index) -> tuple[list[str], list[str]]:
    """Return (matched disease names, close-name suggestions if nothing matched)."""
    q = _stems(query)
    if not q:
        return [], []
    exact = [d for d, st in idx.disease_stems.items() if st == q]
    if exact:
        return exact, []
    contains = [d for d, st in idx.disease_stems.items() if set(q) <= set(st)]
    if contains:
        fewest = min(len(idx.disease_stems[d]) for d in contains)
        return [d for d in contains if len(idx.disease_stems[d]) == fewest], []
    # partial word overlap (e.g. "chronic lymphoid leukemia"): accept the best matches if most words agree
    qset = set(q)
    overlap = {d: len(qset & set(st)) / len(qset | set(st)) for d, st in idx.disease_stems.items()}
    best = max(overlap.values(), default=0)
    if best >= 0.5:
        return [d for d, v in overlap.items() if v == best], []
    suggestions = difflib.get_close_matches(query.strip().lower(), [d.lower() for d in idx.diseases], n=5, cutoff=0.7)
    by_lower = {d.lower(): d for d in idx.diseases}
    return [], [by_lower[s] for s in suggestions]


def tanimoto_to(idx: Index, cand_rows: np.ndarray, known_rows: np.ndarray) -> np.ndarray:
    """(n_cand, n_known) Tanimoto matrix from binary fingerprints."""
    inter = idx.fps[cand_rows] @ idx.fps[known_rows].T
    union = idx.fp_counts[cand_rows][:, None] + idx.fp_counts[known_rows][None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _model_score(idx: Index, drug_id: str, shared: list[str]) -> tuple[float, str] | None:
    """Best uncalibrated bioactivity-model score over covered shared kinase targets."""
    kinases = [MODEL_TARGETS[t] for t in shared if t in MODEL_TARGETS]
    if not kinases:
        return None
    from app.ml.predict_bioactivity import predict_bioactivity
    try:
        scores = predict_bioactivity(idx.smiles[drug_id])
    except (ValueError, FileNotFoundError):
        return None
    name = max(kinases, key=lambda k: scores[k])
    return scores[name], name


def repurpose(disease: str, top_n: int = 10) -> dict:
    idx = load_index()
    matched, suggestions = match_disease(disease, idx)
    if not matched:
        return {"query": disease, "matched_diseases": [], "suggestions": suggestions, "candidates": []}

    known = set().union(*(idx.diseases[d] for d in matched))
    disease_label = matched[0] if len(matched) == 1 else f"{matched[0]} (and {len(matched) - 1} related)"

    # targets of known treatments, and how many known drugs act on each
    target_support: dict[str, set[str]] = {}
    for k in known:
        for t in idx.drug_targets.get(k, ()):
            target_support.setdefault(t, set()).add(k)
    max_support = max((len(v) for v in target_support.values()), default=1)

    known_rows = np.array([idx.row[k] for k in known])
    known_list = list(known)
    cand_ids = [d for d in idx.drug_ids if d not in known]
    cand_rows = np.array([idx.row[d] for d in cand_ids])
    sim = tanimoto_to(idx, cand_rows, known_rows) if len(known_rows) else np.zeros((len(cand_ids), 0))

    results = []
    for i, cid in enumerate(cand_ids):
        shared = sorted(t for t in idx.drug_targets.get(cid, ()) if t in target_support)
        best_j = int(sim[i].argmax()) if sim.shape[1] else -1
        best_sim = float(sim[i, best_j]) if best_j >= 0 else 0.0
        if not shared and best_sim < SIMILARITY_ONLY_MIN:
            continue

        signals = {"similarity": best_sim}
        signals["target"] = max((len(target_support[t]) / max_support for t in shared), default=0.0)
        model = _model_score(idx, cid, shared) if shared else None
        if model:
            signals["model"] = model[0]
        weights = {k: WEIGHTS[k] for k in signals}
        score = sum(signals[k] * weights[k] for k in signals) / sum(weights.values())

        reasons = []
        for t in sorted(shared, key=lambda t: -len(target_support[t]))[:2]:
            info = idx.target_info[t]
            treat = sorted(idx.names[k] for k in target_support[t])
            shown = ", ".join(treat[:3]) + (f" and {len(treat) - 3} more" if len(treat) > 3 else "")
            uni = f" [UniProt {info['uniprot'].split(';')[0]}]" if info["uniprot"] else ""
            reasons.append(f"Shares target {info['name']}{uni} with {len(treat)} known treatment(s) for "
                           f"{disease_label}: {shown}")
        if best_sim >= SIMILARITY_REASON_MIN:
            reasons.append(f"Structurally similar (Tanimoto {best_sim:.2f}) to {idx.names[known_list[best_j]]}, "
                           f"approved for {disease_label}")
        if model:
            reasons.append(f"Bioactivity model score is {model[0]:.0%} for {model[1]} activity (uncalibrated; IC50 <= 1 uM)")
        if not reasons:
            reasons.append(f"Structurally related (Tanimoto {best_sim:.2f}) to {idx.names[known_list[best_j]]}")

        results.append({
            "drug": idx.names[cid], "chembl_id": cid, "smiles": idx.smiles[cid],
            "score": round(score, 3),
            "signals": {k: round(v, 3) for k, v in signals.items()},
            "shared_targets": [{"chembl_id": t, "name": idx.target_info[t]["name"]} for t in shared],
            "most_similar_known_drug": ({"name": idx.names[known_list[best_j]], "chembl_id": known_list[best_j],
                                        "smiles": idx.smiles[known_list[best_j]], "tanimoto": round(best_sim, 3)}
                                        if best_j >= 0 else None),
            "currently_approved_for": sorted(set(idx.drug_diseases.get(cid, [])))[:5],
            "reasons": reasons,
        })

    results.sort(key=lambda r: -r["score"])
    return {
        "query": disease, "matched_diseases": matched, "suggestions": [],
        "known_treatments": sorted(idx.names[k] for k in known),
        "known_treatment_targets": sorted(
            ({"chembl_id": t, "name": idx.target_info[t]["name"], "n_known_drugs": len(v)}
             for t, v in target_support.items()), key=lambda x: -x["n_known_drugs"])[:10],
        "n_candidates": len(results), "candidates": results[:top_n],
        "weights_note": "score = weighted mean of available signals (target 0.5, similarity 0.3, bioactivity model 0.2; "
                        "the model signal is only available for the 6 kinases it was trained on)",
        "disclaimer": "Computational hypotheses for further research, not medical advice or clinical evidence.",
    }
