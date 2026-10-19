"""Train toxicity classifiers: 12 Tox21 assay endpoints + ClinTox clinical-trial toxicity.

One XGBoost model per endpoint (Tox21 labels are sparse per assay), evaluated on a
scaffold split. Class imbalance is handled with scale_pos_weight.

Run: python -m app.ml.train_toxicity
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from app.ml.featurize import FEATURE_NAMES, featurize_many
from app.ml.train_bioactivity import scaffold_split

BACKEND = Path(__file__).resolve().parents[2]
DATA = BACKEND / "data" / "toxicity"
MODEL_PATH = BACKEND / "app" / "models" / "toxicity.pkl"
TEST_FRACTION = 0.2
SEED = 42

TOX21_ENDPOINTS = {
    "NR-AR": "Androgen receptor signaling",
    "NR-AR-LBD": "Androgen receptor (ligand-binding domain)",
    "NR-AhR": "Aryl hydrocarbon receptor activation",
    "NR-Aromatase": "Aromatase inhibition",
    "NR-ER": "Estrogen receptor signaling",
    "NR-ER-LBD": "Estrogen receptor (ligand-binding domain)",
    "NR-PPAR-gamma": "PPAR-gamma activation",
    "SR-ARE": "Oxidative stress response (ARE)",
    "SR-ATAD5": "DNA damage response (ATAD5)",
    "SR-HSE": "Heat shock response",
    "SR-MMP": "Mitochondrial membrane potential disruption",
    "SR-p53": "p53 pathway activation (genotoxic stress)",
}
CLINTOX_ENDPOINT = ("ClinTox", "Failed clinical trials due to toxicity")


def make_model(pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.5, scale_pos_weight=pos_weight, eval_metric="auc",
        n_jobs=-1, random_state=SEED,
    )


def train_endpoint(name, X, y, smiles, results):
    """Evaluate on a scaffold split, then refit on all labelled rows. Returns the shipped model."""
    tr, te = scaffold_split(smiles, TEST_FRACTION, SEED)
    pw = lambda yy: float((yy == 0).sum() / max((yy == 1).sum(), 1))
    model = make_model(pw(y[tr])).fit(X[tr], y[tr])
    proba = model.predict_proba(X[te])[:, 1]
    ok = len(set(y[te])) == 2
    m = {
        "n_train": len(tr), "n_test": len(te),
        "positive_rate": float(y.mean()),
        "roc_auc": roc_auc_score(y[te], proba) if ok else None,
        "average_precision": average_precision_score(y[te], proba) if ok else None,
        "test_positive_rate_baseline": float(y[te].mean()),
    }
    results[name] = m
    auc = f"{m['roc_auc']:.3f}" if ok else "n/a"
    ap = f"{m['average_precision']:.3f}" if ok else "n/a"
    print(f"{name:14s} {len(y):6d} {m['positive_rate']:7.1%} {auc:>7s} {ap:>7s} {m['test_positive_rate_baseline']:9.3f}")
    return make_model(pw(y)).fit(X, y)


def main():
    models, metrics = {}, {}
    print(f"{'endpoint':14s} {'n':>6s} {'pos%':>7s} {'ROC AUC':>7s} {'AP':>7s} {'AP base':>9s}")

    tox = pd.read_csv(DATA / "tox21.csv.gz")
    X_all, kept = featurize_many(tox.smiles)
    tox = tox.iloc[kept].reset_index(drop=True)
    for ep in TOX21_ENDPOINTS:
        lab = tox[ep].notna().to_numpy()
        models[ep] = train_endpoint(ep, X_all[lab], tox.loc[lab, ep].astype(int).to_numpy(),
                                    tox.smiles[lab].reset_index(drop=True), metrics)

    ct = pd.read_csv(DATA / "clintox.csv.gz")
    X_ct, kept = featurize_many(ct.smiles)
    ct = ct.iloc[kept].reset_index(drop=True)
    models["ClinTox"] = train_endpoint("ClinTox", X_ct, ct["CT_TOX"].astype(int).to_numpy(), ct.smiles, metrics)

    descriptions = {**TOX21_ENDPOINTS, CLINTOX_ENDPOINT[0]: CLINTOX_ENDPOINT[1]}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "descriptions": descriptions, "feature_names": FEATURE_NAMES,
                 "metrics": metrics}, MODEL_PATH)
    MODEL_PATH.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    aucs = [m["roc_auc"] for m in metrics.values() if m["roc_auc"] is not None]
    print(f"\nMean ROC AUC over {len(aucs)} endpoints: {np.mean(aucs):.3f}\nsaved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
