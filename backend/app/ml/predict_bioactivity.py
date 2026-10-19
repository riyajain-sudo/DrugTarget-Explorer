"""Load the trained bioactivity model and score a SMILES against each kinase.

Run: python -m app.ml.predict_bioactivity "<SMILES>"
"""
import sys
from functools import lru_cache
from pathlib import Path

import joblib

from app.ml.featurize import featurize

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "bioactivity.pkl"


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def predict_bioactivity(smiles: str) -> dict[str, float]:
    """Return uncalibrated {kinase: predict_proba score for IC50 <= 1 uM}."""
    bundle = load_bundle()
    x = featurize(smiles).reshape(1, -1)
    return {t: float(m.predict_proba(x)[0, 1]) for t, m in bundle["models"].items()}


def main():
    smiles = sys.argv[1] if len(sys.argv) > 1 else "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"  # erlotinib
    try:
        scores = predict_bioactivity(smiles)
    except ValueError as e:
        sys.exit(str(e))
    print(f"SMILES: {smiles}\nUncalibrated model score for active (IC50 <= 1 uM) per kinase:")
    for t, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {t:7s} {p:.3f}  {'#' * int(p * 30)}")


if __name__ == "__main__":
    main()
