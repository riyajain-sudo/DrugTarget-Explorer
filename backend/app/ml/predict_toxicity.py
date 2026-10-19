"""Load the trained toxicity models and score a SMILES on every endpoint.

Run: python -m app.ml.predict_toxicity "<SMILES>"
"""
import sys
from functools import lru_cache
from pathlib import Path

import joblib

from app.ml.featurize import featurize

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "toxicity.pkl"
FLAG_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def predict_toxicity(smiles: str) -> dict[str, float]:
    """Return uncalibrated {endpoint: predict_proba score for a toxic/positive result}."""
    bundle = load_bundle()
    x = featurize(smiles).reshape(1, -1)
    return {ep: float(m.predict_proba(x)[0, 1]) for ep, m in bundle["models"].items()}


def main():
    smiles = sys.argv[1] if len(sys.argv) > 1 else "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    try:
        scores = predict_toxicity(smiles)
    except ValueError as e:
        sys.exit(str(e))
    desc = load_bundle()["descriptions"]
    flagged = [ep for ep, p in scores.items() if p >= FLAG_THRESHOLD]
    print(f"SMILES: {smiles}\nUncalibrated model score for toxic/positive result per endpoint:")
    for ep, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        mark = " <-- flagged" if p >= FLAG_THRESHOLD else ""
        print(f"  {ep:14s} {p:.3f}  {'#' * int(p * 30):30s} {desc[ep]}{mark}")
    print(f"\nFlagged {len(flagged)}/{len(scores)} endpoints at threshold {FLAG_THRESHOLD}")


if __name__ == "__main__":
    main()
