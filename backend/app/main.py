from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.routers import molecule, predict, repurpose

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model caches so the first request is fast. A missing model file is
    # reported per request as a 503 rather than stopping the server from starting.
    from app.ml.predict_bioactivity import load_bundle as load_bio
    from app.ml.predict_toxicity import load_bundle as load_tox
    for loader in (load_bio, load_tox):
        try:
            loader()
        except FileNotFoundError:
            pass
    try:
        from app.services.repurposing import load_index
        load_index()
        from app.services.molecule_info import _drug_index
        _drug_index()
    except FileNotFoundError:
        # Expected for a shell-only checkout without the optional seed database.
        pass
    except Exception:
        log.exception("Repurposing or molecule-info startup initialization failed")
    yield


app = FastAPI(title="Drug Discovery API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict.router)
app.include_router(repurpose.router)
app.include_router(repurpose.diseases_router)
app.include_router(molecule.router)


@app.get("/health")
def health():
    """Liveness only; a running process may not have optional artifacts."""
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response):
    """Report required artifact presence without loading models or contacting providers."""
    from app.db.models import DB_PATH
    from app.ml.predict_bioactivity import MODEL_PATH as BIOACTIVITY_MODEL
    from app.ml.predict_toxicity import MODEL_PATH as TOXICITY_MODEL

    artifacts = {
        "bioactivity_model_file": Path(BIOACTIVITY_MODEL).is_file(),
        "toxicity_model_file": Path(TOXICITY_MODEL).is_file(),
        "repurposing_database_file": Path(DB_PATH).is_file(),
    }
    is_ready = all(artifacts.values())
    response.status_code = 200 if is_ready else 503
    return {"ready": is_ready, "artifacts_present": artifacts}
