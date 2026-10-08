# Drug Discovery Tool

ML/GenAI web app that predicts bioactivity, toxicity and drug-likeness for a molecule (with SHAP-based explanations and a plain-English summary) and suggests approved drugs that could be repurposed for a disease.

## Setup (Phase 0)

### Backend (Python 3.13)
```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # add GEMINI_API_KEY later
uvicorn app.main:app --reload
```
Check http://localhost:8000/health.

### Frontend (Node 22)
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173.

## API (Phase 5)

Run from `backend`: `.venv\Scripts\python -m uvicorn app.main:app --reload`
Interactive docs (try requests in the browser): http://localhost:8000/docs

| Endpoint | Returns |
|---|---|
| `POST /predict/bioactivity` | Per-kinase active/inactive, probability, confidence, SHAP breakdown |
| `POST /predict/toxicity` | 13 toxicity endpoints (12 Tox21 + ClinTox), flagged endpoints, SHAP breakdown |
| `POST /predict/druglikeness` | Lipinski, Veber, QED |

Body: `{"smiles": "...", "explain_top_n": 3, "top_k_features": 8}` (last two optional; druglikeness takes only `smiles`).
Invalid SMILES returns HTTP 422.

Tests: `.venv\Scripts\python -m pytest tests -q`

## Repurposing (Phase 7)

`GET /repurpose/{disease}?top_n=10` — e.g. `/repurpose/chronic myeloid leukemia`

Data comes from ChEMBL (DrugBank's target/indication data needs a licensed account): approved drugs,
drug-target mechanisms and approved indications, stored in SQLite (`backend/data/drugs.db`).

Build the database (one-time, from `backend`):
```powershell
.venv\Scripts\python scripts\download_repurposing.py
.venv\Scripts\python -m app.db.seed_data
```
Ranking: shared single-protein target with known treatments (0.5) + Tanimoto similarity to known treatments (0.3)
+ bioactivity-model score (0.2, kinase targets only). Every candidate carries plain-language `reasons`.
Results are research hypotheses, not medical advice.

## Frontend (Phase 8)

Run both servers, in two terminals:
```powershell
# terminal 1
cd backend
.venv\Scripts\python -m uvicorn app.main:app --reload
# terminal 2
cd frontend
npm run dev
```
Open http://localhost:5173 - **Predict molecule** (paste SMILES or draw with Ketcher) and **Repurpose drugs** (disease search).
The frontend calls the backend at `http://localhost:8000`; override with `VITE_API_URL` (see `frontend/.env.example`).
Extra endpoints used by the UI: `GET /diseases?q=` (search dropdown) and `GET /molecule/svg?smiles=` (structure images).
