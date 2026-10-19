const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error(`Cannot reach the backend at ${BASE}. Is the server running?`);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
    } catch { /* non-JSON error body */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const post = (path, body) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const predictBioactivity = (smiles, includeLlmExplanation = false) =>
  post("/predict/bioactivity", { smiles, explain_top_n: 6, top_k_features: 8,
    include_llm_explanation: includeLlmExplanation });
export const predictToxicity = (smiles, includeLlmExplanation = false) =>
  post("/predict/toxicity", { smiles, explain_top_n: 13, top_k_features: 8,
    include_llm_explanation: includeLlmExplanation });
export const predictDruglikeness = (smiles) => post("/predict/druglikeness", { smiles });

export const getMoleculeInfo = (smiles) => request(`/molecule/info?smiles=${encodeURIComponent(smiles)}`);

export const analyzeMolecule = async (smiles, includeLlmExplanation = false) => {
  const [bioactivity, toxicity, druglikeness, info] = await Promise.all([
    predictBioactivity(smiles, includeLlmExplanation), predictToxicity(smiles, includeLlmExplanation), predictDruglikeness(smiles),
    getMoleculeInfo(smiles).catch(() => null), // identity details are a bonus: never fail the analysis
  ]);
  return { bioactivity, toxicity, druglikeness, info };
};

export const searchDiseases = (q, limit = 8) =>
  request(`/diseases?q=${encodeURIComponent(q)}&limit=${limit}`);
export const repurpose = (disease, topN = 10) =>
  request(`/repurpose/${encodeURIComponent(disease)}?top_n=${topN}`);

export const structureUrl = (smiles, width = 320, height = 220) =>
  `${BASE}/molecule/svg?smiles=${encodeURIComponent(smiles)}&width=${width}&height=${height}`;
