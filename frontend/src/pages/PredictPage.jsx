import { useState } from "react";
import { analyzeMolecule } from "../api/client";
import MoleculeInput from "../components/MoleculeInput.jsx";
import ResultCard from "../components/ResultCard.jsx";
import ShapChart from "../components/ShapChart.jsx";
import StructureImage from "../components/StructureImage.jsx";

function Explanation({ data }) {
  if (!data) return null;
  return (
    <div className="rounded-lg border border-indigo-100 bg-indigo-50 p-3 text-sm text-slate-800">
      <p className="m-0">{data.text}</p>
      <p className="mb-0 mt-1 text-xs text-slate-500">
        {data.source === "gemini" ? `Written by ${data.model} from the SHAP output below` : "Template summary (AI explanation unavailable)"}
      </p>
    </div>
  );
}

function PredictionSection({ title, subtitle, result, positiveLabel, tone }) {
  const [selected, setSelected] = useState(result.predictions[0].endpoint);
  const explanation = result.explanations.find((e) => e.endpoint === selected);
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="m-0 text-lg font-semibold text-slate-900">{title}</h2>
      <p className="mb-3 mt-0.5 text-sm text-slate-500">{subtitle}</p>
      <Explanation data={result.plain_english_explanation} />
      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {result.predictions.map((p) => (
          <ResultCard key={p.endpoint} item={p} positiveLabel={positiveLabel} tone={tone}
                      selected={p.endpoint === selected} onSelect={() => setSelected(p.endpoint)} />
        ))}
      </div>
      <div className="mt-4 border-t border-slate-100 pt-3">
        <h3 className="mb-2 mt-0 text-sm font-semibold text-slate-800">
          Why {selected}? <span className="font-normal text-slate-500">(click a card to change)</span>
        </h3>
        <ShapChart contributions={explanation?.contributions} positiveLabel={positiveLabel}
          omittedAbsMass={explanation?.omitted_abs_shap_mass}
          omittedBitCount={explanation?.omitted_absent_bit_count} />
      </div>
    </section>
  );
}

function DrugLikeness({ data }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="m-0 text-lg font-semibold text-slate-900">Drug-likeness</h2>
      <p className={`mb-2 mt-1 text-sm font-medium ${data.drug_like ? "text-emerald-700" : "text-rose-700"}`}>{data.summary}</p>
      <ul className="m-0 list-none space-y-1 p-0 text-sm">
        {[...data.lipinski.rules, ...data.veber.rules].map((r) => (
          <li key={r.rule} className="flex items-center justify-between gap-2">
            <span className="text-slate-700">{r.rule}</span>
            <span className={`tabular-nums ${r.passed ? "text-emerald-700" : "text-rose-700"}`}>
              {r.value} {r.passed ? "✓" : "✗"}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

const titleCase = (s) => s.toLowerCase().replace(/\b[a-z]/g, (c) => c.toUpperCase());

function MoleculeIdentity({ smiles, info, exampleName }) {
  const [copied, setCopied] = useState(false);
  // name: the approved-drug match if there is one, otherwise the example button that was clicked
  const name = info?.name ? titleCase(info.name) : exampleName;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(smiles);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard unavailable */ }
  };
  const canonicalDiffers = info && info.canonical_smiles !== smiles;
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <StructureImage smiles={smiles} width={400} height={300} label={null} />
      <h2 className="mb-0 mt-3 text-lg font-semibold text-slate-900">{name ?? "Unnamed molecule"}</h2>
      {info?.name && (
        <p className="m-0 text-xs font-medium text-emerald-700">
          Approved drug{info.approved_for.length > 0 && <> · {info.approved_for.slice(0, 3).join(", ")}</>}
        </p>
      )}
      {!name && <p className="m-0 text-xs text-slate-500">Not an approved drug in the local database</p>}
      {info && (
        <p className="mb-0 mt-1 text-sm text-slate-600">
          {info.formula} · {info.molecular_weight} g/mol
        </p>
      )}
      <div className="mt-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Input SMILES</span>
          <button type="button" onClick={copy} className="text-xs text-indigo-700 hover:underline">
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <p className="m-0 mt-1 break-all rounded-md bg-slate-100 p-2 font-mono text-xs text-slate-700">{smiles}</p>
        {canonicalDiffers && (
          <>
            <span className="mt-2 block text-xs font-semibold uppercase tracking-wide text-slate-500">Canonical SMILES</span>
            <p className="m-0 mt-1 break-all rounded-md bg-slate-100 p-2 font-mono text-xs text-slate-700">{info.canonical_smiles}</p>
          </>
        )}
      </div>
    </section>
  );
}

export default function PredictPage() {
  const [state, setState] = useState({ status: "idle" });
  const [includeLlmExplanation, setIncludeLlmExplanation] = useState(false);

  const analyze = async (smiles, exampleName) => {
    setState({ status: "loading", smiles });
    try {
      setState({ status: "done", smiles, exampleName,
        data: await analyzeMolecule(smiles, includeLlmExplanation) });
    } catch (e) {
      setState({ status: "error", smiles, message: e.message });
    }
  };

  const { status, smiles, exampleName, data, message } = state;
  return (
    <div className="space-y-5">
      <div>
        <h1 className="m-0 text-2xl font-bold text-slate-900">Predict a molecule's properties</h1>
        <p className="mt-1 text-slate-600">
          Bioactivity against six kinases, 13 toxicity endpoints and drug-likeness, each with the features driving the prediction.
        </p>
      </div>
      <MoleculeInput onAnalyze={analyze} loading={status === "loading"} />
      <label className="flex max-w-3xl items-start gap-2 text-sm text-slate-600">
        <input type="checkbox" checked={includeLlmExplanation}
          onChange={(event) => setIncludeLlmExplanation(event.target.checked)} />
        <span>Opt in to a Gemini-written summary. If selected, derived model scores, feature labels and SHAP summaries are sent to Google; raw SMILES is not included. Leave unchecked to receive model outputs without an external explanation request.</span>
      </label>

      {status === "loading" && (
        <div role="status" className="rounded-xl border border-slate-200 bg-white p-6 text-center text-slate-600">
          Running models and writing the explanation… (a few seconds)
        </div>
      )}
      {status === "error" && (
        <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-rose-800">
          <strong>Could not analyze that molecule.</strong> {message}
        </div>
      )}
      {status === "done" && (
        <div className="grid gap-5 lg:grid-cols-[18rem_minmax(0,1fr)]">
          <div className="space-y-5">
            <MoleculeIdentity smiles={smiles} info={data.info} exampleName={exampleName} />
            <DrugLikeness data={data.druglikeness} />
          </div>
          <div className="space-y-5">
            <PredictionSection key={`bio-${smiles}`} title="Bioactivity" result={data.bioactivity}
              subtitle="Uncalibrated model scores (0–1) for kinase activity at IC50 ≤ 1 µM; not calibrated probabilities." positiveLabel="active" tone="good" />
            <PredictionSection key={`tox-${smiles}`} title="Toxicity" result={data.toxicity}
              subtitle="Uncalibrated model scores (0–1) for positive assay results; not calibrated probabilities." positiveLabel="toxic" tone="danger" />
          </div>
        </div>
      )}
      <p className="text-xs text-slate-500">
        Computational predictions from models trained on public data (ChEMBL, Tox21, ClinTox). Not a substitute for lab testing.
      </p>
    </div>
  );
}
