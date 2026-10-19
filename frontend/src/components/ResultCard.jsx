// One endpoint's prediction. `tone` colours the positive class (e.g. toxic = red, active = green).
const TONES = {
  danger: { badge: "bg-rose-100 text-rose-800", bar: "bg-rose-500" },
  good: { badge: "bg-emerald-100 text-emerald-800", bar: "bg-emerald-500" },
};

export default function ResultCard({ item, positiveLabel, tone = "danger", selected, onSelect }) {
  const positive = item.prediction === positiveLabel;
  const t = TONES[tone];
  return (
    <button type="button" onClick={onSelect} aria-pressed={selected}
            className={`w-full rounded-lg border p-3 text-left transition ${
              selected ? "border-indigo-500 ring-2 ring-indigo-200" : "border-slate-200 hover:border-slate-300"} bg-white`}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-slate-900">{item.endpoint}</div>
          {item.description && <div className="truncate text-xs text-slate-500" title={item.description}>{item.description}</div>}
        </div>
        <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${positive ? t.badge : "bg-slate-100 text-slate-600"}`}>
          {item.prediction}
        </span>
      </div>
      <div className="mt-2 h-2 rounded bg-slate-100" role="img" aria-label={`Uncalibrated model score ${(item.model_score * 100).toFixed(0)}%`}>
        <div className={`h-2 rounded ${positive ? t.bar : "bg-slate-400"}`} style={{ width: `${item.model_score * 100}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-slate-500">
        <span>model score {(item.model_score * 100).toFixed(0)}%</span>
        {item.model_test_auc != null && <span title="Model ROC AUC on held-out test scaffolds">test AUC {item.model_test_auc.toFixed(2)}</span>}
      </div>
    </button>
  );
}
