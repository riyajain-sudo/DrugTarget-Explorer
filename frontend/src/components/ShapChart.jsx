// Diverging bar chart of SHAP contributions (log-odds). Red pushes toward the positive class, blue away.
export default function ShapChart({ contributions, positiveLabel = "active/toxic", omittedAbsMass = 0, omittedBitCount = 0 }) {
  if (!contributions?.length) return <p className="text-sm text-slate-500">No feature contributions available.</p>;
  const max = Math.max(...contributions.map((c) => Math.abs(c.shap)), 1e-6);
  return (
    <div>
      <ul className="m-0 list-none space-y-1.5 p-0" aria-label="SHAP feature contributions">
        {contributions.map((c) => {
          const pct = (Math.abs(c.shap) / max) * 50;
          const positive = c.shap > 0;
          return (
            <li key={c.feature} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)_3.5rem] items-center gap-2 text-sm">
              <span className="truncate text-right text-slate-700" title={c.label}>{c.label}</span>
              <div className="relative h-5 rounded bg-slate-100">
                <div className="absolute inset-y-0 left-1/2 w-px bg-slate-300" />
                <div className={`absolute inset-y-0.5 rounded ${positive ? "bg-rose-500" : "bg-sky-600"}`}
                     style={positive ? { left: "50%", width: `${pct}%` } : { right: "50%", width: `${pct}%` }} />
              </div>
              <span className={`text-xs tabular-nums ${positive ? "text-rose-600" : "text-sky-700"}`}>
                {c.shap > 0 ? "+" : ""}{c.shap.toFixed(2)}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="mt-2 text-xs text-slate-500">
        <span className="font-medium text-rose-600">Red</span> pushes the prediction toward {positiveLabel};{" "}
        <span className="font-medium text-sky-700">blue</span> pushes away. Values are SHAP log-odds.
      </p>
      <p className="mt-1 text-xs text-slate-500">
        Heuristic, filtered view: top SHAP among descriptors and present fragments. {omittedBitCount} absent fingerprint bits omitted; their absolute SHAP mass is {Number(omittedAbsMass).toFixed(3)}.
      </p>
    </div>
  );
}
