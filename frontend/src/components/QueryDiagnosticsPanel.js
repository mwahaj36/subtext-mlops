"use client";
import React from "react";
import { AlertCircle, Cpu, GitBranch, Layers, Sparkles } from "lucide-react";

const STATUS_STYLES = {
  succeeded: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  not_called: "border-white/15 bg-white/5 text-gray-300",
  timeout: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  invalid_output: "border-orange-500/40 bg-orange-500/10 text-orange-300",
  error: "border-red-500/40 bg-red-500/10 text-red-300",
};

const MODES = [
  { value: "auto", label: "Auto Contract" },
  { value: "dense_only", label: "Dense Only" },
  { value: "static_hybrid", label: "Static Hybrid" },
];

function collectConstraints(diagnostics, appliedFilters) {
  const ext = diagnostics?.extraction || {};
  const filters = appliedFilters || {};
  const items = [];
  const pushList = (label, vals, prefix = "") => {
    if (Array.isArray(vals) && vals.length > 0) {
      items.push(`${label}: ${vals.map((v) => `${prefix}${v}`).join(", ")}`);
    }
  };
  pushList("Genres", filters.genres?.length ? filters.genres : ext.genres);
  pushList("Exclude Genres", filters.excluded_genres?.length ? filters.excluded_genres : ext.excluded_genres, "no ");
  pushList("Exclude Terms", filters.excluded_terms?.length ? filters.excluded_terms : ext.excluded_terms, "no ");
  pushList("Actors", filters.actors?.length ? filters.actors : ext.actors);
  pushList("Directors", filters.directors?.length ? filters.directors : ext.directors);
  pushList("Reference", filters.reference_titles?.length ? filters.reference_titles : ext.reference_titles);
  const yMin = filters.min_year ?? ext.year_min;
  const yMax = filters.max_year ?? ext.year_max;
  if (yMin || yMax) items.push(`Years: ${yMin || "*"}–${yMax || "*"}`);
  const lang = filters.language || ext.language;
  if (lang) items.push(`Language: ${lang.toUpperCase()}`);
  if (filters.min_vote) items.push(`Min Rating: ${filters.min_vote}`);
  return items;
}

export default function QueryDiagnosticsPanel({
  diagnostics,
  appliedFilters,
  unsupportedNotices = [],
  searchError = "",
  retrievalMode = "auto",
  onRetrievalModeChange,
}) {
  if (!diagnostics && !searchError && unsupportedNotices.length === 0) return null;

  const llm = diagnostics?.llm_fallback || { called: false, status: "not_called", reason: "" };
  const statusStyle = STATUS_STYLES[llm.status] || STATUS_STYLES.not_called;
  const scores = diagnostics?.label_scores || {};
  const fineLabels = diagnostics?.predicted_fine_labels || [];
  const broadLabels = diagnostics?.predicted_broad_labels || [];
  const constraints = collectConstraints(diagnostics, appliedFilters);

  return (
    <section
      aria-label="Query Understanding Diagnostics"
      className="w-full max-w-7xl mx-auto mb-6 px-6 md:px-10"
    >
      <div className="bg-white/[0.03] border border-white/10 backdrop-blur-xl p-4 md:p-5 space-y-3">
        {searchError && (
          <div className="flex items-start gap-2.5 border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">
            <AlertCircle size={15} className="shrink-0 mt-0.5 text-red-400" />
            <span>{searchError}</span>
          </div>
        )}

        {unsupportedNotices.length > 0 && (
          <div className="flex items-start gap-2.5 border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200">
            <AlertCircle size={14} className="shrink-0 mt-0.5 text-amber-400" />
            <div className="space-y-0.5">
              {unsupportedNotices.map((msg, i) => (
                <p key={i}>{msg}</p>
              ))}
            </div>
          </div>
        )}

        {diagnostics && (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-3">
              <div className="flex flex-wrap items-center gap-2 text-[10px] font-bold uppercase tracking-wider">
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 border border-[var(--primary)]/40 bg-[var(--primary)]/10 text-[var(--primary)]">
                  <Cpu size={12} />
                  {diagnostics.model_id} v{diagnostics.model_version}
                </span>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 border border-white/15 bg-white/5 text-gray-200">
                  <GitBranch size={12} />
                  Route: {diagnostics.selected_route}
                </span>
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 border ${statusStyle}`}>
                  <Sparkles size={12} />
                  LLM: {llm.status}
                </span>
                {diagnostics.is_compound && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 border border-white/15 bg-white/5 text-gray-300">
                    <Layers size={11} />
                    Compound
                  </span>
                )}
              </div>

              {onRetrievalModeChange && (
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] uppercase tracking-widest text-gray-500 font-black mr-1">
                    Mode
                  </span>
                  {MODES.map((m) => (
                    <button
                      key={m.value}
                      type="button"
                      onClick={() => onRetrievalModeChange(m.value)}
                      className={`px-2.5 py-1 text-[9px] font-black uppercase tracking-wider border transition-colors ${
                        retrievalMode === m.value
                          ? "border-[var(--primary)] bg-[var(--primary)] text-black"
                          : "border-white/10 bg-white/5 text-gray-400 hover:text-white"
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px]">
              <div className="space-y-1.5">
                <div className="text-[9px] uppercase tracking-widest text-gray-400 font-black">
                  Predicted Intents ({broadLabels.join(", ") || "none"})
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {fineLabels.map((lbl) => (
                    <span
                      key={lbl}
                      className="px-2 py-0.5 border border-white/15 bg-black/40 text-gray-100 font-mono text-[10px]"
                    >
                      {lbl} {scores[lbl] !== undefined ? `(${Math.round(scores[lbl] * 100)}%)` : ""}
                    </span>
                  ))}
                </div>
                <p className="text-[10px] text-gray-400">
                  {diagnostics.route_reason}
                  {llm.reason ? ` • Fallback: ${llm.reason}` : ""}
                </p>
              </div>

              <div className="space-y-1.5">
                <div className="text-[9px] uppercase tracking-widest text-gray-400 font-black">
                  Extracted & Applied Constraints (policy v{diagnostics.route_policy_version})
                </div>
                {constraints.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {constraints.map((c, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 border border-white/15 bg-white/5 text-gray-200 text-[10px]"
                      >
                        {c}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-[10px] text-gray-500">
                    Open-ended semantic query (no hard metadata filters applied).
                  </p>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
