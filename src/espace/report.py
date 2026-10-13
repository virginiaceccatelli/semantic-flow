"""Stage 293 CPU analysis: Experiment 1, token-level, Experiment 2 and the combined
analysis, written to ``summary.json`` and ``report.md``."""
from __future__ import annotations

import json
from pathlib import Path

from src.espace import exp1, exp2, tokens
from src.espace.space import Space
from src.espace.stats import fci, fmt

BENCHMARKS = ("cruxeval", "mbpp")


def analyze(run, output, token_level=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frozen = run.frozen()
    summary = {"config": str(run.config_path), "data": str(run.data), "frozen_terms": frozen, "models": {}}
    for model in run.models():
        m = summary["models"][model] = {"experiment1": {}, "tokens": {}, "experiment2": {}}
        for ds in BENCHMARKS:
            r = exp1.analyse(run, model, ds)
            if r is not None:
                m["experiment1"][ds] = r
        if frozen is None:
            m["experiment2"] = "anchor lists not frozen yet: run `293 freeze` first (test results withheld)"
            space = None
        else:
            bank = run.bank(model)
            terms = exp2.present(frozen["terms"], bank)
            space = Space(bank.D, bank.rows(terms))
            if run.prompt(model, "cfpairs") is not None:
                m["experiment2"]["pairs"] = exp2.pairs_analysis(run, model, frozen["terms"],
                                                                run.e["primary_position"])
            for ds in BENCHMARKS:
                if run.prompt(model, ds) is not None:
                    m["experiment2"][ds] = exp2.benchmark_analysis(run, model, ds, frozen["terms"])
        if token_level:
            for ds in BENCHMARKS:
                trs = tokens.traces(run, model, ds, space)
                if trs:
                    m["tokens"][ds] = tokens.analyse(trs, run.e["claim_window"],
                                                     list(exp2.DIMENSIONS) if space is not None else None)
    (output / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (output / "report.md").write_text(render(summary))
    return output


def _exp1(lines, ds, r):
    lines += [f"### {ds}: N={r['n_rows']} answers ({r['n_programs']} programs x seeds), error rate "
              f"{fmt(r['error_rate'])}", ""]
    for seed, s in r["seeds"].items():
        lines.append(f"- seed {seed}: {s['n_traces']} traces, {s['n_labelled']} labelled, excluded "
                     f"{s['excluded'] or '{}'}, answer status {s['answer_status'] or '{}'}, accuracy "
                     f"{fmt(s['accuracy_labelled'])} (all items, exclusions wrong: {fmt(s['accuracy_all_items'])}), "
                     f"mean length {fmt(s['mean_tlen'], 0)}")
    lines += ["", "| Method | AUROC [95% CI] | per-seed mean ± sd | length-matched | "
              f"AUROC by length bin (low → high) |", "|---|---|---|---:|---|"]
    order = exp1.MAIN + [k for k in r["methods"] if k not in exp1.MAIN]
    for name in order:
        v = r["methods"].get(name)
        if not v:
            continue
        bins = " / ".join(fmt(b["auroc"]) for b in v["length_bins"]) if v["length_bins"] else ""
        lines.append(f"| {name} | {fmt(v['auroc'])}{fci(v['ci'])} | {fmt(v['per_seed_mean'])} ± "
                     f"{fmt(v['per_seed_sd'])} | {fmt(v['length_matched'])} | {bins} |")
    b = r["beyond"]
    lines.append("")
    for k, v in b.items():
        if isinstance(v, dict) and "delta" in v:
            lines.append(f"- {k}: ΔAUROC {fmt(v['delta'], 4)}{fci(v['ci'], 4)}")
    if "a_cone_entropy_matched" in b:
        lines.append(f"- A_cone AUROC matched on thinking entropy {fmt(b['a_cone_entropy_matched'])}, "
                     f"on length {fmt(b['a_cone_length_matched'])}")
    lr = b.get("logistic_a_cone_beyond_entropy_length_lexical")
    if lr:
        lines.append(f"- log A_cone beyond log thinking-entropy, log length, lexical: standardised coefficient "
                     f"{fmt(lr['coef_std'])}{fci(lr['coef_ci'])}, LR {fmt(lr['lr'], 2)} "
                     f"(p≈{fmt(lr['p_approx'], 4)}; rows not independent, use the interval)")
    lines.append(f"- Check: E-Space pass vs upstream end-of-thinking U coordinates, max |diff| "
                 f"{fmt(r['eot_projection_max_abs_diff'], 4)}")
    lines.append("")


def _tokens(lines, ds, t):
    lines += [f"### {ds}: {t['n_traces']} traces, {t['n_traces_with_claims']} with checkable claims, "
              f"{t['n_claims']} claims, {t['n_wrong_claims']} wrong", ""]
    w = t["wrong_claim_rate"]
    lines.append(f"- Traces with >= 1 wrong claim: incorrect answers {fmt(w['incorrect_traces'])}, "
                 f"correct answers {fmt(w['correct_traces'])}")
    c = t.get("claim_level")
    if c:
        lines.append(f"- Wrong vs right claims, window mean: cone AUROC {fmt(c['cone_auroc'])}{fci(c['cone_auroc_ci'])}; "
                     f"entropy AUROC {fmt(c['entropy_auroc'])}{fci(c['entropy_auroc_ci'])}; cone matched on "
                     f"entropy {fmt(c['cone_auroc_entropy_matched'])}")
    wt = t.get("within_trace")
    if wt:
        lines.append(f"- Within traces with both: cone wrong − right {fmt(wt['mean_diff'], 4)}, positive in "
                     f"{fmt(wt['share_positive'])} of {wt['n_traces']} (sign test p {fmt(wt['sign_test_p'], 4)})")
    for k, v in t["before_markers"].items():
        lines.append(f"- Cone in the {t['window']} tokens before revision markers minus trace mean ({k}): "
                     f"{fmt(v['mean_excess_cone'], 4)}{fci(v['ci'], 4)} over {v['n_markers']} markers")
    pe = t.get("peak_enrichment")
    if pe:
        lines.append(f"- Top-1% cone tokens near a wrong claim: {fmt(pe['share_peaks_near_wrong'])} vs "
                     f"{fmt(pe['share_tokens_near_wrong'])} of all tokens (ratio {fmt(pe['ratio'], 2)}; "
                     f"{pe['n_traces']} traces)")
    prof = t["cone_by_decile"]
    for k, v in prof.items():
        if v:
            lines.append(f"- Cone by thinking decile, {k}: " + " ".join(fmt(x, 3) for x in v))
    comb = t.get("combined")
    if comb:
        lines += ["", "Combined (exploratory): mean token cone by dominant E-Space category",
                  "", "| Category | correct: share / cone / entropy | incorrect: share / cone / entropy |",
                  "|---|---|---|"]
        a, b = comb["by_dominant_category"]["correct_traces"], comb["by_dominant_category"]["incorrect_traces"]
        for cat in a:
            lines.append(f"| {cat} | {fmt(a[cat]['share'])} / {fmt(a[cat]['cone'])} / {fmt(a[cat]['entropy'])} | "
                         f"{fmt(b[cat]['share'])} / {fmt(b[cat]['cone'])} / {fmt(b[cat]['entropy'])} |")
        if comb["E_at_claims"]:
            lines.append("- E coordinates at wrong / right claims: " + "; ".join(
                f"{k} {fmt(v['wrong'], 4)}/{fmt(v['right'], 4)}" for k, v in comb["E_at_claims"].items()))
        lines.append("- Trace-mean E coordinate as an error score (AUROC): " + "; ".join(
            f"{k} {fmt(v['auroc'])}{fci(v['ci'])}" for k, v in comb["trace_mean_E_auroc"].items()))
    lines.append("")


def _pairs(lines, p):
    lines += [f"Primary: {p['primary']['space']} at {p['primary']['position']}; Holm-adjusted Wilcoxon p: " +
              ", ".join(f"{k} {fmt(v, 4)}" for k, v in p["primary"]["holm_p"].items()),
              f"Terms used: " + "; ".join(f"{c}: +{s['anchors']} −{s['poles']}" for c, s in p["terms_used"].items()),
              f"Controls: {p['n_word_controls']} random-word E-Spaces, {p['n_gauss_controls']} Gaussian bases.", ""]
    lines += ["Behaviour (output prediction on the pairs, seeds pooled): " + "; ".join(
        f"{k} {fmt(v['accuracy_labelled'])} ({v['labelled']}/{v['n']}, excluded {v['excluded']})"
        for k, v in sorted(p["behaviour"].items())), ""]
    for dim, r in p["dimensions"].items():
        lines += [f"#### {dim} ({r['n_test_pairs']} test pairs)", "",
                  "| Position | E_orth d_z [CI] | pair acc | AUROC | E_bipolar d_z | candidate orth d_z | "
                  "|d_z| pct vs random words / Gaussian | other E dims d_z | U d_z |",
                  "|---|---|---:|---:|---:|---:|---|---|---|"]
        for pos, e in r["positions"].items():
            o = e["E_orth"]
            ci_dz = (o.get("ci") or {}).get("dz")
            nw, ng = e["null_abs_dz"]["random_words"], e["null_abs_dz"]["gaussian"]
            others = ", ".join(f"{k[:4]} {fmt(v, 2)}" for k, v in e["specificity_E_orth_dz"].items() if k != dim)
            us = ", ".join(f"{k[:6]} {fmt(v, 2)}" for k, v in e["U_dz"].items())
            lines.append(f"| {pos} | {fmt(o.get('dz'))}{fci(ci_dz)} | {fmt(o.get('pair_accuracy'))} | "
                         f"{fmt(o.get('auroc'))} | {fmt(e['E_bipolar'].get('dz'))} | "
                         f"{fmt(e['E_orth_candidate'].get('dz'))} | {fmt(nw['percentile_of_real'], 0)} / "
                         f"{fmt(ng['percentile_of_real'], 0)} | {others} | {us} |")
        b = r["baselines"]
        lines += ["", f"Baselines: differing-literal value d_z {fmt(b['differing_literal_value'].get('dz'))}, "
                  f"pair acc {fmt(b['differing_literal_value'].get('pair_accuracy'))}; keyword count pair acc "
                  f"{fmt(b['keyword_count'].get('pair_accuracy'))} (identical syntax → 0.5).", ""]


def _bench(lines, ds, r):
    lines += [f"#### {ds} ({r['n_programs']} programs; exploratory)", "",
              "| Dimension | with construct | runtime + | keyword AUROC (runtime) | position | E_orth syntax AUROC | "
              "E_orth runtime AUROC | E_bipolar runtime AUROC |", "|---|---:|---:|---|---|---|---|---|"]
    for dim, v in r["dimensions"].items():
        kw = v["keyword_runtime"]
        for p, e in v["positions"].items():
            rt_o, rt_b = e["E_orth"]["runtime_within_construct"], e["E_bipolar"]["runtime_within_construct"]
            lines.append(f"| {dim} | {v['n_with_construct']} | {v['runtime_positive_among_construct']} | "
                         f"{fmt(kw['auroc']) if kw else '–'} | {p} | {fmt(e['E_orth']['syntax']['auroc'])}"
                         f"{fci(e['E_orth']['syntax']['ci'])} | {fmt(rt_o['auroc']) if rt_o else '–'}"
                         f"{fci(rt_o['ci']) if rt_o else ''} | {fmt(rt_b['auroc']) if rt_b else '–'} |")
    lines.append("")


def render(s):
    lines = ["# Stage 293: execution uncertainty and execution semantics in reasoning models", "",
             "Upstream U-Space pipeline unmodified (prompt: explain_result_v1). Higher score = more likely wrong; "
             "AUROC .5 = chance. Intervals resample programs (all seeds) or pairs. See docs/EXECSEM_REASONING.md.", ""]
    fr = s["frozen_terms"]
    if fr:
        lines += [f"Frozen E-Space anchors (on {fr['freeze_on']}, {fr['created']}): " + "; ".join(
            f"{c}: +{v['anchors']} −{v['poles']}" for c, v in fr["terms"].items()),
            f"Dropped: " + (", ".join(f"{w} ({d['rule']})" for w, d in fr["dropped"].items()) or "none") +
            (f". Flags: {fr['flags']}" if fr["flags"] else ""), ""]
    for model, m in s["models"].items():
        lines += [f"## {model}", "", "### Experiment 1: U-Lens and execution errors", ""]
        for ds, r in m["experiment1"].items():
            _exp1(lines, ds, r)
        if m["tokens"]:
            lines += ["### Experiment 1, token level (thinking trace)", ""]
            for ds, t in m["tokens"].items():
                _tokens(lines, ds, t)
        lines += ["### Experiment 2: E-Space", ""]
        if isinstance(m["experiment2"], str):
            lines += [m["experiment2"], ""]
            continue
        if "pairs" in m["experiment2"]:
            lines += ["#### B. Counterfactual pairs (test split)", ""]
            _pairs(lines, m["experiment2"]["pairs"])
        for ds in BENCHMARKS:
            if ds in m["experiment2"]:
                _bench(lines, ds, m["experiment2"][ds])
    return "\n".join(lines) + "\n"
