"""Causal value interchange on real CruxEval-derived counterfactuals (239--240).

Each pair changes one single-digit literal inside the unique reaching
definition of a tracked use.  The identifier, use span, surrounding syntax and
token count stay fixed.  Both programs are executed in isolated subprocesses;
the tracked use is instrumented and must receive a different runtime value, and
the recorded output must change.  DAS is learned only on calibration programs
and evaluated on disjoint source functions in both intervention directions.
"""
from __future__ import annotations

import ast
import json
import logging
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl,
                                    register_files, sha256, stage_run,
                                    write_json, write_jsonl)
from src.cruxeval.lens import _continuation, _ids
from src.cruxeval.prepare import load_prepared
from src.data.alignment import (TokenAligner, compute_offsets, decode_exact,
                                line_col_to_char)
from src.data.cruxeval_graph import extract_graph
from src.models.loader import ModelConfig, ModelLoader, load_tokenizer

log = logging.getLogger(__name__)


def _char_col(source: str, line: int, byte_col: int) -> int:
    text = source.splitlines()[line - 1]
    return len(text.encode()[:byte_col].decode())


def _abs_span(source: str, node: ast.AST):
    return (line_col_to_char(source, node.lineno, _char_col(source, node.lineno, node.col_offset)),
            line_col_to_char(source, node.end_lineno,
                             _char_col(source, node.end_lineno, node.end_col_offset)))


def _literal_mutations(code: str, definition: dict):
    """Same-width numeric constants in the assignment owning this definition."""
    tree = ast.parse(code); parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node): parent[child] = node
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Store):
            continue
        col = _char_col(code, node.lineno, node.col_offset)
        if (node.id, node.lineno, col) != (definition["name"], definition["line"], definition["col"]):
            continue
        owner = parent.get(node)
        while owner is not None and not isinstance(owner, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            owner = parent.get(owner)
        value = getattr(owner, "value", None)
        if value is None: continue
        for constant in ast.walk(value):
            if isinstance(constant, ast.Constant) and type(constant.value) is int and 0 <= constant.value <= 9:
                start, end = _abs_span(code, constant)
                if code[start:end] == str(constant.value):
                    matches.append((start, end, constant.value))
    for start, end, value in matches:
        for replacement in range(10):
            if replacement != value:
                yield code[:start] + str(replacement) + code[end:], {
                    "start": start, "end": end, "old": value, "new": replacement}


_TRACE_WORKER = r'''import ast,json,sys
r=json.load(sys.stdin); tree=ast.parse(r["code"]); wanted=(r["line"],r["byte_col"]); traces=[]
class T(ast.NodeTransformer):
 def visit_Name(self,n):
  if isinstance(n.ctx,ast.Load) and (n.lineno,n.col_offset)==wanted:
   return ast.copy_location(ast.Call(ast.Name("__semflow_tap__",ast.Load()),[n],[]),n)
  return n
tree=T().visit(tree); ast.fix_missing_locations(tree)
def tap(x): traces.append(repr(x)); return x
ns={"__semflow_tap__":tap}; exec(compile(tree,"<cruxeval-das>","exec"),ns)
actual=eval("f("+r["input"]+")",ns); expected=ast.literal_eval(r["expected"])
print(json.dumps({"actual":repr(actual),"matches_expected":type(actual) is type(expected) and actual==expected,"traces":traces}))
'''


def _trace(code, arguments, expected, line, char_col, timeout=5):
    text = code.splitlines()[line - 1]; byte_col = len(text[:char_col].encode())
    payload = dict(code=code, input=arguments, expected=expected, line=line, byte_col=byte_col)
    result = subprocess.run([sys.executable, "-I", "-c", _TRACE_WORKER],
                            input=json.dumps(payload), text=True, capture_output=True, timeout=timeout)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _first_divergence(tokenizer, code_a, code_b, arguments, out_a, out_b):
    prompt_a, prefix_a, ids_a = _continuation(tokenizer, code_a, arguments, out_a)
    prompt_b, prefix_b, ids_b = _continuation(tokenizer, code_b, arguments, out_b)
    common = 0
    while common < min(len(ids_a), len(ids_b)) and ids_a[common] == ids_b[common]: common += 1
    if common == len(ids_a) or common == len(ids_b): return None
    input_a, input_b = prefix_a + ids_a[:common], prefix_b + ids_b[:common]
    source_a, source_b = decode_exact(tokenizer, input_a), decode_exact(tokenizer, input_b)
    assert _ids(tokenizer, source_a) == input_a
    assert _ids(tokenizer, source_b) == input_b
    return {"prompt_a": prompt_a, "prompt_b": prompt_b,
            "source_a": source_a, "source_b": source_b,
            "input_a": input_a, "input_b": input_b,
            "base_a": ids_a[common], "base_b": ids_b[common], "common_output_tokens": common}


def prepare_value_pairs(prepared, output, model="deepseek-coder-6.7b", seed=42,
                        min_pairs=20, max_pairs=200, tokenizer=None):
    """Stage 239: create and certify real-code value counterfactuals."""
    meta, programs, _, _, _ = load_prepared(prepared)
    tokenizer = tokenizer or load_tokenizer(meta["model_hf_id"])
    rng = random.Random(seed); candidates = list(programs); rng.shuffle(candidates)
    output = Path(output)
    args = dict(prepared_sha256=sha256(Path(prepared)/"gates.json"), model=model,
                seed=seed, min_pairs=min_pairs, max_pairs=max_pairs)
    with stage_run(output, "239_cruxeval_das_prepare", args) as gate:
        pairs, audit = [], []
        for row in candidates:
            if len(pairs) >= max_pairs: break
            events = row["graph"]["events"]
            made = False
            for site in reversed(row["graph"]["use_sites"]):
                if len(site["reaching_definitions"]) != 1: continue
                use, definition = events[site["use_event"]], events[site["reaching_definitions"][0]]
                for variant, mutation in _literal_mutations(row["code"], definition):
                    try:
                        base = _trace(row["code"], row["input"], row["output"], use["line"], use["col"])
                        changed = _trace(variant, row["input"], row["output"], use["line"], use["col"])
                    except Exception as exc:
                        audit.append({"dataset_id":row["id"],"status":"execution_rejected","detail":str(exc)[:200]})
                        continue
                    if not base["matches_expected"] or len(base["traces"]) != 1 or len(changed["traces"]) != 1: continue
                    if base["actual"] == changed["actual"] or base["traces"] == changed["traces"]: continue
                    divergence = _first_divergence(tokenizer, row["code"], variant, row["input"],
                                                   base["actual"], changed["actual"])
                    if divergence is None: continue
                    source_a, source_b = divergence["source_a"], divergence["source_b"]
                    # Recompute each graph and anchor against its own AST/tokenization.
                    graph_a = extract_graph(row["code"], TokenAligner(source_a, compute_offsets(source_a, tokenizer, divergence["input_a"])))
                    graph_b = extract_graph(variant, TokenAligner(source_b, compute_offsets(source_b, tokenizer, divergence["input_b"])))
                    def anchor(graph):
                        matched = [s for s in graph["use_sites"]
                                   if (graph["events"][s["use_event"]]["name"],
                                       graph["events"][s["use_event"]]["line"],
                                       graph["events"][s["use_event"]]["col"])
                                   == (use["name"], use["line"], use["col"])]
                        assert len(matched) == 1
                        assert len(matched[0]["reaching_definitions"]) == 1
                        return graph["events"][matched[0]["use_event"]]["anchor"]
                    pos_a, pos_b = anchor(graph_a), anchor(graph_b)
                    if pos_a != pos_b or len(divergence["input_a"]) != len(divergence["input_b"]): continue
                    differing = sum(a != b for a,b in zip(divergence["input_a"], divergence["input_b"]))
                    if differing != 1: continue
                    split = "calibration" if int(row["source_group"][:8],16) % 5 < 3 else "test"
                    pairs.append({"pair_id":row["id"],"source_group":row["source_group"],"split":split,
                                  "use_name":use["name"],"use_line":use["line"],"use_col":use["col"],
                                  "position":pos_a,"base_code":row["code"],"variant_code":variant,
                                  "arguments":row["input"],"base_output":base["actual"],
                                  "variant_output":changed["actual"],"base_value":base["traces"][0],
                                  "variant_value":changed["traces"][0],"mutation":mutation,
                                  **divergence})
                    audit.append({"dataset_id":row["id"],"status":"accepted","detail":mutation})
                    made=True; break
                if made: break
        assert len(pairs) >= min_pairs, f"Only {len(pairs)} execution-grounded pairs; need {min_pairs}"
        assert {p["split"] for p in pairs} == {"calibration","test"}
        assert {p["source_group"] for p in pairs if p["split"]=="calibration"}.isdisjoint(
               {p["source_group"] for p in pairs if p["split"]=="test"})
        write_jsonl(output/"pairs.jsonl", pairs); write_jsonl(output/"audit.jsonl", audit)
        write_json(output/"meta.json", {"model":model,"model_hf_id":meta["model_hf_id"],
                   "n_pairs":len(pairs),"n_calibration":sum(p["split"]=="calibration" for p in pairs),
                   "n_test":sum(p["split"]=="test" for p in pairs),
                   "construction":"one-token literal mutation in unique reaching definition; traced use and output both change"})
        register_files(gate, output, ["pairs.jsonl","audit.jsonl","meta.json"])
    return output


def _orientation(pair, reverse=False):
    if not reverse:
        return pair["input_a"], pair["input_b"], pair["base_a"], pair["base_b"], "base_to_variant"
    return pair["input_b"], pair["input_a"], pair["base_b"], pair["base_a"], "variant_to_base"


def run_value_das(prepared, output, layer, rank=1, model="deepseek-coder-6.7b",
                  dtype="float16", device="cuda", steps=200, batch_size=8,
                  lr=.01, seed=42, min_behavior=.60, bootstrap=1000):
    """Stage 240: learn on calibration functions; evaluate both directions on test."""
    from src.models.das import (AlignmentExample, AnswerActuatorExample,
        interchange_report, learn_alignment, learn_answer_actuator,
        make_interchange_fn, mean_difference_subspace, norm_matched_random,
        random_subspace)
    from src.models.hooks import extract_hidden_states, transform_positions

    checked_gate(prepared, "239_cruxeval_das_prepare")
    pairs=read_jsonl(Path(prepared)/"pairs.jsonl"); output=Path(output)
    cfg=ModelConfig.from_registry(model, dtype={"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[dtype], device=device)
    args=dict(prepared_sha256=sha256(Path(prepared)/"gates.json"),layer=layer,rank=rank,
              model=model,dtype=dtype,device=device,steps=steps,batch_size=batch_size,
              lr=lr,seed=seed,min_behavior=min_behavior,bootstrap=bootstrap)
    with stage_run(output,"240_cruxeval_value_das",args) as gate:
        loader=ModelLoader(cfg); mdl=loader.model; d=cfg.d_model
        def state(ids,pos):
            cache=extract_hidden_states(mdl,torch.tensor([ids],device=next(mdl.parameters()).device),[layer])
            return cache.get(layer)[pos].float().numpy()
        cache={}
        for pair in pairs:
            state_a = state(pair["input_a"], pair["position"])
            state_b = state(pair["input_b"], pair["position"])
            cache[(pair["pair_id"],"base_to_variant","host")]=state_a
            cache[(pair["pair_id"],"base_to_variant","donor")]=state_b
            cache[(pair["pair_id"],"variant_to_base","host")]=state_b
            cache[(pair["pair_id"],"variant_to_base","donor")]=state_a
        calibration=[]
        for pair in pairs:
            if pair["split"]!="calibration": continue
            for reverse in (False,True):
                host,donor,base,target,arm=_orientation(pair,reverse)
                calibration.append(AlignmentExample(input_ids=torch.tensor([host]),position=pair["position"],
                    donor_state=cache[(pair["pair_id"],arm,"donor")],target_token_id=target,
                    base_token_id=base,group=pair["source_group"]))
        fit=learn_alignment(mdl,calibration,layer,"use",rank,d,steps,batch_size,lr,seed)
        assert fit.converged and fit.subspace.orthogonality_error()<1e-5
        fit.subspace.save(output/"subspace.pkl")
        # A subspace is sign-invariant.  Use one canonical direction here;
        # averaging both orientations would add every delta and its negation,
        # turning the mean-difference baseline into the zero vector.
        deltas=[cache[(p["pair_id"],"base_to_variant","donor")]
                -cache[(p["pair_id"],"base_to_variant","host")]
                for p in pairs if p["split"]=="calibration"]
        mean_basis=mean_difference_subspace(deltas); random_basis=random_subspace(d,rank,seed)
        actuator_examples=[]
        for pair in pairs:
            if pair["split"] != "calibration":
                continue
            for reverse in (False, True):
                host, donor, base, target, arm = _orientation(pair, reverse)
                donor_state = cache[(pair["pair_id"], arm, "donor")]
                rep = interchange_report(cache[(pair["pair_id"], arm, "host")],
                                         donor_state, fit.subspace.basis)
                actuator_examples.append(AnswerActuatorExample(
                    torch.tensor([host]), pair["position"], target, base,
                    rep["edit_norm"], pair["source_group"]))
        actuator=learn_answer_actuator(mdl,actuator_examples,layer,d,steps,batch_size,lr,seed)
        assert actuator.converged
        with open(output/"answer_actuator.pkl","wb") as f: pickle.dump(actuator,f)
        rows=[]
        for pair in pairs:
            if pair["split"]!="test": continue
            for reverse in (False,True):
                host,donor,base,target,arm=_orientation(pair,reverse); pos=pair["position"]
                h=cache[(pair["pair_id"],arm,"host")]; other=cache[(pair["pair_id"],arm,"donor")]
                ids=torch.tensor([host],device=next(mdl.parameters()).device)
                clean=mdl(input_ids=ids,use_cache=False).logits[0,-1].float()
                clean_ld=float(clean[target]-clean[base]); clean_correct=int(clean[base]>clean[target])
                das_rep=interchange_report(h,other,fit.subspace.basis)
                matched_basis,_=norm_matched_random(h,other,das_rep["edit_fraction"],d,rank,
                                                    seed+int(pair["source_group"][:6],16)%10000)
                variants={"das_value":fit.subspace.basis,"mean_difference":mean_basis,
                          "random_rank":random_basis,"random_norm":matched_basis,
                          "noop":fit.subspace.basis,"whole_state":None}
                for name,basis in variants.items():
                    donor_state=h if name=="noop" else other
                    logits=transform_positions(mdl,ids,{layer:{pos:make_interchange_fn(basis,donor_state)}})[0,-1].float()
                    rep=interchange_report(h,donor_state,basis)
                    rows.append(_das_row(pair,arm,name,base,target,clean_ld,clean_correct,logits,rep))
                va,vb=actuator.vectors.get(base),actuator.vectors.get(target)
                if va is not None and vb is not None:
                    direction=vb-va; direction=direction/(np.linalg.norm(direction)+1e-12)
                    synthetic=h+das_rep["edit_norm"]*direction
                    logits=transform_positions(mdl,ids,{layer:{pos:make_interchange_fn(direction[:,None],synthetic)}})[0,-1].float()
                    rows.append(_das_row(pair,arm,"answer_actuator",base,target,clean_ld,clean_correct,logits,
                                         interchange_report(h,synthetic,direction[:,None])))
        frame=pd.DataFrame(rows); behavior=float(frame.drop_duplicates(["pair_id","arm"]).clean_correct.mean())
        assert behavior>=min_behavior,f"Clean forced-choice behavior {behavior:.3f} below {min_behavior}"
        assert (frame[frame.variant=="noop"].edit_norm==0).all()
        das_dose = frame[frame.variant == "das_value"].set_index(["pair_id", "arm"])["edit_fraction"]
        random_dose = frame[frame.variant == "random_norm"].set_index(["pair_id", "arm"])["edit_fraction"]
        common = das_dose.index.intersection(random_dose.index)
        assert (random_dose.loc[common].to_numpy() + 1e-8 >=
                das_dose.loc[common].to_numpy()).all(), "Magnitude control undershot DAS"
        frame.to_csv(output/"interchange_rows.csv",index=False)
        summary=_das_summary(frame, seed=seed, n_boot=bootstrap)
        summary.to_csv(output/"interchange_summary.csv",index=False)
        write_json(output/"meta.json",{"model":model,"layer":layer,"rank":rank,
                   "clean_behavior":behavior,"fit_converged":fit.converged,
                   "actuator_converged":actuator.converged,"fit_history":fit.history,
                   "answer_actuator_test_coverage":float(
                       (frame.variant == "answer_actuator").sum() /
                       frame[frame.variant == "das_value"].shape[0]),
                   "calibration_groups":sorted({e.group for e in calibration}),
                   "test_groups":sorted({p["source_group"] for p in pairs if p["split"]=="test"})})
        register_files(gate,output,["subspace.pkl","answer_actuator.pkl","interchange_rows.csv",
                                   "interchange_summary.csv","meta.json"])
    return output


def _cluster_ci(part, column, seed, n_boot):
    groups = list(part.source_group.unique())
    assert groups
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        values = np.concatenate([part.loc[part.source_group == group, column].to_numpy()
                                 for group in chosen])
        means.append(float(values.mean()))
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def _das_summary(frame, seed=42, n_boot=1000):
    """Tidy per-arm estimates with source-program bootstrap intervals."""
    rows = []
    for n, ((variant, arm), part) in enumerate(frame.groupby(["variant", "arm"], sort=True)):
        row = {"variant": variant, "arm": arm, "n": len(part),
               "n_programs": part.source_group.nunique(),
               "install_rate": float(part.says_target.mean()),
               "forced_target_rate": float(part.forced_target.mean()),
               "mean_delta_logit_diff": float(part.delta_logit_diff.mean()),
               "mean_edit_fraction": float(part.edit_fraction.mean())}
        for column, prefix in (("says_target", "install_rate"),
                               ("forced_target", "forced_target_rate"),
                               ("delta_logit_diff", "mean_delta_logit_diff")):
            lo, hi = _cluster_ci(part, column, seed + n * 17, n_boot)
            row[prefix + "_ci_low"], row[prefix + "_ci_high"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def _das_row(pair,arm,variant,base,target,clean_ld,clean_correct,logits,rep):
    patched=float(logits[target]-logits[base])
    return {"pair_id":pair["pair_id"],"source_group":pair["source_group"],"arm":arm,
            "variant":variant,"base_token_id":base,"target_token_id":target,
            "clean_logit_diff":clean_ld,"patched_logit_diff":patched,
            "delta_logit_diff":patched-clean_ld,"clean_correct":clean_correct,
            "says_target":int(int(logits.argmax())==target),"forced_target":int(logits[target]>logits[base]),
            **rep}


def report_value_das(runs, output):
    """Stage 241: combine independently fitted layer runs into one tidy report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_paths = [Path(path) for path in runs]
    gates = [checked_gate(path, "240_cruxeval_value_das") for path in run_paths]
    prep = {gate["args"]["prepared_sha256"] for gate in gates}
    assert len(prep) == 1, "DAS layer runs use different counterfactual populations"
    layers = [int(gate["args"]["layer"]) for gate in gates]
    assert len(layers) == len(set(layers)), "Duplicate DAS layer run"
    output = Path(output)
    args = {"runs": [{"path": str(path), "gate_sha256": sha256(path / "gates.json")}
                     for path in run_paths]}
    with stage_run(output, "241_cruxeval_value_das_report", args) as gate:
        summaries, rows = [], []
        for path, layer in zip(run_paths, layers):
            summary = pd.read_csv(path / "interchange_summary.csv")
            summary.insert(0, "layer", layer); summaries.append(summary)
            raw = pd.read_csv(path / "interchange_rows.csv")
            raw.insert(0, "layer", layer); rows.append(raw)
        summary = pd.concat(summaries, ignore_index=True).sort_values(
            ["layer", "variant", "arm"])
        raw = pd.concat(rows, ignore_index=True).sort_values(
            ["layer", "pair_id", "arm", "variant"])
        summary.to_csv(output / "interchange_summary.csv", index=False)
        raw.to_csv(output / "interchange_rows.csv.gz", index=False,
                   compression={"method": "gzip", "mtime": 0})

        both = summary.groupby(["layer", "variant"], as_index=False).agg(
            forced_target_rate=("forced_target_rate", "mean"),
            install_rate=("install_rate", "mean"),
            mean_delta_logit_diff=("mean_delta_logit_diff", "mean"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        keep = {"das_value", "mean_difference", "random_rank", "random_norm",
                "whole_state", "answer_actuator"}
        for variant, part in both[both.variant.isin(keep)].groupby("variant"):
            axes[0].plot(part.layer, part.forced_target_rate, marker="o", label=variant)
            axes[1].plot(part.layer, part.mean_delta_logit_diff, marker="o", label=variant)
        axes[0].set(xlabel="layer", ylabel="counterfactual forced-choice rate", ylim=(0, 1))
        axes[1].axhline(0, color="black", linewidth=.8)
        axes[1].set(xlabel="layer", ylabel="change in target-minus-base logit")
        axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(output / "cruxeval_value_das.png", dpi=180); plt.close(fig)

        best = both[both.variant == "das_value"].sort_values(
            "forced_target_rate", ascending=False).head(1)
        text = ["# CruxEval value interchange", "",
                "DAS was fitted on calibration source functions and frozen before evaluation on disjoint source functions.", "",
                best.to_markdown(index=False) if len(best) else "No DAS rows.", "",
                "Interpret the learned intervention against the magnitude-matched random, mean-difference, answer-actuator, no-op, and whole-state arms. A positive result requires held-out movement toward the donor output beyond the matched controls; stage completion alone is not a positive result."]
        (output / "report.md").write_text("\n".join(text) + "\n")
        register_files(gate, output, ["interchange_summary.csv", "interchange_rows.csv.gz",
                                      "cruxeval_value_das.png", "report.md"])
    return output
