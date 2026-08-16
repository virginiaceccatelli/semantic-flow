"""E12 gate registry: what has passed, and what may therefore run.

The E10/E11 tracks gate informally — stage 60 exits non-zero, and a job script
chains stages with `|| exit 1`. That is enough when the whole track runs in one
job and nothing else. It was not enough in practice: stage 72 was not re-run
before stage 73, no frozen probes were on disk, and the `probe_basis` control
was silently skipped rather than refused. `results/STATUS.yaml` records that as
an outstanding item because nothing in the pipeline noticed at the time.

E12 makes the dependency explicit and persistent. Every stage declares the
gates it requires; `require_gates` reads the registry on disk and refuses to
run unless they have all passed. A gate is only ever recorded by the stage that
measured it.

**The override is not an escape hatch, it is a marked one.** Running a stage
with `--override-gate REASON` is permitted — a diagnostic that needs to see
what a downstream stage does when an upstream gate failed is a legitimate and
frequent need — but it is written into the gate file, into every row the stage
emits, and into the run manifest. A number produced under an override can never
be mistaken later for one produced under a passing gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import yaml

logger = logging.getLogger(__name__)

# ── gate specifications ──────────────────────────────────────────────────────
# One mechanism, one spec per experiment. E12 (store transitions), E13 (binding
# interchange) and E15 (source→sink under obfuscation) have different gates but
# identical semantics: a stage declares what it needs, the registry says what
# has passed, and a stage that is not entitled to run refuses rather than
# producing an uninterpretable number.


@dataclass(frozen=True)
class GateSpec:
    """The gates of one experiment: their order, meaning, owner and consumers."""

    experiment: str
    order: tuple[str, ...]
    meaning: dict[str, str]
    owner: dict[str, str]
    requirements: dict[str, tuple[str, ...]]
    default_root: str

    def root_for(self, model: str) -> Path:
        return Path(self.default_root) / model


STORE = GateSpec(
    experiment="E12",
    order=("G0", "G1", "G2", "G3", "G4", "G5"),
    meaning={
        "G0": "generation, alignment, invariants and independent ground truth pass",
        "G1": "the model can solve the programs (behavioural accuracy)",
        "G2": "the text-absent value is decodable above the measured surface controls",
        "G3": "the natural state transition is measurable (frozen-decoder transfer)",
        "G4": "whole-state interchange produces the correctly TRANSFORMED state",
        "G5": "low-rank interchange beats matched controls and transfers across operations",
    },
    owner={
        "G0": "81_store_verify", "G1": "82_store_behaviour", "G2": "84_store_decode",
        "G3": "85_store_transition", "G4": "86_store_ceiling", "G5": "87_store_interchange",
    },
    # Stage 83 (extraction) needs G1 because extracting activations for programs
    # the model cannot solve buys nothing; stage 86 needs G3 because the
    # trichotomy is read through the frozen decoder that G3 validates.
    requirements={
        "80_store_pairs": (), "81_store_verify": (),
        "82_store_behaviour": ("G0",), "83_store_extract": ("G0", "G1"),
        "84_store_decode": ("G0", "G1"), "85_store_transition": ("G0", "G1", "G2"),
        "86_store_ceiling": ("G0", "G1", "G2", "G3"),
        "87_store_interchange": ("G0", "G1", "G2", "G3", "G4"),
        "88_store_report": (),
    },
    default_root="results/store",
)

BINDING = GateSpec(
    experiment="E13",
    order=("H0", "H1", "H2", "H3", "H4", "H5"),
    meaning={
        "H0": "the four-program factorial verifies: invariants, alignment, execution truth",
        "H1": "the model returns the correctly bound variable (behavioural accuracy)",
        "H2": "which definition is in scope is decodable at the use anchor",
        "H3": "whole-state interchange flips the answer — the ceiling, per arm",
        "H4": "low-rank interchange beats matched controls on the TRAINING arm",
        "H5": "the same subspace transfers to the HELD-OUT value assignment, "
              "where an answer direction cannot",
    },
    owner={
        "H0": "101_binding_verify", "H1": "102_binding_behaviour",
        "H2": "104_binding_decode", "H3": "105_binding_ceiling",
        "H4": "106_binding_interchange", "H5": "106_binding_interchange",
    },
    requirements={
        "100_binding_pairs": (), "101_binding_verify": (),
        "102_binding_behaviour": ("H0",), "103_binding_extract": ("H0", "H1"),
        "104_binding_decode": ("H0", "H1"), "105_binding_ceiling": ("H0", "H1", "H2"),
        "106_binding_interchange": ("H0", "H1", "H2", "H3"),
        "107_binding_report": (),
    },
    default_root="results/binding",
)

SINKFLOW = GateSpec(
    experiment="E15",
    order=("S0", "S1", "S2", "S3"),
    meaning={
        "S0": "the benchmark is exactly the designed one: counts, balance, splits, "
              "parsing, anchor alignment, independently recovered labels, pair "
              "invariants, and an obfuscation ladder that preserves every label",
        "S1": "activations exist for every program, and the source and sink anchors "
              "land on token boundaries in the encoding that was actually stored",
        "S2": "the readout is fitted on CLEAN TRAINING programs only, with its "
              "selectivity control and its no-hidden-state surface baseline both run",
        "S3": "the frozen readout was evaluated on held-out clean text and on every "
              "obfuscation level, with both classes present in every reported cell",
    },
    owner={
        "S0": "120_sinkflow_generate", "S1": "121_sinkflow_extract",
        "S2": "122_sinkflow_probe", "S3": "123_sinkflow_obfuscation",
    },
    requirements={
        "120_sinkflow_generate": (),
        "121_sinkflow_extract": ("S0",),
        "122_sinkflow_probe": ("S0", "S1"),
        "123_sinkflow_obfuscation": ("S0", "S1", "S2"),
        "124_sinkflow_report": (),
    },
    default_root="results/sinkflow",
)

SPECS = {spec.experiment: spec for spec in (STORE, BINDING, SINKFLOW)}

# Kept so E12's existing call sites and tests read unchanged. New code should
# take a GateSpec rather than reach for these.
GATE_ORDER = STORE.order
GATE_MEANING = STORE.meaning
GATE_OWNER = STORE.owner
STAGE_REQUIREMENTS = STORE.requirements


def spec_for_stage(stage: str) -> GateSpec:
    """The spec that owns a stage name. Keeps call sites from having to say."""
    for spec in SPECS.values():
        if stage in spec.requirements:
            return spec
    raise ValueError(f"no gate spec declares stage '{stage}'")


class GateFailure(RuntimeError):
    """Raised when a stage's prerequisites have not passed."""


@dataclass
class Gate:
    name: str
    passed: bool
    value: Optional[float] = None
    detail: str = ""
    stage: str = ""
    timestamp: str = ""
    override: bool = False
    override_reason: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": bool(self.passed), "value": self.value,
                "detail": self.detail, "stage": self.stage,
                "timestamp": self.timestamp, "override": bool(self.override),
                "override_reason": self.override_reason, "extra": dict(self.extra)}


def gates_path(model: str, root: Optional[Path] = None,
               spec: GateSpec = STORE) -> Path:
    return (root or spec.root_for(model)) / "gates.yaml"


def load_gates(model: str, root: Optional[Path] = None,
               spec: GateSpec = STORE) -> dict[str, Gate]:
    path = gates_path(model, root, spec)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {name: Gate(**entry) for name, entry in (raw.get("gates") or {}).items()}


def record_gate(
    model: str,
    name: str,
    passed: bool,
    detail: str,
    stage: str,
    value: Optional[float] = None,
    extra: Optional[dict] = None,
    root: Optional[Path] = None,
    spec: Optional[GateSpec] = None,
) -> Gate:
    """Write one gate's verdict. Only the owning stage should call this."""
    spec = spec or spec_for_stage(stage)
    if name not in spec.meaning:
        raise ValueError(f"unknown gate '{name}' for {spec.experiment}; "
                         f"known: {sorted(spec.meaning)}")
    if spec.owner.get(name) != stage:
        logger.warning("gate %s recorded by %s, expected owner %s",
                       name, stage, spec.owner.get(name))
    gates = load_gates(model, root, spec)
    gates[name] = Gate(
        name=name, passed=bool(passed), value=None if value is None else float(value),
        detail=detail, stage=stage,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        extra=dict(extra or {}))
    _write(model, gates, root, spec)
    return gates[name]


def record_override(
    model: str,
    stage: str,
    missing: Sequence[str],
    reason: str,
    root: Optional[Path] = None,
    spec: Optional[GateSpec] = None,
) -> None:
    """Mark the gates a stage ran in spite of. Permanent, and per-gate."""
    spec = spec or spec_for_stage(stage)
    gates = load_gates(model, root, spec)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for name in missing:
        gate = gates.get(name) or Gate(name=name, passed=False,
                                       detail="never recorded", stage="")
        gate.override = True
        gate.override_reason = (
            f"{gate.override_reason + '; ' if gate.override_reason else ''}"
            f"{stage} ran anyway at {stamp}: {reason}")
        gates[name] = gate
    _write(model, gates, root, spec)
    logger.warning("OVERRIDE recorded: %s ran with failed/missing gates %s (%s)",
                   stage, list(missing), reason)


def require_gates(
    model: str,
    stage: str,
    override_reason: Optional[str] = None,
    root: Optional[Path] = None,
    spec: Optional[GateSpec] = None,
) -> dict:
    """Refuse to run `stage` unless its prerequisites passed.

    Returns a provenance dict the stage must put in every row it writes and in
    its manifest, so a downstream reader can see the gate state a number was
    produced under without going back to the registry.
    """
    spec = spec or spec_for_stage(stage)
    required = spec.requirements.get(stage, ())
    gates = load_gates(model, root, spec)
    missing = [name for name in required
               if name not in gates or not gates[name].passed]

    if not missing:
        return {"gates_required": ",".join(required), "gate_override": False,
                "gate_override_reason": ""}

    detail = "; ".join(
        f"{name}: {'FAILED — ' + gates[name].detail if name in gates else 'never recorded'}"
        for name in missing)
    if override_reason:
        record_override(model, stage, missing, override_reason, root, spec)
        return {"gates_required": ",".join(required), "gate_override": True,
                "gate_override_reason": f"{override_reason} [missing: {','.join(missing)}]"}

    raise GateFailure(
        f"{stage} requires {list(required)} and cannot run.\n  {detail}\n"
        f"  Registry: {gates_path(model, root, spec)}\n"
        f"  Fix the upstream stage, or re-run with "
        f"--override-gate 'why you are running this anyway' to record a "
        f"diagnostic run. Overridden runs are marked in every output row.")


def gate_table(model: str, root: Optional[Path] = None,
               spec: GateSpec = STORE) -> list[dict]:
    """The registry as tidy rows, in gate order — what the report stage reads."""
    gates = load_gates(model, root, spec)
    rows = []
    for name in spec.order:
        gate = gates.get(name)
        rows.append({
            "gate": name,
            "meaning": spec.meaning[name],
            "owner_stage": spec.owner[name],
            "recorded": gate is not None,
            "passed": bool(gate.passed) if gate else False,
            "value": gate.value if gate else None,
            "override": bool(gate.override) if gate else False,
            "detail": gate.detail if gate else "not recorded",
        })
    return rows


def first_blocking_gate(model: str, root: Optional[Path] = None,
                        spec: GateSpec = STORE) -> Optional[str]:
    """The earliest gate that has not passed — where the run actually stopped."""
    gates = load_gates(model, root, spec)
    for name in spec.order:
        if name not in gates or not gates[name].passed:
            return name
    return None


def _write(model: str, gates: dict[str, Gate], root: Optional[Path],
           spec: GateSpec = STORE) -> None:
    path = gates_path(model, root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": spec.experiment,
        "model": model,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("A passing gate says the measurement works at that step. It is "
                 "not itself a scientific claim."),
        "gates": {name: gates[name].to_dict() for name in spec.order if name in gates},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
