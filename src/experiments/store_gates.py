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

# The gate sequence, in order. Each entry: what it asserts, and which stage
# is allowed to record it.
GATE_ORDER = ("G0", "G1", "G2", "G3", "G4", "G5")

GATE_MEANING: dict[str, str] = {
    "G0": "generation, alignment, invariants and independent ground truth pass",
    "G1": "the model can solve the programs (behavioural accuracy)",
    "G2": "the text-absent value is decodable above the measured surface controls",
    "G3": "the natural state transition is measurable (frozen-decoder transfer)",
    "G4": "whole-state interchange produces the correctly TRANSFORMED state",
    "G5": "low-rank interchange beats matched controls and transfers across operations",
}

GATE_OWNER: dict[str, str] = {
    "G0": "81_store_verify",
    "G1": "82_store_behaviour",
    "G2": "84_store_decode",
    "G3": "85_store_transition",
    "G4": "86_store_ceiling",
    "G5": "87_store_interchange",
}

# What each stage needs before it is interpretable. Stage 83 (extraction) needs
# G1 because extracting activations for programs the model cannot solve buys
# nothing; stage 86 needs G3 because the trichotomy is read through the frozen
# decoder that G3 validates.
STAGE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "80_store_pairs": (),
    "81_store_verify": (),
    "82_store_behaviour": ("G0",),
    "83_store_extract": ("G0", "G1"),
    "84_store_decode": ("G0", "G1"),
    "85_store_transition": ("G0", "G1", "G2"),
    "86_store_ceiling": ("G0", "G1", "G2", "G3"),
    "87_store_interchange": ("G0", "G1", "G2", "G3", "G4"),
    "88_store_report": (),
}


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


def gates_path(model: str, root: Optional[Path] = None) -> Path:
    return (root or Path("results/store") / model) / "gates.yaml"


def load_gates(model: str, root: Optional[Path] = None) -> dict[str, Gate]:
    path = gates_path(model, root)
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
) -> Gate:
    """Write one gate's verdict. Only the owning stage should call this."""
    if name not in GATE_MEANING:
        raise ValueError(f"unknown gate '{name}'; known: {sorted(GATE_MEANING)}")
    if GATE_OWNER.get(name) != stage:
        logger.warning("gate %s recorded by %s, expected owner %s",
                       name, stage, GATE_OWNER.get(name))
    gates = load_gates(model, root)
    gates[name] = Gate(
        name=name, passed=bool(passed), value=None if value is None else float(value),
        detail=detail, stage=stage,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        extra=dict(extra or {}))
    _write(model, gates, root)
    return gates[name]


def record_override(
    model: str,
    stage: str,
    missing: Sequence[str],
    reason: str,
    root: Optional[Path] = None,
) -> None:
    """Mark the gates a stage ran in spite of. Permanent, and per-gate."""
    gates = load_gates(model, root)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for name in missing:
        gate = gates.get(name) or Gate(name=name, passed=False,
                                       detail="never recorded", stage="")
        gate.override = True
        gate.override_reason = (
            f"{gate.override_reason + '; ' if gate.override_reason else ''}"
            f"{stage} ran anyway at {stamp}: {reason}")
        gates[name] = gate
    _write(model, gates, root)
    logger.warning("OVERRIDE recorded: %s ran with failed/missing gates %s (%s)",
                   stage, list(missing), reason)


def require_gates(
    model: str,
    stage: str,
    override_reason: Optional[str] = None,
    root: Optional[Path] = None,
) -> dict:
    """Refuse to run `stage` unless its prerequisites passed.

    Returns a provenance dict the stage must put in every row it writes and in
    its manifest, so a downstream reader can see the gate state a number was
    produced under without going back to the registry.
    """
    required = STAGE_REQUIREMENTS.get(stage, ())
    gates = load_gates(model, root)
    missing = [name for name in required
               if name not in gates or not gates[name].passed]

    if not missing:
        return {"gates_required": ",".join(required), "gate_override": False,
                "gate_override_reason": ""}

    detail = "; ".join(
        f"{name}: {'FAILED — ' + gates[name].detail if name in gates else 'never recorded'}"
        for name in missing)
    if override_reason:
        record_override(model, stage, missing, override_reason, root)
        return {"gates_required": ",".join(required), "gate_override": True,
                "gate_override_reason": f"{override_reason} [missing: {','.join(missing)}]"}

    raise GateFailure(
        f"{stage} requires {list(required)} and cannot run.\n  {detail}\n"
        f"  Registry: {gates_path(model, root)}\n"
        f"  Fix the upstream stage, or re-run with "
        f"--override-gate 'why you are running this anyway' to record a "
        f"diagnostic run. Overridden runs are marked in every output row.")


def gate_table(model: str, root: Optional[Path] = None) -> list[dict]:
    """The registry as tidy rows, in gate order — what stage 88 reports."""
    gates = load_gates(model, root)
    rows = []
    for name in GATE_ORDER:
        gate = gates.get(name)
        rows.append({
            "gate": name,
            "meaning": GATE_MEANING[name],
            "owner_stage": GATE_OWNER[name],
            "recorded": gate is not None,
            "passed": bool(gate.passed) if gate else False,
            "value": gate.value if gate else None,
            "override": bool(gate.override) if gate else False,
            "detail": gate.detail if gate else "not recorded",
        })
    return rows


def first_blocking_gate(model: str, root: Optional[Path] = None) -> Optional[str]:
    """The earliest gate that has not passed — where the run actually stopped."""
    gates = load_gates(model, root)
    for name in GATE_ORDER:
        if name not in gates or not gates[name].passed:
            return name
    return None


def _write(model: str, gates: dict[str, Gate], root: Optional[Path]) -> None:
    path = gates_path(model, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("E12 is instrument validation. A passing gate says the "
                 "measurement works, not that a scientific claim holds."),
        "gates": {name: gates[name].to_dict() for name in GATE_ORDER if name in gates},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
