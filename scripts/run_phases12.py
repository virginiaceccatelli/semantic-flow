"""Run Phase 1 and Phase 2 end-to-end. Saves results to results/phase1 and results/phase2.

Usage (in semflow env):
    python scripts/run_phases12.py
"""

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/run_phases12.log"),
    ],
)
log = logging.getLogger(__name__)

import torch

log.info("Python: %s", sys.version.split()[0])
log.info("torch: %s", torch.__version__)
log.info("MPS: %s", torch.backends.mps.is_available())

from src.models.loader import ModelConfig, ModelLoader
from src.data.dataset import CodeProbeDataset
from src.experiments.phase1_lexical import run_phase1
from src.experiments.phase2_graph import run_phase2
from src.probes.base import ProbeConfig

config = ModelConfig.from_registry("deepseek-coder-1.3b", device="mps")
loader = ModelLoader(config)
model = loader.model
tokenizer = loader.tokenizer
log.info("Model on: %s", next(model.parameters()).device)

dataset = CodeProbeDataset.load("data/synthetic/phase1_binding.jsonl")
log.info("Dataset: %d examples", len(dataset))

layers = config.probe_layers
log.info("Probing layers: %s", layers)

cfg = ProbeConfig(cv_folds=5, solver="saga", max_iter=100)

# ── Phase 1 ──────────────────────────────────────────────────────────────────
Path("results/phase1").mkdir(parents=True, exist_ok=True)
t0 = time.time()
log.info("=== Phase 1: Lexical & Binding ===")
p1 = run_phase1(model, tokenizer, dataset.examples, layers=layers,
                output_dir="results/phase1", config=cfg)
t1 = time.time()
log.info("Phase 1 done in %.0fs", t1 - t0)

log.info("Phase 1 summary:")
for task, results in p1.items():
    best = max(results, key=lambda r: r.selectivity)
    log.info("  %-35s best_layer=%2d  acc=%.3f  sel=%.3f  auc=%.3f",
             task, best.layer, best.accuracy, best.selectivity, best.auc)

# ── Phase 2 ──────────────────────────────────────────────────────────────────
Path("results/phase2").mkdir(parents=True, exist_ok=True)
t2 = time.time()
log.info("=== Phase 2: Def-Use Edge Recovery ===")
p2 = run_phase2(model, tokenizer, dataset.examples, layers=layers,
                output_dir="results/phase2", config=cfg)
t3 = time.time()
log.info("Phase 2 done in %.0fs", t3 - t2)

log.info("Phase 2 summary:")
for task, results in p2.items():
    best = max(results, key=lambda r: r.selectivity)
    log.info("  %-35s best_layer=%2d  acc=%.3f  sel=%.3f  auc=%.3f",
             task, best.layer, best.accuracy, best.selectivity, best.auc)

log.info("All done. Total: %.0fs", t3 - t0)
