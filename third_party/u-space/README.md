# Pinned U-Space reference

Source: https://github.com/s2labres/U-Space
Commit: `cce4d2633ed763ba9b33efb11c7099a361209129` (also in `COMMIT`).

`reference_uspace.py` is the unmodified upstream `src/uspace.py`;
`uspace_terms.yaml` is the unmodified upstream `data/uspace_terms.yaml`.
Both are retained with the upstream BSD-3-Clause `LICENSE`.
Stage 290 adapts the numerical construction to this repository's loaded model
and existing J-lens. Tests compare its numerical readout against this reference.
No upstream model weights, data, or Python package dependencies are vendored.
