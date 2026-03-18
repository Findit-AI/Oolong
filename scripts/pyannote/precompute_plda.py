#!/usr/bin/env python3
"""Precompute VBx model parameters from pyannote PLDA + xvec_transform files.

Reads:
  - models/speaker-diarization-community-1/plda/plda.npz       (mu, tr, psi)
  - models/speaker-diarization-community-1/plda/xvec_transform.npz (mean1, mean2, lda)

Writes:
  - models/speaker-diarization-community-1/plda/vbx_model.npz
    Keys: mean1 (256,), mean2 (128,), lda (256,128), plda_mu (128,), plda_tr (128,128), plda_psi (128,)
    All float64.

Replicates the generalized eigenvalue decomposition from pyannote's vbx_setup()
so Rust only needs matrix multiply. The eigendecomposition whitens the PLDA space
and produces positive eigenvalues (plda_psi) suitable for sqrt() in VBx.
"""

from pathlib import Path

import numpy as np
from numpy.linalg import inv
from scipy.linalg import eigh

PLDA_DIR = Path(__file__).resolve().parents[2] / "models" / "speaker-diarization-community-1" / "plda"


def main() -> None:
    # ── Load raw PLDA parameters ──────────────────────────────────────────
    plda = np.load(PLDA_DIR / "plda.npz")
    plda_mu = plda["mu"].astype(np.float64)        # (128,)
    plda_tr = plda["tr"].astype(np.float64)         # (128, 128)
    plda_psi = plda["psi"].astype(np.float64)       # (128,)

    # ── Load xvector transform parameters ─────────────────────────────────
    xvec = np.load(PLDA_DIR / "xvec_transform.npz")
    mean1 = xvec["mean1"].astype(np.float64)        # (256,)
    mean2 = xvec["mean2"].astype(np.float64)        # (128,)
    lda = xvec["lda"].astype(np.float64)            # (256, 128)

    # ── Generalized eigenvalue decomposition (matches pyannote vbx_setup) ─
    # Compute within-class and between-class matrices
    W = inv(plda_tr.T @ plda_tr)
    B = inv((plda_tr.T / plda_psi) @ plda_tr)
    # Solve generalized eigenvalue problem: B v = λ W v
    acvar, wccn = eigh(B, W)
    # Reverse order (largest eigenvalue first) — matches pyannote convention
    plda_psi_new = acvar[::-1]
    plda_tr_new = wccn.T[::-1]

    # Verify eigenvalues are positive (guaranteed by positive definite B, W)
    assert np.all(plda_psi_new > 0), f"plda_psi has non-positive values: min={plda_psi_new.min()}"

    # ── Save combined model ───────────────────────────────────────────────
    out_path = PLDA_DIR / "vbx_model.npz"
    np.savez(
        out_path,
        mean1=mean1,
        mean2=mean2,
        lda=lda,
        plda_mu=plda_mu,
        plda_tr=plda_tr_new,
        plda_psi=plda_psi_new,
    )

    # ── Print verification ────────────────────────────────────────────────
    print(f"Saved: {out_path}")
    saved = np.load(out_path)
    for key in ["mean1", "mean2", "lda", "plda_mu", "plda_tr", "plda_psi"]:
        arr = saved[key]
        print(f"  {key:10s}  shape={str(arr.shape):16s}  dtype={arr.dtype}  min={arr.min():.4f}  max={arr.max():.4f}")


if __name__ == "__main__":
    main()
