"""
refresh_sbrs_columns.py
=======================
Re-derive every SBRS-dependent column and metric at the calibrated beta / bands,
without retraining anything.

`calibrate_sbrs.py` moved beta 0.5 -> 2.5 and the ALERT/BLOCK cut-points
0.20/0.60 -> 1.22/1.84. But the CSVs and `metrics.json` on disk were written by
the *earlier* run of `train_and_evaluate.py`, so their `sbrs_value`,
`sbrs_category`, `hybrid_action` columns and their enforcement metrics were all
still at the old beta. The dashboard reads those files, so it was drawing the new
band lines across old scores.

Nothing here is re-estimated. SBRS is a pure function of two columns that are
already on disk:

    SBRS = S * (1 + beta * A_hybrid) / 100

so recomputing it from the stored `pii_sensitivity_score` and
`hybrid_anomaly_score` reproduces exactly what a full re-run of
`train_and_evaluate.py` would write - the models, their scores and the splits are
untouched. Only the enforcement layer downstream of them moves.

    python -m evaluation.refresh_sbrs_columns
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import (CSV_COLUMNS, PAPER_DOC_BANDS, SBRS_BANDS, SBRS_BETA,
                     band, binary_metrics, sbrs)
from .train_and_evaluate import METRICS_JSON, SCORED_ALL, V2_CSV


def _apply(df: pd.DataFrame) -> pd.DataFrame:
    vals = np.array([
        sbrs(float(s), float(a), SBRS_BETA)
        for s, a in zip(df.pii_sensitivity_score, df.hybrid_anomaly_score)
    ])
    bands = [band(v) for v in vals]
    df = df.copy()
    df["sbrs_value"] = np.round(vals, 4)
    df["sbrs_category"] = [b[0] for b in bands]
    df["hybrid_action"] = [b[1] for b in bands]
    return df


def main() -> None:
    all_df = pd.read_csv(SCORED_ALL)
    old_alert = float((all_df.hybrid_action != "PERMIT").mean())
    old_block = float((all_df.hybrid_action == "BLOCK").mean())

    all_df = _apply(all_df)
    all_df.to_csv(SCORED_ALL, index=False)

    test = all_df[all_df.split == "test"]
    test[list(CSV_COLUMNS)].to_csv(V2_CSV, index=False)

    y = test.is_true_threat.astype(bool).to_numpy()
    act = test.hybrid_action.to_numpy()
    doc = np.array([band(v, PAPER_DOC_BANDS)[1] for v in test.sbrs_value])

    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    m["sbrs_enforcement_config"] = {
        "beta": SBRS_BETA,
        "bands": [[t, c, a] for t, c, a in SBRS_BANDS],
        "source": "evaluation/calibrate_sbrs.py (validation-calibrated); "
                  "see evaluation/SBRS_RECALIBRATION.md",
    }
    m["test_enforcement_bands"]["hybrid_alert_or_block"] = binary_metrics(y, act != "PERMIT")
    m["test_enforcement_bands"]["hybrid_block_only"] = binary_metrics(y, act == "BLOCK")
    m["test_enforcement_bands"]["paper_doc_bands_alert_or_block"] = binary_metrics(
        y, doc != "PERMIT")

    bb = test[test.is_benign_burst.astype(bool)]
    m["benign_burst"]["hybrid_action_counts"] = {
        k: int(v) for k, v in bb.hybrid_action.value_counts().items()}
    m["benign_burst"]["mean_sbrs"] = float(bb.sbrs_value.mean())

    METRICS_JSON.write_text(json.dumps(m, indent=2), encoding="utf-8")

    enf = m["test_enforcement_bands"]["hybrid_alert_or_block"]
    print(f"beta={SBRS_BETA}  bands={[b[0] for b in SBRS_BANDS[:2]]}")
    print(f"  ALERT+BLOCK rate (all splits) : {old_alert:.1%} -> "
          f"{(all_df.hybrid_action != 'PERMIT').mean():.1%}")
    print(f"  auto-BLOCK rate  (all splits) : {old_block:.1%} -> "
          f"{(all_df.hybrid_action == 'BLOCK').mean():.1%}")
    print(f"  TEST enforcement F1 (ALERT or BLOCK) : {enf['f1']:.3f} "
          f"(precision {enf['precision']:.3f}, recall {enf['recall']:.3f})")
    print(f"\n  -> {SCORED_ALL}\n  -> {V2_CSV}\n  -> {METRICS_JSON}")


if __name__ == "__main__":
    main()
