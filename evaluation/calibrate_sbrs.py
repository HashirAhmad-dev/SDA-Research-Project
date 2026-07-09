"""
calibrate_sbrs.py
=================
Re-derives the SBRS enforcement parameters (beta, ALERT/BLOCK band cut-points)
from real data instead of the hand-picked defaults.

Reuses the artefacts from the anomaly evaluation -- it does NOT regenerate data:
    evaluation/data/scored_all_splits.csv   (S, A_hybrid, labels, split)
which was produced by `evaluation/train_and_evaluate.py` from the trained
LSTM + IsolationForest on the fixed val/test splits.

The SBRS formula is unchanged (paper Sec. IV):

    SBRS = S * (1 + beta * A_hybrid) / 100

The problem being fixed: at beta = 0.5 the content term S dominates, so 85%+ of
sessions auto-ALERT and ~39% auto-BLOCK purely on content sensitivity, almost
regardless of behaviour. See the "before" column of the printout / report.

Discipline (identical to the anomaly work): everything is chosen on VALIDATION,
the TEST split is scored exactly once at the end.

Step 3 - beta:  chosen to maximise separation of true threats from benign on
                val (ROC-AUC of SBRS vs is_true_threat). Because AUC plateaus
                (malicious insiders are behaviourally invisible and cap it), we
                take the KNEE -- the smallest beta reaching >=99% of the maximum
                achievable AUC -- rather than the noisy argmax, to avoid
                over-distorting the score scale.

Step 4 - bands: derived from where the BENIGN val SBRS scores actually cluster,
                as an explicit false-positive budget (this is precisely the
                quantity the recalibration is trying to fix):
                t_alert = 95th percentile of benign val SBRS  -> the SOC reviews
                          only the top ~5% of normal traffic by score.
                t_block = 99.5th percentile of benign val SBRS -> auto-block only
                          scores essentially never produced by normal traffic.
                Both cut-points sit in the sparse upper tail of the benign
                distribution, so normal sessions clear and anomalies escalate.
                (An F1-maximising t_alert was tried and rejected: on this
                imbalanced, partly-unseparable data it collapses to a very high
                threshold that permits 66% of threats -- useless as a review
                queue. See SBRS_RECALIBRATION.md.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCORED = Path(__file__).resolve().parent / "data" / "scored_all_splits.csv"
OUT_JSON = Path(__file__).resolve().parent / "data" / "sbrs_calibration.json"

# Current (pre-recalibration) config, for the before/after comparison.
OLD_BETA = 0.5
OLD_BANDS = (0.20, 0.60)          # backend/risk_orchestrator.py
DOC_BANDS = (0.50, 1.00)          # Context/.../SBRS.md

# False-positive budgets on BENIGN traffic that define the band cut-points.
ALERT_BENIGN_PCTL = 95.0          # SOC reviews the top ~5% of normal traffic
BLOCK_BENIGN_PCTL = 99.5          # auto-block the top ~0.5%


def sbrs(S: np.ndarray, A: np.ndarray, beta: float) -> np.ndarray:
    return S * (1.0 + beta * A) / 100.0


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos = pos.astype(bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * (~pos).sum()))


def prf(pred: np.ndarray, y: np.ndarray) -> dict:
    pred, y = pred.astype(bool), y.astype(bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f, "fpr": fpr}


def enforcement_metrics(S, A, y, beta, t_alert, t_block, bursts=None) -> dict:
    """PERMIT / ALERT / BLOCK metrics against ground truth."""
    v = sbrs(S, A, beta)
    permit = v < t_alert
    alert = (v >= t_alert) & (v < t_block)
    block = v >= t_block
    flagged = ~permit                              # ALERT or BLOCK = detection
    m = {
        "flagged_f1": prf(flagged, y)["f1"],
        "flagged": prf(flagged, y),
        "block": prf(block, y),
        "auto_alert_or_block_rate": float(flagged.mean()),
        "auto_block_rate": float(block.mean()),
        # the headline "content dominates" numbers: rates on BENIGN traffic only
        "benign_alert_or_block_rate": float(flagged[~y].mean()),
        "benign_block_rate": float(block[~y].mean()),
        "threat_permit_rate": float(permit[y].mean()),   # threats let through
        "n_permit": int(permit.sum()), "n_alert": int(alert.sum()),
        "n_block": int(block.sum()),
    }
    if bursts is not None and bursts.any():
        m["benign_burst_alert_or_block_rate"] = float(flagged[bursts].mean())
        m["benign_burst_block_rate"] = float(block[bursts].mean())
    return m


def derive_bands(S, A, y, beta):
    """Band cut-points = upper-tail percentiles of the BENIGN SBRS distribution.

    A false-positive budget: t_alert lets ~5% of benign traffic into the review
    queue, t_block auto-blocks only the ~0.5% tail. Returns (t_alert, t_block,
    val flagged-F1 at t_alert) computed on the given (validation) split.
    """
    v = sbrs(S, A, beta)
    benign = v[~y.astype(bool)]
    t_alert = float(np.percentile(benign, ALERT_BENIGN_PCTL))
    t_block = float(np.percentile(benign, BLOCK_BENIGN_PCTL))
    val_f1 = prf(v >= t_alert, y)["f1"]
    return t_alert, t_block, val_f1


def main() -> None:
    d = pd.read_csv(SCORED)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))

    def cols(df):
        return (df.pii_sensitivity_score.values.astype(float),
                df.hybrid_anomaly_score.values.astype(float),
                df.is_true_threat.values.astype(bool),
                df.is_benign_burst.values.astype(bool))

    Sv, Av, yv, bv = cols(va)
    St, At, yt, bt = cols(te)
    print(f"val: {len(va)} sessions ({yv.sum()} threats) | "
          f"test: {len(te)} sessions ({yt.sum()} threats)  [scored once]\n")

    # -- Step 2: joint S x A distribution on val --------------------------
    print("[joint S x A on val] mean A by (threat class):")
    for lbl, mask in (("normal benign", ~yv & ~bv), ("benign burst", bv),
                      ("THREAT", yv)):
        print(f"  {lbl:<15} n={int(mask.sum()):<5} "
              f"S mean={Sv[mask].mean():5.1f}  A mean={Av[mask].mean():.3f}")

    # -- Step 3: beta from separation (val AUC), take the knee ------------
    betas = np.round(np.concatenate([np.arange(0, 5.01, 0.5),
                                     np.arange(6, 21, 2)]), 2)
    aucs = {float(b): auc(sbrs(Sv, Av, b), yv) for b in betas}
    auc_max = max(aucs.values())
    knee = min(b for b, a in aucs.items() if a >= 0.99 * auc_max)
    print(f"\n[beta] val AUC(SBRS vs threat): max={auc_max:.4f} at "
          f"beta={max(aucs, key=aucs.get)}; knee (>=99% of max) = beta={knee}")
    beta = knee

    # -- Step 4: bands from val score distribution -----------------------
    t_alert, t_block, val_f1 = derive_bands(Sv, Av, yv, beta)
    print(f"[bands] derived on val: t_alert={t_alert:.3f}  t_block={t_block:.3f} "
          f"(val flagged-F1={val_f1:.3f})")

    # where do val SBRS scores cluster? (sanity: cut-points in a valley)
    v_be = sbrs(Sv[~yv], Av[~yv], beta)
    v_th = sbrs(Sv[yv], Av[yv], beta)
    print(f"        benign SBRS: p50={np.percentile(v_be,50):.3f} "
          f"p90={np.percentile(v_be,90):.3f} p99={np.percentile(v_be,99):.3f}")
    print(f"        threat SBRS: p10={np.percentile(v_th,10):.3f} "
          f"p50={np.percentile(v_th,50):.3f} p90={np.percentile(v_th,90):.3f}")

    # -- Step 6: TEST-only enforcement, new vs old -----------------------
    new = enforcement_metrics(St, At, yt, beta, t_alert, t_block, bursts=bt)
    old = enforcement_metrics(St, At, yt, OLD_BETA, *OLD_BANDS, bursts=bt)
    doc = enforcement_metrics(St, At, yt, OLD_BETA, *DOC_BANDS, bursts=bt)

    # Per-class recall under the new bands (what actually gets caught into review)
    tct = te.threat_class.values
    vt_new = sbrs(St, At, beta)
    per_class = {}
    for cls in ("malicious_insider", "compromised_account",
                "negligent_insider", "overscoped_thirdparty"):
        mask = tct == cls
        if mask.any():
            per_class[cls] = {
                "n": int(mask.sum()),
                "alert_or_block_recall": float((vt_new[mask] >= t_alert).mean()),
                "block_recall": float((vt_new[mask] >= t_block).mean()),
            }

    result = {
        "reused_data": str(SCORED),
        "val_sessions": int(len(va)), "val_threats": int(yv.sum()),
        "test_sessions": int(len(te)), "test_threats": int(yt.sum()),
        "beta": {"chosen": beta, "criterion":
                 "smallest beta reaching >=99% of max val AUC(SBRS vs threat)",
                 "val_auc_by_beta": aucs, "val_auc_max": auc_max},
        "bands": {"t_alert": t_alert, "t_block": t_block,
                  "derivation": "t_alert=%.1fth pctl of benign val SBRS; "
                  "t_block=%.1fth pctl of benign val SBRS"
                  % (ALERT_BENIGN_PCTL, BLOCK_BENIGN_PCTL)},
        "test": {"new_calibrated": new,
                 "old_code_beta0.5_bands0.20_0.60": old,
                 "doc_beta0.5_bands0.50_1.00": doc},
        "test_per_class_recall_new": per_class,
    }
    OUT_JSON.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")

    # -- Console before/after --------------------------------------------
    print("\n" + "=" * 74)
    print(f"TEST enforcement (n={len(te)}, {yt.sum()} threats) - before vs after")
    print("-" * 74)
    hdr = f"{'config':<34}{'enfF1':>8}{'ALERT+BLK%':>12}{'BLOCK%':>9}{'benignFlag%':>12}"
    print(hdr)
    for name, m in (("OLD  beta=0.5 bands 0.20/0.60", old),
                    ("DOC  beta=0.5 bands 0.50/1.00", doc),
                    (f"NEW  beta={beta} bands {t_alert:.2f}/{t_block:.2f}", new)):
        print(f"{name:<34}{m['flagged_f1']:>8.3f}"
              f"{m['auto_alert_or_block_rate']*100:>11.1f}%"
              f"{m['auto_block_rate']*100:>8.1f}%"
              f"{m['benign_alert_or_block_rate']*100:>11.1f}%")
    print("-" * 74)
    print(f"benign-burst ALERT+BLOCK rate: old {old.get('benign_burst_alert_or_block_rate',0)*100:.0f}%"
          f"  ->  new {new.get('benign_burst_alert_or_block_rate',0)*100:.0f}%")
    print(f"threats let through (PERMIT): old {old['threat_permit_rate']*100:.0f}%"
          f"  ->  new {new['threat_permit_rate']*100:.0f}%")
    print("-" * 74)
    print("NEW per-class test recall (ALERT+BLOCK | BLOCK):")
    for cls, pc in per_class.items():
        print(f"  {cls:<22} n={pc['n']:<3} "
              f"{pc['alert_or_block_recall']*100:>3.0f}% | {pc['block_recall']*100:>3.0f}%")
    print("=" * 74)
    print(f"\n  -> {OUT_JSON}")
    print("\nApply to backend/risk_orchestrator.py:")
    print(f"    DEFAULT_BETA = {beta}")
    print(f"    SBRS_BANDS = [({t_block:.2f}, 'HIGH-RISK', 'BLOCK'), "
          f"({t_alert:.2f}, 'SENSITIVE', 'ALERT'), (0.00, 'SAFE', 'PERMIT')]")


if __name__ == "__main__":
    main()
