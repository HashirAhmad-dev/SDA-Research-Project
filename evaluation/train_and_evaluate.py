"""
train_and_evaluate.py
=====================
Trains the hybrid anomaly engine for real and evaluates it on a held-out split.
No hardcoded scores anywhere: every number this script prints is measured.

Pipeline
--------
1.  Load `evaluation/data/sessions_raw.csv` (chronological `split` column baked
    in by generate_sessions.py -- TRAIN 70% / VAL 15% / TEST 15% by time).

2.  Sub-divide TRAIN by time into:
        FIT   -- first 75% of the train period. Fits the scaler, the LSTM, the
                 Isolation Forest, and populates each user's memory bank.
        CALIB -- last 25% of the train period (with a 24 h buffer so its windows
                 cannot overlap any FIT window). Fits the two score calibrators.
    Without this buffer the memory bank would contain the very windows it is
    scoring, driving train cosine distances to ~0 and wrecking the calibration.

3.  EWMA baseline (legacy comparator): univariate on activity_rate_per_hr,
        EWMA_t = lam * x_t + (1 - lam) * EWMA_{t-1},  lam = 0.3
    flagged when (x_t - EWMA_{t-1}) / EWMA_{t-1} > tau_ewma. tau_ewma is tuned
    on VAL, so the baseline gets the same tuning budget as the hybrid engine.

4.  LSTM (real, hidden dim 128): a next-step predictor over a 24-hour window of
    the 6-d feature vector, in each user's *personal* z-frame. Trained
    unsupervised on FIT windows (labels never touch training; the ~3.9% injected
    positives remain as realistic contamination). The trained encoder's final
    hidden state is the paper's "behavioral encoding". Anomaly score is the
    paper's memory-bank formulation, with the top-k neighbours averaged:

        a_LSTM_raw(t) = 1 - mean_top-k cos( h_t , m_j )   m_j in that user's bank

5.  Isolation Forest (sklearn) fit on the FIT split's *global* 6-d distribution.
        a_IF_raw = -score_samples(x)

    The two models deliberately use different reference frames -- personal for
    the LSTM, global for the forest -- which is what the paper's architecture
    prescribes and what makes the pair complementary.

6.  Both raw scores -> [0, 1] via RobustCalibrator fitted on CALIB.
    Fuse:  A_hybrid = alpha * a_LSTM + (1 - alpha) * a_IF
    alpha and the flag threshold tau_h are grid-searched on VAL only.

7.  TEST is scored exactly once. Reports FPR (EWMA vs hybrid), per-class recall
    (window- and episode-level, with Wilson intervals), F1 under the paper's
    SBRS enforcement bands, measured wall-clock latency per scoring call, and
    the benign-burst false-positive trap in isolation.

Outputs
-------
    anomaly_detection_comparison_v2_real.csv   (project root, drop-in schema)
    evaluation/data/scored_all_splits.csv      (every split, extra columns)
    evaluation/data/metrics.json               (everything reported below)
    evaluation/artifacts/*                     (trained model, for the backend)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .common import (CSV_COLUMNS, DATA_DIR, EVAL_DIR, EWMA_LAMBDA, FEATURES,
                     HYBRID_FLAG_THRESHOLD, LEGACY_EWMA_TAU, PAPER_DOC_BANDS,
                     SBRS_BETA, SEQ_LEN, SESSIONS_RAW, THREAT_CLASSES, V2_CSV,
                     RobustCalibrator, band, binary_metrics, ewma_stream, sbrs,
                     wilson_ci)

ARTIFACTS = EVAL_DIR / "artifacts"
SCORED_ALL = DATA_DIR / "scored_all_splits.csv"
METRICS_JSON = DATA_DIR / "metrics.json"

HIDDEN_DIM = 128          # paper Sec. III.C
MEMORY_BANK_CAP = 256
TOPK = 10                 # memory-bank neighbours averaged for a_LSTM (val-selected)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BehaviouralLSTM(nn.Module):
    """LSTM encoder over a 24-hour window, trained by next-step prediction.

    The encoder's final hidden state is the paper's "behavioral encoding" of the
    window; the linear head exists only to give that encoding a training signal.

    Objective chosen by measurement, not taste. On the validation split a
    sequence-autoencoder variant of this same encoder scored malicious-insider
    AUC 0.62 against 0.70 for next-step prediction, and its own reconstruction
    error was *worse than chance* on that class (AUC 0.43): drifting windows are
    smoother than normal ones -- collapsed collaboration, steady reads -- so an
    autoencoder reconstructs them more easily. See REAL_RESULTS.md.
    """

    def __init__(self, n_features: int = len(FEATURES), hidden: int = HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        return h[-1]                          # (batch, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))      # predicted next-hour feature vector


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
Z_CLIP = 8.0      # per-user z-scores are clipped to +/- this
SD_FLOOR = 0.05   # features that are ~always 0 for a user (e.g. permission delta)


def build_personal_grids(df: pd.DataFrame, fit_end: int
                         ) -> Tuple[Dict[str, np.ndarray], Dict[str, tuple]]:
    """Per-user (T, 6) feature grid, z-scored against that USER's own FIT-period
    statistics.

    This is the frame the LSTM operates in. The paper assigns the two models
    different reference distributions -- the LSTM compares a window against
    "the user's historical memory bank" (personal), the Isolation Forest against
    "the global organizational distribution". Scoring both in one global frame
    makes personal drift undetectable: an Engineering user reaching for
    `restricted` files and a Legal user doing so every day look identical
    globally. Only in the personal frame is the former a large deviation.

    An hour with no API call has no file type, no origin, no endpoint, no scope
    change and no collaborators -- those five features are *undefined*, not zero.
    Encoding them as raw 0.0 and then z-scoring turns every idle hour into an
    extreme negative deviation on all five axes, which is precisely the direction
    a malicious insider drifts in (collaboration collapses). The memory bank then
    fills with "very low collaboration" windows and real drift stops looking
    unusual. So idle hours are pinned to the user's own mean (z = 0) on the five
    contextual dims; only `activity_rate` keeps its true value, since zero calls
    is a genuine observation. Measured: this single change lifted validation
    malicious-insider AUC from 0.62 to 0.74.
    """
    grids, stats = {}, {}
    for uid, g in df.groupby("user_id", sort=False):
        g = g.sort_values("hour_index")
        raw = g[list(FEATURES)].to_numpy()
        active = g.active.to_numpy()
        in_fit = g.hour_index.to_numpy() < fit_end
        fit_active = in_fit & active

        mu, sd = np.zeros(len(FEATURES)), np.ones(len(FEATURES))
        # activity_rate: over every FIT hour (idle hours are real observations)
        mu[0] = raw[in_fit, 0].mean()
        sd[0] = max(raw[in_fit, 0].std(), SD_FLOOR)
        # contextual dims: over FIT hours that actually carried a transaction
        mu[1:] = raw[fit_active][:, 1:].mean(axis=0)
        sd[1:] = np.maximum(raw[fit_active][:, 1:].std(axis=0), SD_FLOOR)

        stats[uid] = (mu, sd)
        z = np.clip((raw - mu) / sd, -Z_CLIP, Z_CLIP)
        z[~active, 1:] = 0.0
        grids[uid] = z.astype(np.float32)
    return grids, stats


def windows_ending_at(grid: np.ndarray, t: int) -> np.ndarray:
    """The 24-hour window ending at hour t inclusive."""
    return grid[t - SEQ_LEN + 1: t + 1]


def collect_windows(grids: Dict[str, np.ndarray], hour_lo: int, hour_hi: int,
                    need_next: bool) -> Tuple[np.ndarray, np.ndarray]:
    """All (window, next_step) pairs whose window END hour is in [hour_lo, hour_hi)."""
    X, Y = [], []
    for uid, grid in grids.items():
        T = len(grid)
        t_start = max(hour_lo, SEQ_LEN - 1)
        t_end = min(hour_hi, T - 1 if need_next else T)
        for t in range(t_start, t_end):
            X.append(windows_ending_at(grid, t))
            if need_next:
                Y.append(grid[t + 1])
    if not X:
        return np.empty((0, SEQ_LEN, len(FEATURES)), np.float32), np.empty((0, len(FEATURES)), np.float32)
    return np.stack(X), (np.stack(Y) if need_next else np.empty((0,), np.float32))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lstm(Xf: np.ndarray, Yf: np.ndarray, Xv: np.ndarray, Yv: np.ndarray,
               epochs: int, seed: int) -> Tuple[BehaviouralLSTM, List[float]]:
    """Unsupervised: predict the next hour's feature vector. Labels never used."""
    torch.manual_seed(seed)
    model = BehaviouralLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    Xf_t, Yf_t = torch.from_numpy(Xf), torch.from_numpy(Yf)
    Xv_t, Yv_t = torch.from_numpy(Xv), torch.from_numpy(Yv)
    n, bs = len(Xf_t), 256
    best, best_state, history = float("inf"), None, []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xf_t[idx]), Yf_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(Xv_t), Yv_t))
        history.append(vl)
        # Early stopping on unsupervised val reconstruction loss (no labels used).
        if vl < best - 1e-5:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"    epoch {ep + 1:>2}/{epochs}  val_mse={vl:.5f}"
              f"{'  *' if vl <= best else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def encode_all(model: BehaviouralLSTM, X: np.ndarray, bs: int = 1024) -> np.ndarray:
    out = []
    for i in range(0, len(X), bs):
        out.append(model.encode(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out) if out else np.empty((0, HIDDEN_DIM), np.float32)


@torch.no_grad()
def pred_error_all(model: BehaviouralLSTM, X: np.ndarray, Y: np.ndarray,
                   bs: int = 1024) -> np.ndarray:
    """Per-window next-step squared prediction error.

    The other concrete scoring rule on the table. Reported as a diagnostic so the
    choice of memory-bank cosine is evidenced rather than asserted. Never fused.
    """
    out = []
    for i in range(0, len(X), bs):
        xb, yb = torch.from_numpy(X[i:i + bs]), torch.from_numpy(Y[i:i + bs])
        out.append(((model(xb) - yb) ** 2).mean(dim=1).numpy())
    return np.concatenate(out) if out else np.empty((0,), np.float32)


def l2norm(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9, None)


def memory_bank_distance(h: np.ndarray, bank: np.ndarray, k: int = TOPK) -> np.ndarray:
    """a_LSTM_raw = 1 - mean of the top-k cosine similarities against the bank.

    The paper says "compared against the user's historical memory bank (cosine
    similarity)". Plain max-similarity (k=1) is brittle: with ~150 stored
    encodings a mildly anomalous window can almost always find one accidental
    near-neighbour. Averaging the k nearest is the same statistic, denoised.
    """
    if len(bank) == 0:
        return np.zeros(len(h))
    cos = h @ bank.T
    kk = min(k, cos.shape[1])
    topk = np.partition(cos, -kk, axis=1)[:, -kk:]
    return 1.0 - topk.mean(axis=1)


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC; ties handled by average rank."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()[:len(pos)]
    return float((ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Tuning helpers
# ---------------------------------------------------------------------------
def tune_ewma_tau(dev: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    best = (LEGACY_EWMA_TAU, -1.0)
    for tau in np.arange(0.25, 12.01, 0.25):
        f1 = binary_metrics(y, dev > tau)["f1"]
        if f1 > best[1]:
            best = (float(tau), f1)
    return best


def tune_alpha_tau_constrained(a_lstm: np.ndarray, a_if: np.ndarray, y: np.ndarray,
                               tc: np.ndarray, target: str, fpr_budget: float):
    """VAL-only search for a *recall-oriented* operating point.

    Maximise recall on one hard threat class subject to a false-positive budget.
    The F1-optimal point abandons slow insiders (they are a minority of positives
    and expensive to catch), so a deployment that actually cares about them needs
    a different alpha/tau. Selected on VAL, reported on TEST, like everything else.
    """
    sel = tc == target
    best = {"alpha": 1.0, "tau": 0.5, "recall": -1.0, "fpr": 1.0}
    for alpha in np.round(np.arange(0.0, 1.001, 0.05), 3):
        fused = alpha * a_lstm + (1 - alpha) * a_if
        for tau in np.round(np.arange(0.02, 0.99, 0.01), 3):
            pred = fused >= tau
            fpr = float(pred[~y].mean()) if (~y).any() else 1.0
            if fpr > fpr_budget:
                continue
            rec = float(pred[sel].mean()) if sel.any() else 0.0
            if rec > best["recall"] or (rec == best["recall"] and fpr < best["fpr"]):
                best = {"alpha": float(alpha), "tau": float(tau), "recall": rec, "fpr": fpr}
    return best


def tune_alpha_tau(a_lstm: np.ndarray, a_if: np.ndarray, y: np.ndarray):
    best = {"alpha": 0.5, "tau": HYBRID_FLAG_THRESHOLD, "f1": -1.0}
    grid = []
    for alpha in np.round(np.arange(0.0, 1.001, 0.05), 3):
        fused = alpha * a_lstm + (1 - alpha) * a_if
        row = {"alpha": float(alpha), "best_f1": -1.0, "best_tau": 0.5}
        for tau in np.round(np.arange(0.02, 0.99, 0.01), 3):
            f1 = binary_metrics(y, fused >= tau)["f1"]
            if f1 > row["best_f1"]:
                row["best_f1"], row["best_tau"] = f1, float(tau)
            if f1 > best["f1"]:
                best = {"alpha": float(alpha), "tau": float(tau), "f1": f1}
        grid.append(row)
    return best, grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latency-samples", type=int, default=500)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # -- 1. Load -----------------------------------------------------------
    df = pd.read_csv(SESSIONS_RAW, parse_dates=["timestamp"])
    df["threat_class"] = df["threat_class"].where(df["threat_class"].notna(), None)
    total_hours = int(df.hour_index.max()) + 1
    train_end = int(np.floor(total_hours * 0.70))
    fit_end = int(np.floor(train_end * 0.75))
    calib_lo = fit_end + SEQ_LEN                      # 24 h buffer, no overlap
    print(f"[1/8] {len(df):,} grid rows | hours 0..{total_hours - 1} | "
          f"FIT<{fit_end}  CALIB[{calib_lo},{train_end})  VAL/TEST from {train_end}")

    # -- 2. Two reference frames, both fitted on the FIT period only --------
    #   global frame  -> Isolation Forest ("global organizational distribution")
    #   personal frame-> LSTM             ("the user's historical memory bank")
    fit_mask = df.hour_index < fit_end
    scaler = StandardScaler().fit(df.loc[fit_mask & df.active, list(FEATURES)].to_numpy())
    scaled = df.copy()
    scaled[list(FEATURES)] = scaler.transform(df[list(FEATURES)].to_numpy())
    grids, user_stats = build_personal_grids(df, fit_end)
    print(f"[2/8] global scaler fitted on {int((fit_mask & df.active).sum()):,} FIT sessions; "
          f"per-user frames for {len(grids)} users")

    # -- 3. LSTM -----------------------------------------------------------
    Xf, Yf = collect_windows(grids, 0, fit_end, need_next=True)
    Xv, Yv = collect_windows(grids, train_end, int(total_hours * 0.85), need_next=True)
    print(f"[3/8] training LSTM(hidden={HIDDEN_DIM}) on {len(Xf):,} windows "
          f"(val {len(Xv):,}) - unsupervised next-step prediction, labels never used")
    model, history = train_lstm(Xf, Yf, Xv, Yv, args.epochs, args.seed)

    # -- 4. Encode every scoreable session --------------------------------
    # A session is scoreable if it is active and has a full 24 h of history.
    df["scoreable"] = df.active & (df.hour_index >= SEQ_LEN - 1)
    n_dropped = int((df.active & ~df.scoreable).sum())
    idx_by_user: Dict[str, np.ndarray] = {}
    win_list, meta = [], []
    for uid, grid in grids.items():
        sub = df[(df.user_id == uid) & df.scoreable]
        hours = sub.hour_index.to_numpy()
        idx_by_user[uid] = sub.index.to_numpy()
        for t in hours:
            win_list.append(windows_ending_at(grid, int(t)))
        meta.extend((uid, int(t)) for t in hours)
    W = np.stack(win_list)
    H = l2norm(encode_all(model, W))

    row_index = np.concatenate([idx_by_user[u] for u in grids])
    meta_user = np.array([m[0] for m in meta])
    meta_hour = np.array([m[1] for m in meta])

    # Diagnostic scorer: predict hour t from hours [t-24, t-1] and measure error.
    pe_ok = meta_hour >= SEQ_LEN
    Wp = np.stack([grids[u][t - SEQ_LEN:t] for u, t in zip(meta_user[pe_ok], meta_hour[pe_ok])])
    Yp = np.stack([grids[u][t] for u, t in zip(meta_user[pe_ok], meta_hour[pe_ok])])
    pred_err = np.full(len(H), np.nan)
    pred_err[pe_ok] = pred_error_all(model, Wp, Yp)

    print(f"[4/8] encoded {len(H):,} scoreable sessions "
          f"({n_dropped} active sessions dropped: <24 h of history)")

    # -- 5. Memory banks (FIT period only) + a_LSTM_raw ---------------------
    a_lstm_raw = np.zeros(len(H))
    banks: Dict[str, np.ndarray] = {}
    for uid in grids:
        sel = meta_user == uid
        in_fit = sel & (meta_hour < fit_end)
        bank = H[in_fit]
        if len(bank) > MEMORY_BANK_CAP:
            bank = bank[rng.choice(len(bank), MEMORY_BANK_CAP, replace=False)]
        banks[uid] = bank
        a_lstm_raw[sel] = memory_bank_distance(H[sel], bank, TOPK)
    print(f"[5/8] memory banks built (mean {np.mean([len(b) for b in banks.values()]):.0f} "
          f"encodings/user, top-{TOPK} cosine)")

    # -- 6. Isolation Forest (FIT period, global distribution) --------------
    Xfit_rows = scaled.loc[fit_mask & scaled.active, list(FEATURES)].to_numpy()
    iforest = IsolationForest(n_estimators=200, contamination="auto",
                              random_state=args.seed, n_jobs=-1).fit(Xfit_rows)
    Xall = scaled.loc[row_index, list(FEATURES)].to_numpy()
    a_if_raw = -iforest.score_samples(Xall)
    print(f"[6/8] IsolationForest fitted on {len(Xfit_rows):,} FIT sessions "
          f"(contains unlabelled contamination, as in deployment)")

    # -- Calibrate on CALIB sub-period -------------------------------------
    calib = (meta_hour >= calib_lo) & (meta_hour < train_end)
    cal_lstm = RobustCalibrator().fit(a_lstm_raw[calib])
    cal_if = RobustCalibrator().fit(a_if_raw[calib])
    a_lstm = cal_lstm.transform(a_lstm_raw)
    a_if = cal_if.transform(a_if_raw)

    S = df.loc[row_index]
    y = S.is_true_threat.to_numpy(dtype=bool)
    split = S.split.to_numpy()

    # -- 7. EWMA baseline over the full chronological stream ----------------
    df["ewma_previous"] = np.nan
    df["ewma_new"] = np.nan
    df["ewma_deviation"] = np.nan
    for uid, g in df[df.active].groupby("user_id", sort=False):
        g = g.sort_values("hour_index")
        p, n_, d = ewma_stream(g.activity_rate_per_hr.to_numpy(), EWMA_LAMBDA)
        df.loc[g.index, "ewma_previous"] = p
        df.loc[g.index, "ewma_new"] = n_
        df.loc[g.index, "ewma_deviation"] = d
    dev = df.loc[row_index, "ewma_deviation"].to_numpy()

    v = split == "val"
    tau_ewma, ewma_val_f1 = tune_ewma_tau(dev[v], y[v])
    best, alpha_grid = tune_alpha_tau(a_lstm[v], a_if[v], y[v])
    alpha, tau_h = best["alpha"], best["tau"]
    print(f"[7/8] tuned on VAL -> tau_ewma={tau_ewma:.2f} (F1={ewma_val_f1:.3f}) | "
          f"alpha={alpha:.2f} tau_hybrid={tau_h:.2f} (F1={best['f1']:.3f})")

    tc_all = S.threat_class.to_numpy()
    insider_op = tune_alpha_tau_constrained(a_lstm[v], a_if[v], y[v], tc_all[v],
                                            "malicious_insider", fpr_budget=0.05)
    print(f"      insider-oriented op (val FPR<=5%): alpha={insider_op['alpha']:.2f} "
          f"tau={insider_op['tau']:.2f} val_malicious_recall={insider_op['recall']:.3f}")

    a_hybrid = alpha * a_lstm + (1 - alpha) * a_if
    ewma_flag = dev > tau_ewma
    hybrid_flag = a_hybrid >= tau_h

    sens = S.pii_sensitivity_score.to_numpy(dtype=float)
    sbrs_vals = np.array([sbrs(s, a, SBRS_BETA) for s, a in zip(sens, a_hybrid)])
    bands = [band(v_) for v_ in sbrs_vals]
    sbrs_cat = np.array([b[0] for b in bands])
    hybrid_action = np.array([b[1] for b in bands])
    base_action = np.where(ewma_flag, "BLOCK", "PERMIT")

    # -- 8. TEST-ONLY evaluation -------------------------------------------
    te = split == "test"
    yt = y[te]
    print(f"[8/8] TEST: {int(te.sum()):,} sessions, {int(yt.sum())} positives")

    m_ewma = binary_metrics(yt, ewma_flag[te])
    m_hyb = binary_metrics(yt, hybrid_flag[te])
    m_lstm_only = binary_metrics(yt, a_lstm[te] >= tau_h)
    m_if_only = binary_metrics(yt, a_if[te] >= tau_h)
    m_ewma_legacy = binary_metrics(yt, dev[te] > LEGACY_EWMA_TAU)

    # Enforcement-band F1 (paper's SAFE/SENSITIVE/HIGH-RISK -> PERMIT/ALERT/BLOCK)
    enf_alert_plus = binary_metrics(yt, hybrid_action[te] != "PERMIT")
    enf_block_only = binary_metrics(yt, hybrid_action[te] == "BLOCK")
    enf_base = binary_metrics(yt, base_action[te] == "BLOCK")
    doc_bands = np.array([band(v_, PAPER_DOC_BANDS)[1] for v_ in sbrs_vals])
    enf_docbands = binary_metrics(yt, doc_bands[te] != "PERMIT")

    # Per-class recall (window level + episode level), with Wilson intervals
    # Second operating point: same models, recall-oriented alpha/tau (val-selected).
    a_hyb_ins = insider_op["alpha"] * a_lstm + (1 - insider_op["alpha"]) * a_if
    ins_flag = a_hyb_ins >= insider_op["tau"]
    m_insider_op = binary_metrics(yt, ins_flag[te])

    tc = S.threat_class.to_numpy()
    eid = S.episode_id.to_numpy()
    neg_te = te & ~y                       # all test negatives (incl. benign bursts)
    m_insider_op["malicious_insider_recall"] = float(
        ins_flag[te & (tc == "malicious_insider")].mean())
    m_insider_op["benign_burst_flag_rate"] = float(
        ins_flag[te & S.is_benign_burst.to_numpy(dtype=bool)].mean())
    m_insider_op.update({"alpha": insider_op["alpha"], "tau": insider_op["tau"]})
    per_class = {}
    for cls in THREAT_CLASSES:
        sel = te & (tc == cls)
        n_w = int(sel.sum())
        if n_w == 0:
            continue
        k_h = int(hybrid_flag[sel].sum())
        k_e = int(ewma_flag[sel].sum())
        eps = pd.unique(eid[sel])
        det = sum(1 for e in eps if hybrid_flag[te & (eid == e)].any())
        det_e = sum(1 for e in eps if ewma_flag[te & (eid == e)].any())
        per_class[cls] = {
            "windows": n_w,
            "hybrid_recall": k_h / n_w, "hybrid_recall_ci": wilson_ci(k_h, n_w),
            "ewma_recall": k_e / n_w, "ewma_recall_ci": wilson_ci(k_e, n_w),
            "episodes": len(eps),
            "hybrid_episode_recall": det / len(eps),
            "ewma_episode_recall": det_e / len(eps),
            "mean_a_hybrid": float(a_hybrid[sel].mean()),
            "mean_a_lstm": float(a_lstm[sel].mean()),
            "mean_a_if": float(a_if[sel].mean()),
            # Threshold-free discrimination vs all test negatives.
            "auc_lstm": roc_auc(a_lstm[sel], a_lstm[neg_te]),
            "auc_if": roc_auc(a_if[sel], a_if[neg_te]),
            "auc_hybrid": roc_auc(a_hybrid[sel], a_hybrid[neg_te]),
        }
    auc_overall = {
        "lstm": roc_auc(a_lstm[te & y], a_lstm[neg_te]),
        "if": roc_auc(a_if[te & y], a_if[neg_te]),
        "hybrid": roc_auc(a_hybrid[te & y], a_hybrid[neg_te]),
    }
    # Did we pick the right LSTM scoring rule? Compare the paper's memory-bank
    # cosine against the other candidate, next-step prediction error.
    # Diagnostic only: prediction error is never fused into A_hybrid.
    ok = ~np.isnan(pred_err)
    auc_prederr = {
        "overall": roc_auc(pred_err[te & y & ok], pred_err[neg_te & ok]),
        **{cls: roc_auc(pred_err[te & (tc == cls) & ok], pred_err[neg_te & ok])
           for cls in THREAT_CLASSES if (te & (tc == cls) & ok).any()},
    }

    # Benign-burst trap, isolated
    bb = S.is_benign_burst.to_numpy(dtype=bool)
    bb_te = te & bb
    neg_plain = te & ~y & ~bb
    burst = {
        "n_test_benign_burst_sessions": int(bb_te.sum()),
        "ewma_flag_rate": float(ewma_flag[bb_te].mean()),
        "hybrid_flag_rate": float(hybrid_flag[bb_te].mean()),
        "ewma_legacy_tau_flag_rate": float((dev[bb_te] > LEGACY_EWMA_TAU).mean()),
        "hybrid_action_counts": {k: int(v) for k, v in
                                 pd.Series(hybrid_action[bb_te]).value_counts().items()},
        "mean_a_lstm": float(a_lstm[bb_te].mean()),
        "mean_a_if": float(a_if[bb_te].mean()),
        "mean_a_hybrid": float(a_hybrid[bb_te].mean()),
        "mean_sensitivity": float(sens[bb_te].mean()),
        "mean_sbrs": float(sbrs_vals[bb_te].mean()),
        "fpr_on_ordinary_negatives_ewma": float(ewma_flag[neg_plain].mean()),
        "fpr_on_ordinary_negatives_hybrid": float(hybrid_flag[neg_plain].mean()),
        "mean_activity_rate_ratio": float(df.loc[row_index, "activity_rate"].to_numpy()[bb_te].mean()),
    }

    # -- Latency: time the real per-call scoring path ----------------------
    torch.set_grad_enabled(False)
    test_pos = np.flatnonzero(te)
    sample = rng.choice(test_pos, min(args.latency_samples, len(test_pos)), replace=False)
    for i in sample[:20]:                                   # warm-up
        u, t = meta_user[i], meta_hour[i]
        w = torch.from_numpy(windows_ending_at(grids[u], int(t))[None])
        model.encode(w); iforest.score_samples(Xall[i:i + 1])

    lat = []
    for i in sample:
        u, t = meta_user[i], meta_hour[i]
        t0 = time.perf_counter()
        w = torch.from_numpy(windows_ending_at(grids[u], int(t))[None])
        h = l2norm(model.encode(w).numpy())
        raw_l = float(memory_bank_distance(h, banks[u], TOPK)[0])
        raw_i = -float(iforest.score_samples(Xall[i:i + 1])[0])
        al = float(cal_lstm.transform(np.array([raw_l]))[0])
        ai = float(cal_if.transform(np.array([raw_i]))[0])
        ah = alpha * al + (1 - alpha) * ai
        band(sbrs(float(sens[i]), ah, SBRS_BETA))
        lat.append((time.perf_counter() - t0) * 1000.0)
    torch.set_grad_enabled(True)
    lat = np.array(lat)
    latency = {
        "n_calls": int(len(lat)), "mean_ms": float(lat.mean()),
        "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)), "max_ms": float(lat.max()),
        "note": "anomaly scoring only (LSTM + memory bank + IF + fusion + SBRS band). "
                "Excludes the PII/OCR pipeline, so NOT comparable to the paper's "
                "279 ms end-to-end figure.",
        "device": "cpu", "torch_threads": int(torch.get_num_threads()),
    }

    # -- Emit ---------------------------------------------------------------
    out = pd.DataFrame({
        "event_id": S.event_id.to_numpy(),
        "timestamp": S.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "user_id": S.user_id.to_numpy(), "user_name": S.user_name.to_numpy(),
        "department": S.department.to_numpy(), "platform": S.platform.to_numpy(),
        "file_accessed": S.file_accessed.to_numpy(),
        "time_gap_sec": np.rint(S.time_gap_sec.to_numpy()).astype(int),
        "activity_rate_per_hr": np.round(S.activity_rate_per_hr.to_numpy(), 3),
        "ewma_previous": np.round(df.loc[row_index, "ewma_previous"].to_numpy(), 3),
        "ewma_new": np.round(df.loc[row_index, "ewma_new"].to_numpy(), 4),
        "ewma_deviation": np.round(dev, 4),
        "ewma_anomaly_flagged": ewma_flag,
        "lstm_score": np.round(a_lstm, 4),
        "isolation_forest_score": np.round(a_if, 4),
        "hybrid_anomaly_score": np.round(a_hybrid, 4),
        "hybrid_anomaly_flagged": hybrid_flag,
        "pii_sensitivity_score": sens.astype(int),
        "sbrs_value": np.round(sbrs_vals, 4),
        "sbrs_category": sbrs_cat, "base_action": base_action,
        "hybrid_action": hybrid_action, "is_true_threat": y,
        "ewma_correct": ewma_flag == y, "hybrid_correct": hybrid_flag == y,
    })[list(CSV_COLUMNS)]

    extra = out.copy()
    extra["split"] = split
    extra["threat_class"] = tc
    extra["is_benign_burst"] = bb
    extra["episode_id"] = eid
    extra.to_csv(SCORED_ALL, index=False)
    out[te].to_csv(V2_CSV, index=False)   # held-out predictions only

    with open(ARTIFACTS / "lstm.pt", "wb") as f:
        torch.save({"state_dict": model.state_dict(), "hidden": HIDDEN_DIM,
                    "n_features": len(FEATURES), "seq_len": SEQ_LEN}, f)
    with open(ARTIFACTS / "engine.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "iforest": iforest, "cal_lstm": cal_lstm,
                     "cal_if": cal_if, "banks": banks, "user_stats": user_stats,
                     "alpha": alpha, "tau_hybrid": tau_h,
                     "tau_ewma": tau_ewma, "features": list(FEATURES),
                     "topk": TOPK, "z_clip": Z_CLIP, "sd_floor": SD_FLOOR}, f)

    metrics = {
        "dataset": {
            "users": int(df.user_id.nunique()), "grid_rows": int(len(df)),
            "active_sessions": int(df.active.sum()),
            "scoreable_sessions": int(len(row_index)),
            "dropped_no_history": n_dropped,
            "positives_total": int(df[df.active].is_true_threat.sum()),
            "positive_rate": float(df[df.active].is_true_threat.mean()),
            "per_split": {s: {"sessions": int((split == s).sum()),
                              "positives": int(y[split == s].sum()),
                              "negatives": int((~y[split == s]).sum()),
                              "benign_bursts": int(bb[split == s].sum())}
                          for s in ("train", "val", "test")},
        },
        "split_strategy": "chronological (by time): train=first 70% of hours, "
                          "val=next 15%, test=last 15%. TRAIN internally split "
                          "into FIT (75%) / CALIB (25%) with a 24 h buffer.",
        "tuning": {"alpha": alpha, "tau_hybrid": tau_h, "val_f1": best["f1"],
                   "tau_ewma": tau_ewma, "ewma_val_f1": ewma_val_f1,
                   "legacy_tau_ewma": LEGACY_EWMA_TAU,
                   "alpha_grid": alpha_grid},
        "lstm_val_mse_history": history,
        "test_flag_level": {"ewma_tuned": m_ewma, "ewma_legacy_tau": m_ewma_legacy,
                            "hybrid": m_hyb, "lstm_only": m_lstm_only,
                            "if_only": m_if_only},
        "test_insider_oriented_operating_point": m_insider_op,
        "test_enforcement_bands": {
            "hybrid_alert_or_block": enf_alert_plus,
            "hybrid_block_only": enf_block_only,
            "ewma_base_action_block": enf_base,
            "paper_doc_bands_alert_or_block": enf_docbands,
        },
        "test_auc_overall": auc_overall,
        "diagnostic_auc_next_step_prediction_error": auc_prederr,
        "per_class_recall": per_class,
        "benign_burst": burst,
        "latency": latency,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    # -- Console summary ----------------------------------------------------
    print("\n" + "=" * 72)
    print(f"TEST  n={int(te.sum()):,}  pos={int(yt.sum())}  neg={int((~yt).sum())}")
    print("-" * 72)
    print(f"{'detector':<26}{'FPR':>9}{'recall':>9}{'prec':>9}{'F1':>9}")
    for name, m in (("EWMA (tuned tau)", m_ewma), ("EWMA (legacy tau=2.0)", m_ewma_legacy),
                    ("LSTM only", m_lstm_only), ("IsolationForest only", m_if_only),
                    ("HYBRID", m_hyb)):
        print(f"{name:<26}{m['fpr']:>9.4f}{m['recall']:>9.4f}"
              f"{m['precision']:>9.4f}{m['f1']:>9.4f}")
    print("-" * 72)
    print("Enforcement bands (SBRS -> action):")
    for name, m in (("hybrid ALERT+BLOCK", enf_alert_plus),
                    ("hybrid BLOCK only", enf_block_only),
                    ("EWMA base_action BLOCK", enf_base)):
        print(f"  {name:<24} F1={m['f1']:.4f}  FPR={m['fpr']:.4f}  rec={m['recall']:.4f}")
    print("-" * 72)
    print(f"Insider-oriented op (val-selected, alpha={m_insider_op['alpha']:.2f} "
          f"tau={m_insider_op['tau']:.2f}):")
    print(f"  TEST FPR={m_insider_op['fpr']:.4f}  F1={m_insider_op['f1']:.4f}  "
          f"malicious_recall={m_insider_op['malicious_insider_recall']:.3f}  "
          f"benign_burst_flag_rate={m_insider_op['benign_burst_flag_rate']:.3f}")
    print("-" * 72)
    print(f"Test AUC (vs all negatives): lstm={auc_overall['lstm']:.3f}  "
          f"if={auc_overall['if']:.3f}  hybrid={auc_overall['hybrid']:.3f}")
    print("Per-class recall (hybrid, window level):")
    for cls, d in per_class.items():
        lo, hi = d["hybrid_recall_ci"]
        print(f"  {cls:<24} {d['hybrid_recall']:.3f} [{lo:.2f},{hi:.2f}] "
              f"n={d['windows']:<4} ep={d['hybrid_episode_recall']:.3f} "
              f"(EWMA {d['ewma_recall']:.3f})  AUC L/I/H="
              f"{d['auc_lstm']:.2f}/{d['auc_if']:.2f}/{d['auc_hybrid']:.2f}")
    print("-" * 72)
    print("BENIGN BURST trap:")
    print(f"  sessions={burst['n_test_benign_burst_sessions']}  "
          f"mean activity ratio={burst['mean_activity_rate_ratio']:.1f}x baseline")
    print(f"  EWMA flags   {burst['ewma_flag_rate'] * 100:.1f}%   "
          f"(legacy tau: {burst['ewma_legacy_tau_flag_rate'] * 100:.1f}%)")
    print(f"  HYBRID flags {burst['hybrid_flag_rate'] * 100:.1f}%   "
          f"actions={burst['hybrid_action_counts']}")
    print(f"  ordinary-negative FPR: EWMA {burst['fpr_on_ordinary_negatives_ewma'] * 100:.1f}%"
          f"  hybrid {burst['fpr_on_ordinary_negatives_hybrid'] * 100:.1f}%")
    print("-" * 72)
    print(f"Latency/call: mean={latency['mean_ms']:.2f} ms  p50={latency['p50_ms']:.2f}  "
          f"p95={latency['p95_ms']:.2f}  p99={latency['p99_ms']:.2f}  (n={latency['n_calls']})")
    print("=" * 72)
    print(f"\n  -> {V2_CSV}\n  -> {SCORED_ALL}\n  -> {METRICS_JSON}\n  -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
