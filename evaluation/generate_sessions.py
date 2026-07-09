"""
generate_sessions.py
====================
Synthetic multi-user SaaS telemetry for a *real* evaluation of the hybrid
anomaly engine (paper Sec. III.C).

Model
-----
`N_USERS` synthetic employees, each with a persistent profile: department,
platform, home city, personal file-type mix, personal endpoint mix, personal
collaboration level, and a personal activity-rate distribution. Behaviour is
simulated on a **full hourly grid** across `DAYS` days -- inactive hours are
retained (activity 0) because the LSTM consumes a contiguous 24-hour window.
A *session* is any hour in which the user made >= 1 API call.

Each active hour is encoded as the paper's six-dimensional vector:

    [activity_rate, file_type_category, geo_index,
     endpoint_operation, permission_scope_delta, collaboration_density]

`activity_rate` is calls-in-window normalised against a **causal** trailing
7-day rolling mean of that user's own activity (paper: "normalized against
rolling mean"), so it never sees the future.

Splits
------
Split is **chronological (by time), assigned here, once**: the first 70% of the
hour grid is TRAIN, the next 15% VAL, the last 15% TEST. This is the deployment-
realistic choice -- you fit on the past and score the future -- and, unlike a
by-user split, it lets every user carry a personal LSTM memory bank built from
their own training-period history (a by-user split leaves test users with no
history at all, which the paper's architecture requires). Attack episodes are
injected wholly inside a single split, never straddling a boundary.

Injected ground truth
---------------------
Four adversary classes from `Threat Model and Goals.md`, plus one labelled
*negative* trap:

  a) malicious_insider      slow multi-hour drift off the personal baseline,
                            every per-window step kept inside plausible bounds
  b) compromised_account    short burst from an unfamiliar geo + risky endpoints
  c) negligent_insider      oversharing / permission-change spike, no precursor
  d) overscoped_thirdparty  sustained anomalous permission_scope_delta
  e) benign_burst           NEGATIVE. Team bulk-downloading low-sensitivity
                            templates before a deadline. Huge activity spike,
                            every other dimension normal. The canonical
                            false-positive trap for rate-based detectors
                            (paper, Introduction).

Outputs (all under evaluation/data/)
------------------------------------
    sessions_raw.csv        every hour of the grid, features + labels + split
    user_profiles.csv       the 50 generated profiles
    generation_config.json  seed, parameters, and exact realised counts
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from faker import Faker

from .common import (DATA_DIR, DEPARTMENTS, ENDPOINT_ORDINAL, FILE_TYPE_CLASSES,
                     FILE_TYPE_ORDINAL, GEN_CONFIG, PLATFORMS,
                     SENSITIVITY_BY_FILE_CLASS, SESSIONS_RAW, USER_PROFILES)

# ---------------------------------------------------------------------------
# Geography: geo_index = haversine(origin, home) / 5000 km, clipped to [0, 1]
# ---------------------------------------------------------------------------
HOME_CITIES = {
    "Karachi": (24.86, 67.01), "Lahore": (31.55, 74.34),
    "Islamabad": (33.68, 73.05), "Rawalpindi": (33.60, 73.04),
    "Faisalabad": (31.42, 73.08), "Peshawar": (34.01, 71.58),
}
FOREIGN_CITIES = {
    "Dubai": (25.20, 55.27), "Moscow": (55.75, 37.62), "Lagos": (6.52, 3.38),
    "Singapore": (1.35, 103.82), "Amsterdam": (52.37, 4.90),
    "Kyiv": (50.45, 30.52), "Sao Paulo": (-23.55, -46.63),
}
GEO_NORM_KM = 5000.0

# Department -> prior over file-type classes (public, internal, confidential, restricted)
DEPT_FILE_PRIOR = {
    "Finance":     [0.05, 0.30, 0.45, 0.20],
    "Legal":       [0.03, 0.22, 0.35, 0.40],
    "HR":          [0.05, 0.30, 0.40, 0.25],
    "Engineering": [0.15, 0.62, 0.20, 0.03],
    "Sales":       [0.30, 0.52, 0.16, 0.02],
}
# Department -> prior over endpoint ops (read, write, share, delete, perm_change)
DEPT_OP_PRIOR = {
    "Finance":     [0.62, 0.26, 0.08, 0.03, 0.01],
    "Legal":       [0.70, 0.18, 0.09, 0.02, 0.01],
    "HR":          [0.60, 0.24, 0.12, 0.03, 0.01],
    "Engineering": [0.52, 0.34, 0.09, 0.04, 0.01],
    "Sales":       [0.55, 0.22, 0.19, 0.03, 0.01],
}

FILE_STEMS = {
    "public":       ["brochure", "press_release", "public_faq", "template", "logo_pack"],
    "internal":     ["sprint_notes", "roadmap", "team_sync", "onboarding_guide", "runbook"],
    "confidential": ["salary_band", "vendor_contract", "financials_q", "customer_list"],
    "restricted":   ["audit_report", "merger_memo", "board_minutes", "pentest_findings"],
}
FILE_EXTS = ["pdf", "docx", "xlsx", "png", "csv"]


def _haversine_km(a: tuple, b: tuple) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _hour_factor(hour: int, is_weekend: bool) -> float:
    """Diurnal + weekly activity envelope."""
    if is_weekend:
        return 0.10 if 10 <= hour <= 18 else 0.02
    if 9 <= hour <= 17:
        return 1.0
    if hour in (8, 18):
        return 0.55
    if hour in (7, 19, 20):
        return 0.18
    return 0.03


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
def build_profiles(n_users: int, rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    rows = []
    home_names = list(HOME_CITIES)
    for i in range(n_users):
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        home = home_names[int(rng.integers(len(home_names)))]
        rows.append({
            "user_id": f"USR{i + 1:03d}",
            "user_name": fake.name(),
            "department": dept,
            "platform": PLATFORMS[int(rng.integers(len(PLATFORMS)))],
            "home_city": home,
            # peak calls/hour during core work hours
            "peak_rate": float(rng.uniform(6.0, 26.0)),
            # multiplicative hour-to-hour noise (some users are far burstier)
            "rate_sigma": float(rng.uniform(0.25, 0.55)),
            # personal collaboration level (sharing-graph degree, normalised)
            "collab_mean": float(rng.uniform(0.12, 0.55)),
            "collab_sd": float(rng.uniform(0.03, 0.10)),
            # probability the user works from an unusual-but-domestic city
            "p_domestic_roam": float(rng.uniform(0.01, 0.05)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Normal behaviour on the hourly grid
# ---------------------------------------------------------------------------
def simulate_normal(profiles: pd.DataFrame, days: int, start: pd.Timestamp,
                    rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    hours = pd.date_range(start, periods=days * 24, freq="h")
    is_weekend = np.array([ts.weekday() >= 5 for ts in hours])
    records: List[Dict[str, Any]] = []

    for _, p in profiles.iterrows():
        home_xy = HOME_CITIES[p.home_city]
        file_prior = np.array(DEPT_FILE_PRIOR[p.department])
        op_prior = np.array(DEPT_OP_PRIOR[p.department])
        # Beta params matching the user's collaboration mean/sd
        cm, cs = p.collab_mean, p.collab_sd
        kappa = max(cm * (1 - cm) / (cs ** 2) - 1, 2.0)
        a_beta, b_beta = cm * kappa, (1 - cm) * kappa

        for h_i, ts in enumerate(hours):
            base = p.peak_rate * _hour_factor(ts.hour, bool(is_weekend[h_i]))
            mult = rng.lognormal(0.0, p.rate_sigma)
            calls = int(rng.poisson(max(base * mult, 1e-9)))

            if calls == 0:
                records.append({
                    "user_id": p.user_id, "timestamp": ts, "hour_index": h_i,
                    "calls": 0, "file_class": "public", "endpoint_class": "read",
                    "origin_city": p.home_city, "geo_index": 0.0,
                    "permission_scope_delta": 0.0, "collaboration_density": 0.0,
                    "active": False, "is_true_threat": False, "threat_class": None,
                    "is_benign_burst": False, "episode_id": None,
                })
                continue

            file_class = FILE_TYPE_CLASSES[int(rng.choice(4, p=file_prior))]
            endpoint_class = list(ENDPOINT_ORDINAL)[int(rng.choice(5, p=op_prior))]

            u = rng.random()
            if u < 0.006:  # rare foreign trip (benign)
                city = list(FOREIGN_CITIES)[int(rng.integers(len(FOREIGN_CITIES)))]
                origin_xy = FOREIGN_CITIES[city]
            elif u < 0.006 + p.p_domestic_roam:
                city = list(HOME_CITIES)[int(rng.integers(len(HOME_CITIES)))]
                origin_xy = HOME_CITIES[city]
            else:
                city, origin_xy = p.home_city, home_xy

            geo_index = min(_haversine_km(origin_xy, home_xy) / GEO_NORM_KM, 1.0)
            perm_delta = 0.0 if rng.random() < 0.93 else float(rng.beta(1.5, 8.0))
            collab = float(np.clip(rng.beta(a_beta, b_beta), 0.0, 1.0))

            records.append({
                "user_id": p.user_id, "timestamp": ts, "hour_index": h_i,
                "calls": calls, "file_class": file_class,
                "endpoint_class": endpoint_class, "origin_city": city,
                "geo_index": geo_index, "permission_scope_delta": perm_delta,
                "collaboration_density": collab, "active": True,
                "is_true_threat": False, "threat_class": None,
                "is_benign_burst": False, "episode_id": None,
            })

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["user_id", "hour_index"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split assignment (chronological)
# ---------------------------------------------------------------------------
def assign_splits(df: pd.DataFrame, days: int) -> pd.DataFrame:
    total_hours = days * 24
    train_end = int(total_hours * 0.70)
    val_end = int(total_hours * 0.85)
    split = np.where(df.hour_index < train_end, "train",
                     np.where(df.hour_index < val_end, "val", "test"))
    df["split"] = split
    return df


# ---------------------------------------------------------------------------
# Attack injection
# ---------------------------------------------------------------------------
def _active_positions(df: pd.DataFrame, user: str, split: str) -> np.ndarray:
    """Row positions of a user's active sessions inside one split, chronological."""
    m = (df.user_id.values == user) & (df.active.values) & (df.split.values == split)
    return np.flatnonzero(m)


def inject_episodes(df: pd.DataFrame, profiles: pd.DataFrame,
                    rng: np.random.Generator, plan: Dict[str, int]) -> pd.DataFrame:
    """Mutate consecutive active sessions in place. Episodes stay inside one split."""
    prof = profiles.set_index("user_id")
    users = list(profiles.user_id)
    # Allocate episodes across splits proportional to split duration (70/15/15).
    split_weights = {"train": 0.70, "val": 0.15, "test": 0.15}
    episode_counter = 0
    used: set[int] = set()   # row positions already claimed by some episode

    def claim(positions: np.ndarray) -> bool:
        if any(int(p) in used for p in positions):
            return False
        used.update(int(p) for p in positions)
        return True

    def pick_window(user: str, split: str, length: int) -> np.ndarray | None:
        pos = _active_positions(df, user, split)
        if len(pos) < length + 2:
            return None
        for _ in range(12):  # bounded retries to avoid overlapping episodes
            s = int(rng.integers(1, len(pos) - length))
            win = pos[s:s + length]
            if claim(win):
                return win
        return None

    def episodes_for(kind: str, n_total: int, len_lo: int, len_hi: int):
        nonlocal episode_counter
        for split, w in split_weights.items():
            n = int(round(n_total * w))
            placed = 0
            attempts = 0
            while placed < n and attempts < n * 40:
                attempts += 1
                user = users[int(rng.integers(len(users)))]
                length = int(rng.integers(len_lo, len_hi + 1))
                win = pick_window(user, split, length)
                if win is None:
                    continue
                episode_counter += 1
                eid = f"{kind[:4].upper()}-{episode_counter:04d}"
                yield kind, user, win, eid
                placed += 1

    # --- (a) malicious insider: slow drift, each per-window step small ---------
    for kind, user, win, eid in episodes_for("malicious_insider", plan["malicious_insider"], 6, 10):
        n = len(win)
        for k, pos in enumerate(win):
            frac = (k + 1) / n                      # 0 -> 1 across the episode
            # activity creeps up gently; EWMA adapts, so per-window ratio stays low
            df.at[pos, "calls"] = int(df.at[pos, "calls"] * (1.0 + 0.55 * frac))
            # progressively reaches for more sensitive material
            if frac > 0.35:
                df.at[pos, "file_class"] = "confidential" if frac < 0.7 else "restricted"
            # works alone: collaboration degree collapses
            df.at[pos, "collaboration_density"] = float(
                max(df.at[pos, "collaboration_density"] * (1.0 - 0.75 * frac), 0.0))
            df.at[pos, "endpoint_class"] = "read"
            df.at[pos, "is_true_threat"] = True
            df.at[pos, "threat_class"] = "malicious_insider"
            df.at[pos, "episode_id"] = eid

    # --- (b) compromised account: burst from unfamiliar geo/endpoint ----------
    for kind, user, win, eid in episodes_for("compromised_account", plan["compromised_account"], 1, 3):
        home_xy = HOME_CITIES[prof.at[user, "home_city"]]
        city = list(FOREIGN_CITIES)[int(rng.integers(len(FOREIGN_CITIES)))]
        gi = min(_haversine_km(FOREIGN_CITIES[city], home_xy) / GEO_NORM_KM, 1.0)
        for pos in win:
            df.at[pos, "calls"] = int(max(df.at[pos, "calls"], 4) * rng.uniform(3.5, 7.0))
            df.at[pos, "origin_city"] = city
            df.at[pos, "geo_index"] = gi
            df.at[pos, "file_class"] = "restricted" if rng.random() < 0.6 else "confidential"
            df.at[pos, "endpoint_class"] = "share" if rng.random() < 0.5 else "delete"
            df.at[pos, "permission_scope_delta"] = float(rng.uniform(0.25, 0.6))
            df.at[pos, "is_true_threat"] = True
            df.at[pos, "threat_class"] = "compromised_account"
            df.at[pos, "episode_id"] = eid

    # --- (c) negligent insider: oversharing spike, no precursor ---------------
    for kind, user, win, eid in episodes_for("negligent_insider", plan["negligent_insider"], 1, 2):
        for pos in win:
            df.at[pos, "endpoint_class"] = "share" if rng.random() < 0.65 else "permission_change"
            df.at[pos, "permission_scope_delta"] = float(rng.uniform(0.45, 0.85))
            df.at[pos, "collaboration_density"] = float(np.clip(rng.uniform(0.75, 0.98), 0, 1))
            df.at[pos, "file_class"] = "confidential" if rng.random() < 0.7 else "restricted"
            # activity rate stays ordinary -> invisible to a rate-only detector
            df.at[pos, "is_true_threat"] = True
            df.at[pos, "threat_class"] = "negligent_insider"
            df.at[pos, "episode_id"] = eid

    # --- (d) over-scoped third-party app: sustained permission_scope_delta ----
    for kind, user, win, eid in episodes_for("overscoped_thirdparty", plan["overscoped_thirdparty"], 2, 4):
        scope = float(rng.uniform(0.82, 1.0))
        for pos in win:
            df.at[pos, "permission_scope_delta"] = scope
            df.at[pos, "endpoint_class"] = "permission_change"
            df.at[pos, "calls"] = int(max(df.at[pos, "calls"], 2) * rng.uniform(1.1, 1.8))
            df.at[pos, "file_class"] = "confidential" if rng.random() < 0.5 else "internal"
            df.at[pos, "is_true_threat"] = True
            df.at[pos, "threat_class"] = "overscoped_thirdparty"
            df.at[pos, "episode_id"] = eid

    # --- (e) BENIGN burst: the false-positive trap. Label stays negative. -----
    for kind, user, win, eid in episodes_for("benign_burst", plan["benign_burst"], 1, 3):
        for pos in win:
            # massive rate spike ...
            df.at[pos, "calls"] = int(max(df.at[pos, "calls"], 5) * rng.uniform(5.0, 11.0))
            # ... but everything else is textbook-normal for this user
            df.at[pos, "file_class"] = "public" if rng.random() < 0.7 else "internal"
            df.at[pos, "endpoint_class"] = "read"
            df.at[pos, "permission_scope_delta"] = 0.0
            df.at[pos, "geo_index"] = 0.0
            df.at[pos, "origin_city"] = prof.at[user, "home_city"]
            df.at[pos, "collaboration_density"] = float(np.clip(
                prof.at[user, "collab_mean"] * rng.uniform(1.0, 1.4), 0, 1))
            df.at[pos, "is_true_threat"] = False       # <-- ground truth NEGATIVE
            df.at[pos, "threat_class"] = None
            df.at[pos, "is_benign_burst"] = True
            df.at[pos, "episode_id"] = eid

    return df


# ---------------------------------------------------------------------------
# Derived features (computed AFTER injection, causally)
# ---------------------------------------------------------------------------
def derive_features(df: pd.DataFrame, profiles: pd.DataFrame,
                    rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    df = df.sort_values(["user_id", "hour_index"]).reset_index(drop=True)

    # activity_rate = calls / trailing 7-day (168 h) causal mean of own activity.
    parts = []
    for _, g in df.groupby("user_id", sort=False):
        calls = g["calls"].astype(float)
        roll = calls.shift(1).rolling(168, min_periods=6).mean()
        roll = roll.fillna(calls.shift(1).expanding(min_periods=1).mean()).fillna(1.0)
        g = g.copy()
        g["activity_rate"] = np.clip(calls / roll.clip(lower=0.25), 0.0, 10.0)
        g.loc[~g.active, "activity_rate"] = 0.0
        parts.append(g)
    df = pd.concat(parts).sort_values(["user_id", "hour_index"]).reset_index(drop=True)

    df["activity_rate_per_hr"] = df["calls"].astype(float)
    df["file_type_category"] = df["file_class"].map(FILE_TYPE_ORDINAL).astype(float)
    df["endpoint_operation"] = df["endpoint_class"].map(ENDPOINT_ORDINAL).astype(float)
    df.loc[~df.active, ["file_type_category", "endpoint_operation",
                        "geo_index", "permission_scope_delta",
                        "collaboration_density"]] = 0.0

    # Payload sensitivity S (what pii_pipeline would return) -- file class + noise.
    base_s = df["file_class"].map(SENSITIVITY_BY_FILE_CLASS).astype(float)
    noise = rng.normal(0.0, 7.0, len(df))
    df["pii_sensitivity_score"] = np.clip(np.rint(base_s + noise), 0, 100).astype(int)

    # time_gap_sec: seconds since this user's previous *active* session.
    df["time_gap_sec"] = 0.0
    for uid, g in df[df.active].groupby("user_id", sort=False):
        gaps = g["timestamp"].diff().dt.total_seconds()
        med = gaps.median()
        df.loc[g.index, "time_gap_sec"] = gaps.fillna(med if pd.notna(med) else 3600.0)

    prof = profiles.set_index("user_id")
    df["user_name"] = df.user_id.map(prof.user_name)
    df["department"] = df.user_id.map(prof.department)
    df["platform"] = df.user_id.map(prof.platform)

    stems = {c: FILE_STEMS[c] for c in FILE_TYPE_CLASSES}
    df["file_accessed"] = [
        f"{stems[c][int(rng.integers(len(stems[c])))]}_{int(rng.integers(1, 99)):02d}"
        f".{FILE_EXTS[int(rng.integers(len(FILE_EXTS)))]}"
        for c in df["file_class"]
    ]
    df["event_id"] = [f"ANOM-{i + 1:05d}" for i in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--n-users", type=int, default=50)
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=str, default="2026-04-01 00:00:00")
    # Episode counts. Tuned so the realised positive rate lands inside the 3-5%
    # band requested for the evaluation (a first pass at half these counts gave
    # 1.87%, and only 48 test positives -- too thin for per-class recall).
    # Episodes have different mean lengths, so counts are not proportional to
    # the window counts they produce.
    ap.add_argument("--n-malicious", type=int, default=38)
    ap.add_argument("--n-compromised", type=int, default=60)
    ap.add_argument("--n-negligent", type=int, default=72)
    ap.add_argument("--n-overscoped", type=int, default=34)
    ap.add_argument("--n-benign-burst", type=int, default=96)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    Faker.seed(args.seed)
    fake = Faker()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] profiles: {args.n_users} users")
    profiles = build_profiles(args.n_users, rng, fake)

    print(f"[2/5] simulating {args.days} days x 24 h of normal behaviour ...")
    df = simulate_normal(profiles, args.days, pd.Timestamp(args.start), rng, fake)
    df = assign_splits(df, args.days)

    print("[3/5] injecting attack episodes + benign-burst traps ...")
    plan = {
        "malicious_insider": args.n_malicious,
        "compromised_account": args.n_compromised,
        "negligent_insider": args.n_negligent,
        "overscoped_thirdparty": args.n_overscoped,
        "benign_burst": args.n_benign_burst,
    }
    df = inject_episodes(df, profiles, rng, plan)

    print("[4/5] deriving causal features ...")
    df = derive_features(df, profiles, rng, fake)

    print("[5/5] writing artefacts ...")
    df.to_csv(SESSIONS_RAW, index=False)
    profiles.to_csv(USER_PROFILES, index=False)

    active = df[df.active]
    counts = {
        "grid_rows": int(len(df)),
        "active_sessions": int(len(active)),
        "positive_sessions": int(active.is_true_threat.sum()),
        "positive_rate": float(active.is_true_threat.mean()),
        "benign_burst_sessions": int(active.is_benign_burst.sum()),
        "by_threat_class": {
            k: int(v) for k, v in active.threat_class.value_counts().items()
        },
        "episodes": {
            k: int(active[active.threat_class == k].episode_id.nunique())
            for k in plan if k != "benign_burst"
        } | {"benign_burst": int(active[active.is_benign_burst].episode_id.nunique())},
        "per_split": {
            s: {
                "sessions": int((active.split == s).sum()),
                "positives": int(active[active.split == s].is_true_threat.sum()),
                "benign_bursts": int(active[active.split == s].is_benign_burst.sum()),
                "by_class": {k: int(v) for k, v in
                             active[active.split == s].threat_class.value_counts().items()},
            } for s in ("train", "val", "test")
        },
    }
    config = {"args": vars(args), "episode_plan": plan, "realised": counts}
    GEN_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\n  grid rows        : {counts['grid_rows']:,}")
    print(f"  active sessions  : {counts['active_sessions']:,}")
    print(f"  positives        : {counts['positive_sessions']:,} "
          f"({counts['positive_rate'] * 100:.2f}% of sessions)")
    print(f"  benign bursts    : {counts['benign_burst_sessions']:,} (labelled NEGATIVE)")
    print(f"  by threat class  : {counts['by_threat_class']}")
    for s in ("train", "val", "test"):
        ps = counts["per_split"][s]
        print(f"  {s:<5} sessions={ps['sessions']:>6,}  pos={ps['positives']:>4}  "
              f"benign_burst={ps['benign_bursts']:>3}")
    print(f"\n  -> {SESSIONS_RAW}\n  -> {USER_PROFILES}\n  -> {GEN_CONFIG}")


if __name__ == "__main__":
    main()
