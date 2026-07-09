"""
build_pii_testset.py
====================
Builds a labelled, three-category document corpus for a *real* evaluation of the
multimodal PII pipeline (paper Sec. III.B).

    (a) text_extractable  HR/finance-style .txt snippets
    (b) scanned           the same style rendered to an image, then degraded to
                          look photocopied: noise, slight rotation, JPEG artefacts
    (c) handwritten       rendered in a handwriting face, then degraded harder:
                          blur, low contrast, aggressive rotation, heavy JPEG

Every document ships a ground-truth entity list (type + exact value), so the
pipeline can be scored entity-by-entity rather than on a summary statistic.

SAFETY: every PII value is synthesised by Faker. No real person's data is used
anywhere. National ID numbers are random digits in the Pakistani CNIC *format*
and are not issued to anybody; credit-card numbers come from Faker's test
generators; e-mail addresses use the IANA-reserved `example.*` domains via
`ascii_safe_email()`. Nothing here should be treated as live PII, but it is
still synthetic-secret-shaped, so `evaluation/pii_testset/` is gitignored.

Fonts: rendering uses the OpenType faces that ship with Windows -- Arial /
Calibri / Times for print, and Segoe Script / Segoe Print / Comic Sans for
handwriting. They are read from C:/Windows/Fonts at build time and are never
redistributed. On a box without them, pass --font-dir or the build falls back to
PIL's bitmap default (and says so).

Outputs (evaluation/pii_testset/)
---------------------------------
    text/*.txt          scanned/*.jpg          handwritten/*.jpg
    ground_truth.json   every document, its text, and its entity list
    build_config.json   seed, parameters, and the realised document counts
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .pii_common import (BUILD_CONFIG, CATEGORY_FILE_TYPE, DEPARTMENTS,
                         GROUND_TRUTH, PLATFORMS, TESTSET_DIR, tier_counts)

WIN_FONTS = Path("C:/Windows/Fonts")
PRINT_FONTS = ("arial.ttf", "calibri.ttf", "times.ttf", "georgia.ttf", "tahoma.ttf")
HAND_FONTS = ("segoesc.ttf", "segoepr.ttf", "comic.ttf")

IMG_W = 1100
MARGIN = 42


# ---------------------------------------------------------------------------
# Synthetic PII generators (Faker only)
# ---------------------------------------------------------------------------
def _cnic(fake: Faker) -> str:
    """Pakistani CNIC *format*, random digits. Not a real identifier."""
    return f"{fake.numerify('#####')}-{fake.numerify('#######')}-{fake.numerify('#')}"


# Mobile network codes libphonenumber recognises as assigned in PK. Building
# numbers from `numerify('3##')` instead produced unassigned prefixes, which
# libphonenumber correctly rejects -- Presidio then "missed" 40% of the gold
# phones. That is a defect in the benchmark, not in the detector, so the ground
# truth now only contains numbers that are actually valid. (Measured: Presidio
# detects 30/30 phones on valid prefixes vs 15/25 on numerify'd ones.)
PK_MOBILE_PREFIXES = tuple(p for p in range(300, 350) if p != 338)


def _phone(fake: Faker) -> str:
    return f"+92 {fake.random_element(PK_MOBILE_PREFIXES)} {fake.numerify('#######')}"


# Schemes a corporate document would plausibly contain. Faker's `maestro` emits
# 12-digit numbers that no major issuer uses and that Presidio's 13-16 digit
# regex cannot match; excluding them removes a benchmark artefact. Mastercard is
# deliberately KEPT even though Presidio detects only ~6/20 of Faker's valid
# 16-digit Mastercard numbers -- that is a real recogniser gap, reported as a
# finding rather than engineered away.
CARD_SCHEMES = ("visa", "mastercard", "amex", "discover")


def _credit_card(fake: Faker) -> str:
    return fake.credit_card_number(card_type=fake.random_element(CARD_SCHEMES))


def _password(fake: Faker) -> str:
    # Alphanumeric only: punctuation would be mangled by OCR in ways that say
    # more about the renderer than about the detector. Documented in REAL_RESULTS.
    return fake.password(length=10, special_chars=False, upper_case=True,
                         lower_case=True, digits=True)


GENERATORS = {
    "PERSON": lambda f: f.name(),
    "NID": _cnic,
    "CREDIT_CARD": _credit_card,
    "PHONE": _phone,
    "EMAIL": lambda f: f.ascii_safe_email(),      # example.* reserved domains
    "DOB": lambda f: f.date_of_birth(minimum_age=21, maximum_age=62).strftime("%Y-%m-%d"),
    "ADDRESS": lambda f: f.street_address().replace("\n", ", "),
    "PASSWORD": _password,
}

# Field label -> entity type. The label text is *not* part of the entity value.
TEMPLATES: List[Tuple[str, List[str]]] = [
    ("EMPLOYEE RECORD - CONFIDENTIAL",
     ["PERSON", "NID", "DOB", "EMAIL", "PHONE", "ADDRESS"]),
    ("PAYROLL DISBURSEMENT SLIP",
     ["PERSON", "NID", "CREDIT_CARD", "EMAIL"]),
    ("VENDOR ONBOARDING FORM",
     ["PERSON", "EMAIL", "PHONE", "ADDRESS", "CREDIT_CARD"]),
    ("IT ACCOUNT PROVISIONING TICKET",
     ["PERSON", "EMAIL", "PASSWORD"]),
    ("CUSTOMER SUPPORT ESCALATION",
     ["PERSON", "PHONE", "EMAIL"]),
    ("INTERNAL MEMO - TEAM OFFSITE",
     ["PERSON"]),
    ("EXPENSE REIMBURSEMENT CLAIM",
     ["PERSON", "CREDIT_CARD", "ADDRESS", "DOB"]),
    ("BENEFITS ENROLMENT CONFIRMATION",
     ["PERSON", "NID", "DOB", "ADDRESS"]),
    ("PUBLIC PRESS RELEASE DRAFT",
     []),                                      # deliberately PII-free -> SAFE
    ("SECURITY INCIDENT REPORT",
     ["PERSON", "EMAIL", "PASSWORD", "PHONE"]),
]

LABELS = {
    "PERSON": "Name", "NID": "CNIC", "CREDIT_CARD": "Card No",
    "PHONE": "Phone", "EMAIL": "Email", "DOB": "Date of Birth",
    "ADDRESS": "Address", "PASSWORD": "Temp Password",
}

FILLER = [
    "Please file this document per retention policy R-14.",
    "Circulated to the reviewing manager only.",
    "Reference: ticket queue Q3 backlog.",
    "Do not forward outside the department.",
    "Signed off by the compliance desk.",
]


def make_document(fake: Faker, rng: random.Random) -> Tuple[str, List[Dict[str, str]]]:
    """One synthetic document: its text, and the entities embedded in it."""
    title, types = rng.choice(TEMPLATES)
    lines = [title, "=" * len(title), ""]
    entities: List[Dict[str, str]] = []

    for t in types:
        value = GENERATORS[t](fake)
        entities.append({"type": t, "value": value})
        lines.append(f"{LABELS[t]}: {value}")

    lines.append("")
    lines.append(rng.choice(FILLER))
    return "\n".join(lines), entities


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _load_font(font_dir: Path, names: Tuple[str, ...], size: int,
               rng: random.Random) -> Tuple[ImageFont.FreeTypeFont, str]:
    available = [n for n in names if (font_dir / n).exists()]
    if not available:
        return ImageFont.load_default(), "PIL_default_bitmap"
    name = rng.choice(available)
    return ImageFont.truetype(str(font_dir / name), size), name


def _render(text: str, font: ImageFont.FreeTypeFont, line_h: int) -> Image.Image:
    lines = text.split("\n")
    h = MARGIN * 2 + line_h * len(lines)
    img = Image.new("L", (IMG_W, h), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((MARGIN, MARGIN + i * line_h), line, fill=20, font=font)
    return img


def _degrade(img: Image.Image, rng: random.Random, *, rotate: float, blur: float,
             noise: float, jpeg_q: int, contrast: float) -> Tuple[Image.Image, int]:
    """Rotate -> blur -> contrast -> noise -> JPEG. Returns (image, jpeg_quality)."""
    angle = rng.uniform(-rotate, rotate)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(blur * 0.5, blur)))
    if contrast < 1.0:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(contrast, 1.0))

    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.normal(0.0, rng.uniform(noise * 0.5, noise), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")

    q = rng.randint(jpeg_q, jpeg_q + 15)
    return img, q


DEGRADE = {
    # Photocopier-grade: legible, mild artefacts. Expected to clear tau_ocr=0.85.
    "scanned": dict(rotate=1.5, blur=0.7, noise=10.0, jpeg_q=40, contrast=0.85),
    # Whiteboard-photo-grade: soft, low contrast, skewed. Expected to fail tau_ocr.
    "handwritten": dict(rotate=4.0, blur=1.6, noise=18.0, jpeg_q=22, contrast=0.55),
}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--n-per-category", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--font-dir", type=Path, default=WIN_FONTS)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    Faker.seed(args.seed)
    fake = Faker("en_US")

    for sub in ("text", "scanned", "handwritten"):
        (TESTSET_DIR / sub).mkdir(parents=True, exist_ok=True)

    docs: List[Dict[str, Any]] = []
    fonts_used: Dict[str, int] = {}
    doc_n = 0

    for category in ("text_extractable", "scanned", "handwritten"):
        for _ in range(args.n_per_category):
            doc_n += 1
            text, entities = make_document(fake, rng)
            doc_id = f"DOC-{doc_n:04d}"
            h, m, l = tier_counts([e["type"] for e in entities])

            if category == "text_extractable":
                path = TESTSET_DIR / "text" / f"{doc_id}.txt"
                path.write_text(text, encoding="utf-8")
                font_name = "n/a"
            else:
                if category == "scanned":
                    font, font_name = _load_font(args.font_dir, PRINT_FONTS,
                                                 rng.randint(23, 27), rng)
                    line_h = 40
                else:
                    font, font_name = _load_font(args.font_dir, HAND_FONTS,
                                                 rng.randint(27, 32), rng)
                    line_h = 48
                img = _render(text, font, line_h)
                img, q = _degrade(img, rng, **DEGRADE[category])
                path = TESTSET_DIR / category / f"{doc_id}.jpg"
                img.convert("L").save(path, "JPEG", quality=q)
            fonts_used[font_name] = fonts_used.get(font_name, 0) + 1

            docs.append({
                "doc_id": doc_id,
                "category": category,
                "file_type": CATEGORY_FILE_TYPE[category],
                "path": str(path.relative_to(TESTSET_DIR)).replace("\\", "/"),
                "file_name": f"{doc_id}{path.suffix}",
                "text": text,
                "entities": entities,
                "high_entities": h, "medium_entities": m, "low_entities": l,
                "font": font_name,
                # metadata for the drop-in CSV
                "user_id": f"USR{rng.randint(1, 50):03d}",
                "user_name": fake.name(),
                "department": rng.choice(DEPARTMENTS),
                "platform": rng.choice(PLATFORMS),
                "timestamp": fake.date_time_between(
                    start_date="-21d", end_date="now").strftime("%Y-%m-%d %H:%M:%S"),
            })

    counts = {c: sum(d["category"] == c for d in docs)
              for c in ("text_extractable", "scanned", "handwritten")}
    ent_total: Dict[str, int] = {}
    for d in docs:
        for e in d["entities"]:
            ent_total[e["type"]] = ent_total.get(e["type"], 0) + 1

    GROUND_TRUTH.write_text(json.dumps({"documents": docs}, indent=1), encoding="utf-8")
    config = {
        "seed": args.seed,
        "n_per_category_requested": args.n_per_category,
        "realised_counts": counts,
        "total_documents": len(docs),
        "total_entities": sum(ent_total.values()),
        "entities_by_type": ent_total,
        "docs_with_zero_entities": sum(1 for d in docs if not d["entities"]),
        "degradation": DEGRADE,
        "fonts_used": fonts_used,
        "font_dir": str(args.font_dir),
        "pii_source": "Faker (synthetic). No real personal data.",
    }
    BUILD_CONFIG.write_text(json.dumps(config, indent=1), encoding="utf-8")

    print(f"built {len(docs)} documents")
    for c, n in counts.items():
        print(f"  {c:<18} {n}")
    print(f"\n  total entities  : {sum(ent_total.values())}")
    print(f"  by type         : {ent_total}")
    print(f"  PII-free docs   : {config['docs_with_zero_entities']}")
    print(f"  fonts           : {fonts_used}")
    if "PIL_default_bitmap" in fonts_used:
        print("  WARNING: real fonts not found; rendered with PIL's bitmap default.")
    print(f"\n  -> {TESTSET_DIR}\n  -> {GROUND_TRUTH}\n  -> {BUILD_CONFIG}")


if __name__ == "__main__":
    main()
