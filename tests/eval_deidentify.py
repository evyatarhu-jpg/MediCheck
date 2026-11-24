"""
Check the de-identification step on every order screen we have, clean and as
simulated phone photos:
  * leak: after masking, OCR must find no identifier (keyword, 7+ digit run,
    or the fixture patient's name) anywhere in the image
  * kept: every medication row must still be readable (its dose still found)

  python3 tests/eval_deidentify.py [--save DIR]
"""
import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MEDICHECK_PASSWORD", "eval")

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

import app  # noqa: E402
import deidentify  # noqa: E402
from simulate_photo import simulate  # noqa: E402

NAME_WORDS = ["ישראל", "בדוי", "ישראלי", "כהן"]


def ocr_lines(bgr):
    return app._ocr_lines_for_masking(bgr)


def read_back(jpeg):
    img = cv2.imdecode(__import__("numpy").frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = min(1.0, app.OCR_MAX_SIDE / float(max(h, w)))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return [l["text"] for l in ocr_lines(img)]


def doses_of(expected):
    out = []
    for o in expected.get("orders", []):
        d = o.get("dose")
        if d in (None, ""):
            continue
        out.append((o.get("generic") or o.get("drug") or "?", float(d)))
    return out


def dose_found(texts, value):
    joined = " ".join(texts).replace(",", ".")
    for m in re.finditer(r"\d+(?:\.\d+)?", joined):
        if abs(float(m.group()) - value) < 1e-6:
            return True
    return False


def cases():
    for f in sorted((ROOT / "tests/scenarios").glob("scenario_*")):
        yield f.name, f / "1_orders.jpg", json.loads((f / "expected.json").read_text())
    for j in sorted((ROOT / "tests/fixtures/orders").glob("*.json")):
        yield j.stem, j.with_suffix(".jpg"), json.loads(j.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    a = ap.parse_args()
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)

    total = leaks = kept = rows = fails = 0
    ocr_unreadable = [0]
    for level in (0, 1, 2, 3):
        for name, path, exp in cases():
            img = Image.open(path).convert("RGB")
            if level:
                img = simulate(img, level, seed=sum(map(ord, name)))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            total += 1
            try:
                masked, info = deidentify.mask_image(buf.getvalue(), ocr_lines, True, app.OCR_MAX_SIDE)
            except ValueError as e:
                fails += 1
                print(f"L{level} {name}: not sent ({e})")
                continue
            if a.save:
                Path(a.save, f"L{level}_{name}.jpg").write_bytes(masked)
            texts = read_back(masked)
            leaked = [t for t in texts if deidentify.is_identifier_text(t) or any(w in t for w in NAME_WORDS)]
            want = doses_of(exp)
            missing = [g for g, d in want if not dose_found(texts, d)]
            if missing:
                # Only count rows the mask hid: rows OCR cannot read even unmasked are an OCR problem.
                before = read_back(buf.getvalue())
                unreadable = [g for g, d in want if g in missing and not dose_found(before, d)]
                missing = [g for g in missing if g not in unreadable]
                want = [x for x in want if x[0] not in unreadable]
                ocr_unreadable[0] += len(unreadable)
            rows += len(want)
            kept += len(want) - len(missing)
            if leaked:
                leaks += 1
            if leaked or missing:
                print(f"L{level} {name}: masked {info['masked_fraction']:.0%}"
                      + (f" LEAK {leaked}" if leaked else "") + (f" lost {missing}" if missing else ""))
    print(f"=== De-identification: {total} images · leaks {leaks} · not sent {fails} · "
          f"rows hidden by the mask {rows - kept}/{rows} · rows OCR cannot read even unmasked {ocr_unreadable[0]} ===")


if __name__ == "__main__":
    main()
