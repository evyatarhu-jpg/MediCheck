"""
Generate synthetic, fictional test images with known answers for MediCheck:
  - EMR-style medication-order screens (fictional patient, mixed Hebrew/English)
  - package photos (vials, ampoules, tablet boxes, oral bottles; generic names,
    fictional manufacturer, marked as test samples)
Each image gets a JSON file with the expected result. Images are degraded to
look like phone photos (perspective, blur, glare, noise, JPEG).

Doses are plausible-looking test values only. They are NOT clinical guidance.

Usage:  python3 tests/generate_fixtures.py   (writes tests/fixtures/)
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "fixtures"
RNG = random.Random(20261124)

# ---------------------------------------------------------------- orders ----

# (generic, screen name, prefix, dose, unit, route shown in route column, frequency, prn, infusion)
ORDER_POOL = [
    ("Meropenem", "MEROPENEM", "I.", 40, "mg", "IV", "8 h", False, False),
    ("Vancomycin", "VANCOMYCIN", "I.", 150, "mg", "IV", "6 h", False, False),
    ("Furosemide", "FUROSEMIDE(Fusid)", "I.", 2, "mg", "IV", "12 h", False, False),
    ("Captopril", "CAPTOPRIL", "SYR.", 1.5, "mg", "PO", "8 h", False, False),
    ("Clonazepam", "CLONAZEPAM", "GTTS.", 0.1, "mg", "PO", "8 h", False, False),
    ("Promethazine", "PROMETHAZINE", "I.", 1.05, "mg", "IV", "", True, False),
    ("Pantoprazole", "PANTOPRAZOLE", "I.", 2.1, "mg", "IV", "24 h", False, False),
    ("Paracetamol", "PARACETAMOL", "SYR.", 120, "mg", "PO", "6 h", True, False),
    ("Spironolactone", "SPIRONOLACTONE", "TAB.", 6.25, "mg", "PO", "12 h", False, False),
    ("Enoxaparin", "ENOXAPARIN", "I.", 1.5, "mg", "SC", "12 h", False, False),
    ("Amiodarone", "AMIODARONE", "I.", 25, "mg", "IV", "12 h", False, False),
    ("Digoxin", "DIGOXIN", "SYR.", 10, "mcg", "PO", "12 h", False, False),
    ("Hydrocortisone", "HYDROCORTISONE", "I.", 5, "mg", "IV", "6 h", False, False),
    ("Ondansetron", "ONDANSETRON", "I.", 0.8, "mg", "IV", "8 h", True, False),
    ("Ceftriaxone", "CEFTRIAXONE", "I.", 400, "mg", "IV", "24 h", False, False),
    ("Morphine", "MORPHINE", "I.", 0.5, "mg", "IV", "4 h", True, False),
    ("Midazolam", "MIDAZOLAM", "I.", 0.5, "mg", "IV", "", True, False),
    ("Sildenafil", "SILDENAFIL", "SUSP.", 2, "mg", "PO", "8 h", False, False),
    ("Potassium Chloride", "POTASSIUM CHLORIDE", "SYR.", 2, "mEq", "PO", "12 h", False, False),
    ("Ampicillin", "AMPICILLIN", "I.", 250, "mg", "IV", "6 h", False, False),
    ("Levetiracetam", "LEVETIRACETAM", "SYR.", 100, "mg", "PO", "12 h", False, False),
    ("Heparin", "HEPARIN", "I.", 10, "units/kg/h", "IV", "", False, True),
    ("Milrinone", "MILRINONE", "I.", 0.5, "mcg/kg/min", "IV", "", False, True),
    ("Epinephrine", "Adrenaline", "", 0.05, "mcg/kg/min", "IV", "", False, True),
]

NURSING_ROWS = [
    "אומדן כאב", "שינוי תנוחה כל 3 שעות", "טיפול פה", "מעקב אחר שתן ויציאה",
    "ראש מורם ל - 30 מעלות", "כימיה דחופה", "ספירת דם", "אומדן נשימתי לחולה מונשם",
    "טיפול עיניים", "מדידת היקף בטן", "אומדן מצב העור", "אומדן צנתר מרכזי",
    "בקרת חולה מונשם", "מאזן נוזלים", "לבצע טיפול פה",
]

ICON_DRUG = "💉"
ICON_TASK = "📋"


def fmt_num(x):
    s = ("%.3f" % x).rstrip("0").rstrip(".")
    return s


def order_screen_html(rows, show_route, truncate):
    trs = []
    for r in rows:
        if r["kind"] == "task":
            color = RNG.choice(["#c0262d", "#c0262d", "#c68a1a"])
            trs.append(
                f'<tr><td class="cb">☐</td><td class="ic">{ICON_TASK}</td>'
                f'<td class="name" style="color:{color}" dir="rtl">{r["text"]}</td>'
                f'<td class="dose"></td>{"<td class=route></td>" if show_route else ""}'
                f'<td class="freq" style="color:{color}" dir="ltr">{r.get("freq","")}</td></tr>'
            )
        else:
            o = r["order"]
            name = (o["prefix"] + o["screen_name"]) if o["prefix"] else o["screen_name"]
            if truncate and len(name) > 13:
                name = name[:13] + "..."
            color = "#c0262d" if not o["infusion"] else "#2e7d32"
            dose_txt = f'{fmt_num(o["dose"])} {o["unit"]}'
            trs.append(
                f'<tr><td class="cb">☐</td><td class="ic">{ICON_DRUG}</td>'
                f'<td class="name" style="color:{color}" dir="ltr">{name}</td>'
                f'<td class="dose" style="color:{color}" dir="ltr">{dose_txt}</td>'
                f'{"<td class=route style=color:" + color + ">" + o["route"] + "</td>" if show_route else ""}'
                f'<td class="freq" style="color:{color}" dir="ltr">{o["freq"] + (" PRN" if o["prn"] else "")}</td></tr>'
            )
    head_route = "<th>מתן</th>" if show_route else ""
    return f"""<!doctype html><html dir="rtl"><head><meta charset="utf-8"><style>
body {{ margin:0; background:#f1f4f7; font-family: 'DejaVu Sans', Arial; }}
.win {{ width: 900px; margin: 20px; background:#fff; border:1px solid #9aa7b4; }}
.title {{ background:#dfe7ef; padding:6px 10px; font-size:15px; color:#223; border-bottom:1px solid #9aa7b4; }}
.pt {{ padding:6px 10px; font-size:14px; color:#333; border-bottom:1px solid #ccd; }}
table {{ border-collapse: collapse; width:100%; font-size:21px; }}
th {{ background:#e8edf2; font-weight:normal; color:#444; padding:4px 8px; text-align:right; border-bottom:1px solid #b8c2cc; font-size:17px; }}
td {{ padding:3px 8px; border-bottom:1px solid #edf0f3; white-space:nowrap; }}
td.cb {{ width:22px; font-size:20px; color:#333; }} td.ic {{ width:26px; font-size:16px; }}
td.name {{ text-align:left; width:330px; }} td.freq {{ text-align:left; }} td.dose {{ text-align:left; width:170px; }}
</style></head><body><div class="win">
<div class="title">הוראות פעילות</div>
<div class="pt">מטופל: ישראל בדוי · ת.ז. 000000000 · משקל: {rows_weight[0]} ק"ג · ט"נ ילדים</div>
<table><tr><th></th><th></th><th>פרט הוראה</th><th>כמות/ערכים</th>{head_route}<th>תדירות</th></tr>
{''.join(trs)}</table></div></body></html>"""


rows_weight = [8.4]

# --------------------------------------------------------------- packages ---

# kind: vial | ampoule | box | bottle
PACKAGE_POOL = [
    dict(generic="Vancomycin", kind="vial", form="powder", label="VANCOMYCIN", sub="500 mg powder for solution", conc=None, strength=None, route="IV infusion after reconstitution", cap="#d32f2f"),
    dict(generic="Meropenem", kind="vial", form="powder", label="MEROPENEM", sub="1 g powder for injection", conc=None, strength=None, route="IV", cap="#1565c0"),
    dict(generic="Furosemide", kind="ampoule", form="injectable_solution", label="FUROSEMIDE", sub="20 mg/2 mL", conc=(20, "mg", 2), strength=None, route="IV / IM"),
    dict(generic="Epinephrine", kind="ampoule", form="injectable_solution", label="ADRENALINE", sub="1 mg/mL (1:1000)", conc=(1, "mg", 1), strength=None, route="IM / IV"),
    dict(generic="Midazolam", kind="ampoule", form="injectable_solution", label="MIDAZOLAM", sub="5 mg/mL", conc=(5, "mg", 1), strength=None, route="IV / IM"),
    dict(generic="Morphine", kind="ampoule", form="injectable_solution", label="MORPHINE SULFATE", sub="10 mg/mL", conc=(10, "mg", 1), strength=None, route="IV / SC"),
    dict(generic="Potassium Chloride", kind="vial", form="injectable_solution", label="POTASSIUM CHLORIDE", sub="2 mEq/mL · CONCENTRATE · MUST BE DILUTED", conc=(2, "mEq", 1), strength=None, route="IV after dilution", cap="#f9a825"),
    dict(generic="Heparin", kind="vial", form="injectable_solution", label="HEPARIN SODIUM", sub="5,000 units/mL", conc=(5000, "units", 1), strength=None, route="IV / SC", cap="#2e7d32"),
    dict(generic="Amiodarone", kind="ampoule", form="injectable_solution", label="AMIODARONE HCl", sub="150 mg/3 mL", conc=(150, "mg", 3), strength=None, route="IV"),
    dict(generic="Digoxin", kind="bottle", form="oral_liquid", label="DIGOXIN", sub="Oral solution 50 mcg/mL", conc=(50, "mcg", 1), strength=None, route="Oral"),
    dict(generic="Paracetamol", kind="bottle", form="oral_liquid", label="PARACETAMOL", sub="Oral suspension 120 mg/5 mL", conc=(120, "mg", 5), strength=None, route="Oral"),
    dict(generic="Captopril", kind="bottle", form="oral_liquid", label="CAPTOPRIL", sub="Oral solution 1 mg/mL", conc=(1, "mg", 1), strength=None, route="Oral"),
    dict(generic="Spironolactone", kind="box", form="tablet", label="SPIRONOLACTONE", sub="25 mg · 30 tablets", conc=None, strength=(25, "mg"), route="Oral"),
    dict(generic="Amoxicillin", kind="box", form="capsule", label="AMOXICILLIN", sub="250 mg · 20 capsules", conc=None, strength=(250, "mg"), route="Oral"),
    dict(generic="Enoxaparin", kind="ampoule", form="injectable_solution", label="ENOXAPARIN SODIUM", sub="20 mg/0.2 mL", conc=(20, "mg", 0.2), strength=None, route="SC"),
    dict(generic="Ondansetron", kind="ampoule", form="injectable_solution", label="ONDANSETRON", sub="4 mg/2 mL", conc=(4, "mg", 2), strength=None, route="IV / IM"),
    dict(generic="Lidocaine", kind="vial", form="injectable_solution", label="LIDOCAINE HCl 1%", sub="10 mg/mL · 20 mL", conc=(10, "mg", 1), strength=None, route="Infiltration / IV", cap="#6a1b9a"),
    dict(generic="Dopamine", kind="vial", form="injectable_solution", label="DOPAMINE HCl", sub="40 mg/mL · 5 mL", conc=(40, "mg", 1), strength=None, route="IV infusion after dilution", cap="#ef6c00"),
    dict(generic="Dobutamine", kind="vial", form="injectable_solution", label="DOBUTAMINE", sub="12.5 mg/mL · 20 mL", conc=(12.5, "mg", 1), strength=None, route="IV infusion after dilution", cap="#00838f"),
]

BAND_COLORS = ["#d32f2f", "#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#ad1457"]


def expiry_and_lot(expired):
    if expired:
        y, m = 2025, RNG.randint(1, 10)
    else:
        y, m = RNG.choice([2027, 2028]), RNG.randint(1, 12)
    return f"{y}-{m:02d}", f"L{RNG.randint(10000, 99999)}"


def package_html(p, exp, lot, rotate):
    band = RNG.choice(BAND_COLORS)
    label_inner = f"""
      <div class="band" style="background:{band}">{p['label']}</div>
      <div class="sub">{p['sub']}</div>
      <div class="route">{p['route']}</div>
      <div class="meta">LOT {lot} &nbsp; EXP {exp}</div>
      <div class="mfr">SamplePharma Ltd · TEST SAMPLE – NOT FOR CLINICAL USE</div>"""
    style_rot = f"transform: rotate({rotate}deg);" if rotate else ""
    k = p["kind"]
    if k == "vial":
        return f"""<div class="pkg vial" style="{style_rot}"><div class="cap" style="background:{p.get('cap','#888')}"></div><div class="neck"></div>
          <div class="body"><div class="label">{label_inner}</div></div></div>"""
    if k == "ampoule":
        return f"""<div class="pkg amp" style="{style_rot}"><div class="tip"></div><div class="abody"><div class="label">{label_inner}</div></div></div>"""
    if k == "bottle":
        return f"""<div class="pkg bottle" style="{style_rot}"><div class="bcap"></div><div class="bbody"><div class="label">{label_inner}</div></div></div>"""
    return f"""<div class="pkg box" style="{style_rot}"><div class="label boxlabel">{label_inner}</div></div>"""


PACKAGE_CSS = """
body { margin:0; width:1400px; height:1000px; background: radial-gradient(circle at 30% 20%, #f4f1ea, #d8d2c6 70%, #c9c2b4); font-family:'DejaVu Sans', Arial; }
.stage { position:absolute; inset:0; display:flex; align-items:center; justify-content:space-evenly; transform: scale(1.12); }
.pkg { position:relative; filter: drop-shadow(0 18px 18px rgba(0,0,0,.28)); }
.label { background:#fff; padding:10px 12px; color:#111; }
.band { color:#fff; font-weight:bold; font-size:28px; padding:6px 8px; letter-spacing:.5px; }
.sub { font-size:22px; font-weight:bold; margin-top:8px; }
.route { font-size:16px; margin-top:6px; }
.meta { font-size:14px; margin-top:8px; color:#333; }
.mfr { font-size:10px; margin-top:6px; color:#777; }
.vial { width:300px; } .vial .cap { width:150px; height:46px; margin:0 auto; border-radius:8px 8px 2px 2px; }
.vial .neck { width:120px; height:26px; margin:0 auto; background:rgba(220,230,235,.9); }
.vial .body { width:300px; padding:30px 0; background:linear-gradient(90deg,#dfe8ec,#f8fbfc 40%,#cfd9de); border-radius:26px; }
.amp { width:220px; } .amp .tip { width:50px; height:90px; margin:0 auto; background:linear-gradient(90deg,#d9c7a0,#f5ead2,#cdb98f); border-radius:25px 25px 4px 4px; }
.amp .abody { width:220px; padding:24px 0; background:linear-gradient(90deg,#e4d6b8,#fbf4e4 40%,#d6c49e); border-radius:18px; }
.amp .label { font-size:90%; } .amp .band { font-size:22px; } .amp .sub { font-size:19px; }
.bottle { width:300px; } .bottle .bcap { width:170px; height:60px; margin:0 auto; background:#f2f2f2; border:2px solid #ccc; border-radius:10px; }
.bottle .bbody { width:300px; padding:40px 0; background:linear-gradient(90deg,#8a5a2b,#c08a52 45%,#7a4c22); border-radius:30px; }
.box { width:380px; background:#fff; border:1px solid #ccc; } .boxlabel { padding:18px; }
"""


def render(page, html, path, width, height, selector=None):
    page.set_viewport_size({"width": width, "height": height})
    page.set_content(html)
    page.wait_for_timeout(50)
    if selector:
        box = page.locator(selector).bounding_box()
        pad = 40
        page.screenshot(path=str(path), clip={
            "x": max(0, box["x"] - pad), "y": max(0, box["y"] - pad),
            "width": box["width"] + 2 * pad, "height": box["height"] + 2 * pad})
    else:
        page.screenshot(path=str(path), full_page=True)


# ----------------------------------------------------- photo degradation ---

def phone_photo(src, dst, screen=False, severity=1.0):
    img = cv2.imread(str(src))
    h, w = img.shape[:2]

    # Perspective: a slightly skewed shot.
    d = 0.04 * severity
    jitter = lambda: RNG.uniform(-d, d)
    srcq = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dstq = np.float32([
        [w * (0.02 + abs(jitter())), h * (0.02 + abs(jitter()))],
        [w * (0.98 - abs(jitter())), h * (0.01 + abs(jitter()))],
        [w * (0.97 - abs(jitter())), h * (0.98 - abs(jitter()))],
        [w * (0.03 + abs(jitter())), h * (0.97 - abs(jitter()))],
    ])
    M = cv2.getPerspectiveTransform(srcq, dstq)
    fill = tuple(int(v) for v in cv2.mean(img)[:3])
    img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

    # Small rotation.
    angle = RNG.uniform(-3, 3) * severity
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, R, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

    f = img.astype(np.float32)

    if screen:
        # Monitor moiré: faint diagonal sinusoidal stripes.
        yy, xx = np.mgrid[0:h, 0:w]
        stripes = np.sin((xx + yy * 0.7) / RNG.uniform(2.2, 3.5)) * 6 * severity
        f += stripes[..., None]
        f *= np.array([1.0, 0.98, 1.04])  # monitor colour cast

    # Glare: soft bright blob.
    cx, cy = RNG.uniform(0.2, 0.8) * w, RNG.uniform(0.2, 0.8) * h
    yy, xx = np.mgrid[0:h, 0:w]
    glare = np.exp(-(((xx - cx) ** 2) / (2 * (w * 0.18) ** 2) + ((yy - cy) ** 2) / (2 * (h * 0.12) ** 2)))
    f += glare[..., None] * 55 * severity

    # Vignetting.
    vign = 1 - 0.18 * severity * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    f *= vign[..., None]

    f += np.random.default_rng(RNG.randint(0, 10**6)).normal(0, 4 * severity, f.shape)
    img = np.clip(f, 0, 255).astype(np.uint8)

    k = RNG.choice([3, 3, 5]) if severity > 0.6 else 3
    img = cv2.GaussianBlur(img, (k, k), 0)

    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, RNG.randint(68, 82)])


# --------------------------------------------------------------- main -------

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "orders").mkdir(parents=True)
    (OUT / "packages").mkdir(parents=True)
    tmp = OUT / "_tmp"
    tmp.mkdir()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(device_scale_factor=1.5)

        # ----- order screens
        for i in range(1, 9):
            weight = round(RNG.uniform(3.2, 24.0), 1)
            rows_weight[0] = weight
            chosen = RNG.sample(ORDER_POOL, RNG.randint(4, 7))
            show_route = i % 2 == 0
            truncate = i % 3 != 0
            rows, expected = [], []
            tasks = RNG.sample(NURSING_ROWS, RNG.randint(5, 9))
            for t in tasks:
                rows.append({"kind": "task", "text": t, "freq": RNG.choice(["", "", "8 h", "3 h"])})
            for c in chosen:
                generic, screen_name, prefix, dose, unit, route, freq, prn, infusion = c
                o = dict(generic=generic, screen_name=screen_name, prefix=prefix, dose=dose,
                         unit=unit, route=route, freq=freq, prn=prn, infusion=infusion)
                rows.insert(RNG.randint(0, len(rows)), {"kind": "drug", "order": o})
                base_unit = unit.split("/")[0]
                route_determinable = show_route or prefix in ("TAB.", "SYR.", "SUSP.")
                expected.append({
                    "generic": generic,
                    "dose": None if infusion else dose,
                    "unit": base_unit,
                    "rate": f"{fmt_num(dose)} {unit}" if infusion else None,
                    "route": route if route_determinable else None,
                    "route_on_screen": route if show_route else None,
                    "frequency": freq or None,
                    "prn": prn,
                    "continuous_infusion": infusion,
                })
            html = order_screen_html(rows, show_route, truncate)
            raw = tmp / f"orders_{i:02d}.png"
            render(page, html, raw, 950, 1400, selector=".win")
            severity = [0.5, 0.8, 1.0, 1.2][i % 4]
            phone_photo(raw, OUT / "orders" / f"orders_{i:02d}.jpg", screen=True, severity=severity)
            (OUT / "orders" / f"orders_{i:02d}.json").write_text(json.dumps({
                "image": f"orders_{i:02d}.jpg",
                "patient_weight_kg": weight,
                "route_column_shown": show_route,
                "names_truncated": truncate,
                "photo_severity": severity,
                "orders": expected,
            }, ensure_ascii=False, indent=2))

        # ----- package photos
        for i in range(1, 9):
            chosen = RNG.sample(PACKAGE_POOL, RNG.choice([2, 2, 3]))
            sideways = i in (3, 6)
            expired_index = 0 if i in (2, 7) else -1
            parts, expected = [], []
            for j, p in enumerate(chosen):
                exp, lot = expiry_and_lot(j == expired_index)
                rot = 90 if sideways else RNG.choice([0, 0, -4, 3])
                parts.append(package_html(p, exp, lot, rot))
                expected.append({
                    "generic": p["generic"],
                    "dosage_form": p["form"],
                    "strength": {"value": p["strength"][0], "unit": p["strength"][1]} if p["strength"] else None,
                    "concentration": {"amount": p["conc"][0], "unit": p["conc"][1], "volume_ml": p["conc"][2]} if p["conc"] else None,
                    "label_route": p["route"],
                    "expiry": exp,
                    "expired": j == expired_index,
                    "lot": lot,
                })
            html = f"<!doctype html><html><head><meta charset='utf-8'><style>{PACKAGE_CSS}</style></head><body><div class='stage'>{''.join(parts)}</div></body></html>"
            raw = tmp / f"packages_{i:02d}.png"
            render(page, html, raw, 1400, 1000)
            severity = [0.5, 0.8, 1.0, 1.2][i % 4]
            phone_photo(raw, OUT / "packages" / f"packages_{i:02d}.jpg", screen=False, severity=severity)
            (OUT / "packages" / f"packages_{i:02d}.json").write_text(json.dumps({
                "image": f"packages_{i:02d}.jpg",
                "sideways_labels": sideways,
                "photo_severity": severity,
                "packages": expected,
            }, ensure_ascii=False, indent=2))

        browser.close()

    shutil.rmtree(tmp)
    (OUT / "README.md").write_text(
        "# MediCheck test fixtures\n\n"
        "Synthetic, fictional images with known answers. Fictional patient and manufacturer.\n"
        "Doses are test values only and are not clinical guidance.\n\n"
        "- `orders/*.jpg` + `.json`: EMR-style order screens photographed off a monitor\n"
        "- `packages/*.jpg` + `.json`: vials, ampoules, tablet boxes and oral bottles\n\n"
        "Regenerate with `python3 tests/generate_fixtures.py`; score with `python3 tests/eval.py`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
