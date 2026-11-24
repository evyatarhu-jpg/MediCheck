"""
Matched end-to-end scenarios for MediCheck: for each scenario an order screen,
a photo of the packages for those orders, and a photo of what was prepared
(labelled syringes / medicine cups), plus the expected result at every step.

Fictional patient, generic names, fictional manufacturer, labels marked as test
samples. Doses are test values, not clinical guidance.

Usage:  python3 tests/generate_scenarios.py   (writes tests/scenarios/)
"""

import json
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

import generate_fixtures as gf

OUT = Path(__file__).parent / "scenarios"

# Package definitions used by the scenarios (generic names, fictional maker).
PK = {
    "meropenem": dict(generic="Meropenem", kind="vial", form="powder", label="MEROPENEM", sub="500 mg powder · 50 mg/mL after reconstitution with 10 mL", conc=(50, "mg", 1), strength=None, route="IV", cap="#1565c0"),
    "furosemide": dict(generic="Furosemide", kind="ampoule", form="injectable_solution", label="FUROSEMIDE", sub="20 mg/2 mL", conc=(20, "mg", 2), strength=None, route="IV / IM"),
    "spironolactone": dict(generic="Spironolactone", kind="box", form="tablet", label="SPIRONOLACTONE", sub="25 mg · 30 tablets", conc=None, strength=(25, "mg"), route="Oral"),
    "adrenaline": dict(generic="Epinephrine", kind="ampoule", form="injectable_solution", label="ADRENALINE", sub="1 mg/mL (1:1000)", conc=(1, "mg", 1), strength=None, route="IM / IV"),
    "captopril": dict(generic="Captopril", kind="bottle", form="oral_liquid", label="CAPTOPRIL", sub="Oral solution 1 mg/mL", conc=(1, "mg", 1), strength=None, route="Oral"),
    "ceftriaxone": dict(generic="Ceftriaxone", kind="vial", form="powder", label="CEFTRIAXONE", sub="1 g powder for injection", conc=None, strength=None, route="IV / IM", cap="#6a1b9a"),
    "ondansetron": dict(generic="Ondansetron", kind="ampoule", form="injectable_solution", label="ONDANSETRON", sub="4 mg/2 mL", conc=(4, "mg", 2), strength=None, route="IV / IM"),
    "paracetamol_iv": dict(generic="Paracetamol", kind="vial", form="injectable_solution", label="PARACETAMOL", sub="10 mg/mL · solution for infusion · 50 mL", conc=(10, "mg", 1), strength=None, route="IV infusion", cap="#2e7d32"),
    "digoxin": dict(generic="Digoxin", kind="bottle", form="oral_liquid", label="DIGOXIN", sub="Oral solution 50 mcg/mL", conc=(50, "mcg", 1), strength=None, route="Oral"),
    "levetiracetam": dict(generic="Levetiracetam", kind="bottle", form="oral_liquid", label="LEVETIRACETAM", sub="Oral solution 100 mg/mL", conc=(100, "mg", 1), strength=None, route="Oral"),
    "kcl": dict(generic="Potassium Chloride", kind="vial", form="injectable_solution", label="POTASSIUM CHLORIDE", sub="2 mEq/mL · CONCENTRATE · MUST BE DILUTED", conc=(2, "mEq", 1), strength=None, route="IV after dilution", cap="#f9a825"),
    "heparin": dict(generic="Heparin", kind="vial", form="injectable_solution", label="HEPARIN SODIUM", sub="5,000 units/mL", conc=(5000, "units", 1), strength=None, route="IV / SC", cap="#2e7d32"),
}

# Each order: (generic, screen name, prefix, dose, unit, route, freq, prn, infusion)
# expect: prep = {"status": "ok", "amount": x, "unit": "mL"|"tablets"} or {"status": "stop", "reason": ...}
#         verify = "match" | "stop" | "review"
SCENARIOS = [
    dict(
        title="הכנה תקינה של שלוש תרופות",
        weight=8.4,
        orders=[
            (("Meropenem", "MEROPENEM", "I.", 42, "mg", "IV", "8 h", False, False),
             dict(prep={"status": "ok", "amount": 0.84, "unit": "mL", "note": "אבקה: התראת המסה"}, verify="match")),
            (("Furosemide", "FUROSEMIDE", "I.", 2, "mg", "IV", "12 h", False, False),
             dict(prep={"status": "ok", "amount": 0.2, "unit": "mL"}, verify="match")),
            (("Spironolactone", "SPIRONOLACTONE", "TAB.", 25, "mg", "PO", "24 h", False, False),
             dict(prep={"status": "ok", "amount": 1, "unit": "tablets"}, verify="match")),
        ],
        packages=["meropenem", "furosemide", "spironolactone"],
        prepared=[("syringe", "MEROPENEM", "0.84 mL"), ("syringe", "FUROSEMIDE", "0.2 mL"), ("cup", "SPIRONOLACTONE", "1 tablet")],
    ),
    dict(
        title="טעות עשרונית בהכנה והמרת יחידות",
        weight=3.6,
        orders=[
            (("Captopril", "CAPTOPRIL", "SYR.", 1.5, "mg", "PO", "8 h", False, False),
             dict(prep={"status": "ok", "amount": 1.5, "unit": "mL"}, verify="stop", verify_note="במזרק סומן 15 mL במקום 1.5 mL")),
            (("Epinephrine", "ADRENALINE", "I.", 10, "mcg", "IV", "", True, False),
             dict(prep={"status": "ok", "amount": 0.01, "unit": "mL", "note": "המרת mcg ל-mg והמלצה למהול"}, verify="review", verify_note="לא הוכן ולא צולם")),
        ],
        packages=["captopril", "adrenaline"],
        prepared=[("syringe", "CAPTOPRIL", "15 mL")],
    ),
    dict(
        title="אריזה חסרה ושם דומה",
        weight=12.0,
        orders=[
            (("Cefazolin", "CEFAZOLIN", "I.", 250, "mg", "IV", "8 h", False, False),
             dict(prep={"status": "stop", "reason": "missing_package"}, verify="stop", verify_note="במקום ceFAZolin צולמה cefTRIAXone")),
            (("Ondansetron", "ONDANSETRON", "I.", 0.8, "mg", "IV", "8 h", True, False),
             dict(prep={"status": "ok", "amount": 0.4, "unit": "mL"}, verify="match")),
        ],
        packages=["ceftriaxone", "ondansetron"],
        prepared=[("syringe", "ONDANSETRON", "0.4 mL")],
    ),
    dict(
        title="דרך מתן לא מתאימה ואריזה שפג תוקפה",
        weight=6.2,
        orders=[
            (("Paracetamol", "PARACETAMOL", "SYR.", 90, "mg", "PO", "6 h", True, False),
             dict(prep={"status": "stop", "reason": "route"}, verify="stop", verify_note="הוראה PO מול תכשיר לעירוי")),
            (("Digoxin", "DIGOXIN", "SYR.", 10, "mcg", "PO", "12 h", False, False),
             dict(prep={"status": "stop", "reason": "expired"}, verify="stop", verify_note="האריזה פגת תוקף")),
        ],
        packages=["paracetamol_iv", "digoxin"],
        expired=["digoxin"],
        prepared=[("syringe", "DIGOXIN", "0.2 mL")],
    ),
    dict(
        title="חצי טבליה",
        weight=9.8,
        orders=[
            (("Spironolactone", "SPIRONOLACTONE", "TAB.", 12.5, "mg", "PO", "12 h", False, False),
             dict(prep={"status": "stop", "reason": "fraction_tablet"}, verify="stop", verify_note="הוכנה חצי טבליה למרות STOP")),
            (("Levetiracetam", "LEVETIRACETAM", "SYR.", 100, "mg", "PO", "12 h", False, False),
             dict(prep={"status": "ok", "amount": 1, "unit": "mL"}, verify="match")),
        ],
        packages=["spironolactone", "levetiracetam"],
        prepared=[("cup", "SPIRONOLACTONE", "0.5 tablet"), ("syringe", "LEVETIRACETAM", "1 mL")],
    ),
    dict(
        title="תרופות בסיכון גבוה",
        weight=15.5,
        orders=[
            (("Potassium Chloride", "POTASSIUM CHLORIDE", "I.", 2, "mEq", "IV", "12 h", False, False),
             dict(prep={"status": "ok", "amount": 1, "unit": "mL", "note": "סיכון גבוה: נדרש אימות כפול"}, verify="match")),
            (("Heparin", "HEPARIN", "I.", 50, "units", "IV", "", True, False),
             dict(prep={"status": "ok", "amount": 0.01, "unit": "mL", "note": "נפח קטן: מומלץ למהול"}, verify="review", verify_note="לא צולם")),
        ],
        packages=["kcl", "heparin"],
        prepared=[("syringe", "POTASSIUM CHLORIDE", "1 mL")],
    ),
]

PREPARED_CSS = """
body { margin:0; width:1400px; height:900px; background: radial-gradient(circle at 40% 30%, #f6f6f3, #d9d9d3 75%); font-family:'DejaVu Sans', Arial; }
.stage { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:space-evenly; }
.syr { position:relative; width:900px; height:110px; filter: drop-shadow(0 10px 10px rgba(0,0,0,.25)); }
.barrel { position:absolute; left:120px; top:25px; width:620px; height:60px; background:linear-gradient(180deg,#f4fbff,#dfe9ee 60%,#c9d6dc); border:2px solid #b8c6cc; border-radius:8px; }
.tipx { position:absolute; left:60px; top:45px; width:60px; height:20px; background:#d6e0e4; border-radius:4px; }
.plunger { position:absolute; left:740px; top:40px; width:140px; height:30px; background:#eeeeee; border:2px solid #c4c4c4; }
.sticker { position:absolute; left:260px; top:30px; padding:6px 14px; background:#fff; border:2px solid #333; font-size:26px; font-weight:bold; color:#111; white-space:nowrap; }
.cup { width:260px; height:200px; background:linear-gradient(90deg,#fafafa,#ffffff 40%,#e6e6e6); border:2px solid #cfcfcf; border-radius:10px 10px 60px 60px; position:relative; filter: drop-shadow(0 10px 10px rgba(0,0,0,.2)); }
.cup .sticker { left:auto; right:-260px; top:60px; }
.row { display:flex; gap:40px; align-items:center; }
"""


def prepared_html(items):
    parts = []
    for kind, name, amount in items:
        text = f"{name} {amount}"
        if kind == "syringe":
            parts.append(f'<div class="syr"><div class="tipx"></div><div class="barrel"></div><div class="plunger"></div>'
                         f'<div class="sticker">{text}</div></div>')
        else:
            parts.append(f'<div class="row"><div class="cup"><div class="sticker">{text}</div></div><div style="width:260px"></div></div>')
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{PREPARED_CSS}</style></head><body><div class='stage'>{''.join(parts)}</div></body></html>"


UNIT_HE = {"mL": "mL", "tablets": "טבליות"}
REASON_HE = {
    "missing_package": "STOP: לא נמצאה אריזה מתאימה",
    "route": "STOP: דרך המתן לא מתאימה לתכשיר",
    "expired": "STOP: פג תוקף",
    "fraction_tablet": "STOP: נדרש חלק של טבליה",
}
VERIFY_HE = {"match": "MATCH", "stop": "STOP", "review": "נדרש אימות"}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    tmp = OUT / "_tmp"
    tmp.mkdir()

    sheet = ["# תרחישי בדיקה מקצה לקצה\n",
             "בכל תיקייה שלושה צילומים לפי סדר השלבים באפליקציה: 1 הוראות, 2 אריזות, 3 מה שהוכן.",
             "בחר אותם ב\"בחירת תמונה קיימת\" בכל שלב, ובחר את כל ההוראות. דרך המתן מופיעה במסך.",
             "המטופל והיצרן בדויים. המינונים הם ערכי בדיקה בלבד.\n"]

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(device_scale_factor=1.5)

        for n, sc in enumerate(SCENARIOS, 1):
            folder = OUT / f"scenario_{n:02d}"
            folder.mkdir()
            gf.rows_weight[0] = sc["weight"]

            # ---- 1. orders
            rows = [{"kind": "task", "text": t, "freq": gf.RNG.choice(["", "8 h", "3 h"])}
                    for t in gf.RNG.sample(gf.NURSING_ROWS, 5)]
            expected = []
            for (o, exp) in sc["orders"]:
                generic, screen_name, prefix, dose, unit, route, freq, prn, infusion = o
                od = dict(generic=generic, screen_name=screen_name, prefix=prefix, dose=dose, unit=unit,
                          route=route, freq=freq, prn=prn, infusion=infusion)
                rows.insert(gf.RNG.randint(0, len(rows)), {"kind": "drug", "order": od})
                expected.append(dict(generic=generic, dose=dose, unit=unit, route=route, **exp))
            raw = tmp / f"o{n}.png"
            gf.render(page, gf.order_screen_html(rows, True, False), raw, 950, 1400, selector=".win")
            gf.phone_photo(raw, folder / "1_orders.jpg", screen=True, severity=0.7)

            # ---- 2. packages
            parts, pk_truth = [], []
            for key in sc["packages"]:
                p = PK[key]
                exp_date, lot = gf.expiry_and_lot(key in sc.get("expired", []))
                parts.append(gf.package_html(p, exp_date, lot, 0))
                pk_truth.append({"generic": p["generic"], "expiry": exp_date, "expired": key in sc.get("expired", [])})
            html = f"<!doctype html><html><head><meta charset='utf-8'><style>{gf.PACKAGE_CSS}</style></head><body><div class='stage'>{''.join(parts)}</div></body></html>"
            raw = tmp / f"p{n}.png"
            gf.render(page, html, raw, 1400, 1000)
            gf.phone_photo(raw, folder / "2_packages.jpg", screen=False, severity=0.6)

            # ---- 3. prepared
            raw = tmp / f"r{n}.png"
            gf.render(page, prepared_html(sc["prepared"]), raw, 1400, 900)
            gf.phone_photo(raw, folder / "3_prepared.jpg", screen=False, severity=0.5)

            (folder / "expected.json").write_text(json.dumps({
                "title": sc["title"], "patient_weight_kg": sc["weight"],
                "orders": expected, "packages": pk_truth,
                "prepared": [{"kind": k, "label": f"{nm} {a}"} for k, nm, a in sc["prepared"]],
            }, ensure_ascii=False, indent=2))

            sheet.append(f"\n## תרחיש {n}: {sc['title']}\n")
            sheet.append(f"משקל להזנה: {sc['weight']} ק\"ג\n")
            sheet.append("| תרופה | הוראה | שלב 2: חישוב צפוי | שלב 3: אימות צפוי |")
            sheet.append("|---|---|---|---|")
            for e in expected:
                prep = e["prep"]
                if prep["status"] == "ok":
                    prep_txt = f"{gf.fmt_num(prep['amount'])} {UNIT_HE[prep['unit']]}"
                    if prep.get("note"):
                        prep_txt += f" ({prep['note']})"
                else:
                    prep_txt = REASON_HE[prep["reason"]]
                ver = VERIFY_HE[e["verify"]] + (f" ({e['verify_note']})" if e.get("verify_note") else "")
                sheet.append(f"| {e['generic']} | {gf.fmt_num(e['dose'])} {e['unit']} {e['route']} | {prep_txt} | {ver} |")

        browser.close()

    shutil.rmtree(tmp)
    (OUT / "README.md").write_text("\n".join(sheet) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
