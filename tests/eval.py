"""
Score MediCheck recognition against the fixtures in tests/fixtures.

  python3 tests/eval.py            # local OCR pipeline (what runs without an API key)
  python3 tests/eval.py --vision   # vision model; needs ANTHROPIC_API_KEY in the environment

The real app runs in a headless browser against an in-process server, so the
score reflects the same parsing code the phone uses.
"""

import argparse
import base64
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEDICHECK_PASSWORD", "eval")

PAGE_JS = r"""
async ([kind, mode, b64]) => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const blob = new Blob([bytes], {type: 'image/jpeg'});
  const form = new FormData();
  const url = mode === 'vision'
    ? (kind === 'orders' ? '/api/vision-orders' : '/api/vision-packages')
    : (kind === 'orders' ? '/api/ocr-order' : '/api/ocr-medication');
  form.append('image', blob, 'x.jpg');
  const r = await fetch(url, {method: 'POST', body: form, credentials: 'same-origin'});
  const data = await r.json();
  if (data.error) return {error: data.error};

  if (kind === 'orders') {
    let orders;
    if (mode === 'vision') {
      orders = (data.orders || []).map(visionOrderToOrder).filter(o => o.drugName);
    } else {
      const lines = await refineHospitalDoseLines(data.lines || []);
      orders = parseMultipleOrders(data.text || '', data.segments || [], lines, Math.round((data.confidence || 0) * 100));
    }
    return {items: orders.map(o => ({
      drug: findKnownDrug(o.drugName || '') || o.drugName, dose: o.dose, unit: o.doseUnit,
      route: o.route || null, frequency: o.frequency || null, prn: !!o.prn, infusion: !!o.infusion
    })), weight: data.patient_weight_kg || extractWeightFromText(data.text || '') || null};
  }

  let pkgs;
  if (mode === 'vision') {
    pkgs = (data.packages || []).map(v => visionPackageToPackage(v, 0, 0)).filter(p => p.genericDrug);
  } else {
    pkgs = parseMultiplePackages(data.text || '', data.lines || [], data.segments || [], Math.round((data.confidence || 0) * 100));
  }
  return {items: pkgs.map(p => ({
    drug: findKnownDrug(p.genericDrug || '') || p.genericDrug, form: p.formulation,
    strength: p.strength, unit: p.unit, conc: p.concentrationDrug, concUnit: p.concentrationUnit,
    concVol: p.concentrationVolume, expiry: p.expiry || null
  }))};
}
"""


def start_server():
    import app as A
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", 8799, A.app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return False


def score_orders(expected, got):
    res = {"expected": 0, "found": 0, "dose_ok": 0, "dose_wrong": 0, "route_ok": 0, "route_guessed": 0,
           "infusion_ok": 0, "infusions": 0, "extra": 0, "details": []}
    used = set()
    for e in expected:
        match = next((i for i, g in enumerate(got) if i not in used and g["drug"] == e["generic"]), None)
        if e["continuous_infusion"]:
            res["infusions"] += 1
            if match is None or got[match]["infusion"] or got[match]["dose"] in ("", None):
                res["infusion_ok"] += 1
            else:
                res["details"].append(f"infusion {e['generic']} given a single dose {got[match]['dose']} {got[match]['unit']}")
            if match is not None:
                used.add(match)
            continue

        res["expected"] += 1
        if match is None:
            res["details"].append(f"missed {e['generic']} {e['dose']} {e['unit']}")
            continue
        used.add(match)
        g = got[match]
        res["found"] += 1
        if g["dose"] in ("", None):
            res["details"].append(f"{e['generic']}: dose not read (expected {e['dose']} {e['unit']})")
        elif close(g["dose"], e["dose"]) and (g["unit"] or "").lower() == e["unit"].lower():
            res["dose_ok"] += 1
        else:
            res["dose_wrong"] += 1
            res["details"].append(f"WRONG DOSE {e['generic']}: got {g['dose']} {g['unit']}, expected {e['dose']} {e['unit']}")
        if e["route"] is None:
            if g["route"]:
                res["route_guessed"] += 1
                res["details"].append(f"{e['generic']}: route '{g['route']}' guessed; screen does not state it")
            else:
                res["route_ok"] += 1
        elif g["route"] == e["route"]:
            res["route_ok"] += 1
        else:
            res["details"].append(f"{e['generic']}: route got '{g['route']}', expected {e['route']}")
    res["extra"] = len([i for i in range(len(got)) if i not in used])
    for i, g in enumerate(got):
        if i not in used:
            res["details"].append(f"extra order not on screen: {g['drug']} {g['dose']} {g['unit']}")
    return res


def per_ml(amount, volume):
    try:
        return float(amount) / float(volume or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def score_packages(expected, got):
    res = {"expected": len(expected), "found": 0, "value_ok": 0, "value_wrong": 0, "expired_flagged": 0,
           "expired": 0, "extra": 0, "details": []}
    used = set()
    for e in expected:
        match = next((i for i, g in enumerate(got) if i not in used and g["drug"] == e["generic"]), None)
        if e["expired"]:
            res["expired"] += 1
        if match is None:
            res["details"].append(f"missed {e['generic']}")
            continue
        used.add(match)
        g = got[match]
        res["found"] += 1
        if e["concentration"]:
            c = e["concentration"]
            want = per_ml(c["amount"], c["volume_ml"])
            have = per_ml(g["conc"], g["concVol"])
            if have is None:
                res["details"].append(f"{e['generic']}: concentration not read (expected {c['amount']} {c['unit']}/{c['volume_ml']} mL)")
            elif close(have, want, 1e-3) and (g["concUnit"] or "").lower() == c["unit"].lower():
                res["value_ok"] += 1
            else:
                res["value_wrong"] += 1
                res["details"].append(f"WRONG CONCENTRATION {e['generic']}: got {g['conc']} {g['concUnit']}/{g['concVol']} mL, expected {c['amount']} {c['unit']}/{c['volume_ml']} mL")
        elif e["strength"]:
            s = e["strength"]
            if g["strength"] in ("", None):
                res["details"].append(f"{e['generic']}: strength not read")
            elif close(g["strength"], s["value"]) and (g["unit"] or "").lower() == s["unit"].lower():
                res["value_ok"] += 1
            else:
                res["value_wrong"] += 1
                res["details"].append(f"WRONG STRENGTH {e['generic']}: got {g['strength']} {g['unit']}")
        else:
            res["value_ok"] += 1  # powder: identity is what matters here
        if e["expired"] and g.get("expiry") == e["expiry"]:
            res["expired_flagged"] += 1
    for i, g in enumerate(got):
        if i not in used:
            res["extra"] += 1
            res["details"].append(f"extra package: {g['drug']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true")
    args = ap.parse_args()
    mode = "vision" if args.vision else "ocr"
    if mode == "vision" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")

    start_server()
    from playwright.sync_api import sync_playwright

    report = {"mode": mode, "orders": [], "packages": []}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.route("**/fonts.*/**", lambda r: r.abort())
        page.goto("http://127.0.0.1:8799/login")
        page.fill("#password", os.environ["MEDICHECK_PASSWORD"])
        page.click("button[type=submit]")
        page.wait_for_load_state("networkidle")

        for kind in ("orders", "packages"):
            for jf in sorted((FIX / kind).glob("*.json")):
                truth = json.loads(jf.read_text())
                b64 = base64.b64encode((FIX / kind / truth["image"]).read_bytes()).decode()
                out = page.evaluate(PAGE_JS, [kind, mode, b64])
                if "error" in out:
                    report[kind].append({"image": truth["image"], "error": out["error"]})
                    continue
                if kind == "orders":
                    r = score_orders(truth["orders"], out["items"])
                    r["weight_ok"] = close(out.get("weight"), truth["patient_weight_kg"], 1e-3)
                else:
                    r = score_packages(truth["packages"], out["items"])
                r["image"] = truth["image"]
                report[kind].append(r)
        browser.close()

    def total(rows, key):
        return sum(r.get(key, 0) for r in rows if "error" not in r)

    o, p = report["orders"], report["packages"]
    print(f"\n=== MediCheck recognition score ({mode}) ===")
    print(f"Orders:   found {total(o,'found')}/{total(o,'expected')} · dose correct {total(o,'dose_ok')} · "
          f"WRONG dose {total(o,'dose_wrong')} · route correct/not-guessed {total(o,'route_ok')} · "
          f"route guessed {total(o,'route_guessed')} · infusions handled {total(o,'infusion_ok')}/{total(o,'infusions')} · "
          f"extra {total(o,'extra')} · weight read {sum(1 for r in o if r.get('weight_ok'))}/{len(o)}")
    print(f"Packages: found {total(p,'found')}/{total(p,'expected')} · value correct {total(p,'value_ok')} · "
          f"WRONG value {total(p,'value_wrong')} · expired read {total(p,'expired_flagged')}/{total(p,'expired')} · extra {total(p,'extra')}")
    for kind in ("orders", "packages"):
        for r in report[kind]:
            if r.get("error"):
                print(f"  {r['image']}: ERROR {r['error']}")
            for d in r.get("details", []):
                print(f"  {r['image']}: {d}")

    out = Path(__file__).parent / f"results_{mode}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
