"""
Run every scenario in tests/scenarios through the real app code, step by step:
  1. read the order screen     2. read the packages, match, calculate
  3. read the prepared doses and verify
and compare each step with expected.json. Writes tests/scenario_report.html.

  python3 tests/eval_scenarios.py            # local OCR
  python3 tests/eval_scenarios.py --vision   # vision model (needs ANTHROPIC_API_KEY)
"""

import argparse
import base64
import html
import io
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SC = Path(__file__).resolve().parent / "scenarios"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEDICHECK_PASSWORD", "eval")

STEP_JS = r"""
async ([mode, ordersB64, packagesB64, preparedB64, weight, expectedRoutes]) => {
  async function post(url, b64) {
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const form = new FormData();
    form.append('image', new Blob([bytes], {type: 'image/jpeg'}), 'x.jpg');
    const r = await fetch(url, {method: 'POST', body: form, credentials: 'same-origin'});
    return r.json();
  }
  const out = {};

  // ---- Step 1: orders
  let orders;
  if (mode === 'vision') {
    const d = await post('/api/vision-orders', ordersB64);
    if (d.error) return {error: 'orders: ' + d.error};
    orders = (d.orders || []).map(visionOrderToOrder).filter(o => o.drugName);
  } else {
    const d = await post('/api/ocr-order', ordersB64);
    if (d.error) return {error: 'orders: ' + d.error};
    const lines = await refineHospitalDoseLines(d.lines || []);
    orders = parseMultipleOrders(d.text || '', d.segments || [], lines, Math.round((d.confidence || 0) * 100));
  }
  out.ordersRead = orders.map(o => ({drug: findKnownDrug(o.drugName || '') || o.drugName, dose: o.dose, unit: o.doseUnit, route: o.route || ''}));

  // The nurse selects the medication orders; if a route was not read she taps it on the card.
  out.routeTapped = [];
  orders.forEach(o => {
    const name = findKnownDrug(o.drugName || '') || o.drugName;
    if (!o.route && expectedRoutes[name]) { o.route = expectedRoutes[name]; out.routeTapped.push(name); }
  });
  selectedOrders = orders.filter(o => o.dose !== '' && o.doseUnit && o.route);
  patientWeight = weight;

  // ---- Step 2: packages, matching, calculation
  let pkgs;
  if (mode === 'vision') {
    const d = await post('/api/vision-packages', packagesB64);
    if (d.error) return {error: 'packages: ' + d.error};
    pkgs = (d.packages || []).map(v => visionPackageToPackage(v, 0, 0)).filter(p => p.genericDrug);
  } else {
    const d = await post('/api/ocr-medication', packagesB64);
    if (d.error) return {error: 'packages: ' + d.error};
    pkgs = parseMultiplePackages(d.text || '', d.lines || [], d.segments || [], Math.round((d.confidence || 0) * 100));
  }
  out.packagesRead = pkgs.map(p => ({drug: p.genericDrug, form: p.formulation, strength: p.strength, unit: p.unit,
    conc: p.concentrationDrug, concUnit: p.concentrationUnit, concVol: p.concentrationVolume, expiry: p.expiry || ''}));

  packageMatches = computeMatches(selectedOrders, pkgs);
  preparationPlan = packageMatches.map(m => {
    if (m.status === 'matched') {
      const pkg = m.candidates[0].package;
      return {order: m.order, package: pkg, calc: calculateBatchPreparation(m.order, pkg)};
    }
    return {order: m.order, package: null,
      calc: {ok: false, value: 'STOP', warnings: [], explanation: m.status === 'missing' ? 'לא נמצאה אריזה מתאימה' : 'נמצאו כמה אריזות אפשריות'}};
  });
  out.plan = preparationPlan.map(p => ({drug: findKnownDrug(p.order.drugName) || p.order.drugName, ok: p.calc.ok,
    value: p.calc.value, explanation: p.calc.explanation || '', warnings: p.calc.warnings || []}));

  // ---- Step 3: prepared doses
  const d3 = await post('/api/ocr-medication', preparedB64);
  if (d3.error) return {error: 'prepared: ' + d3.error};
  out.preparedText = (d3.lines || []).map(l => l.text);
  out.verify = parsePreparedVerification({lines: d3.lines || [], segments: d3.segments || []}).map(r => ({
    drug: findKnownDrug(r.drug) || r.drug, status: r.status, detected: r.detectedAmount, message: r.message
  }));
  return out;
}
"""


def start_server():
    import app as A
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", 8798, A.app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def reason_kind(text):
    t = text or ""
    if "לא נמצאה אריזה" in t:
        return "missing_package"
    if "פג תוקף" in t:
        return "expired"
    if "טבליה" in t or "קפסולה" in t:
        return "fraction_tablet"
    if "PO" in t or "דרך" in t or "להזרקה" in t:
        return "route"
    return "other"


def thumb(path, width=520):
    from PIL import Image
    im = Image.open(path)
    im = im.resize((width, int(im.size[1] * width / im.size[0])))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def close(a, b):
    try:
        return abs(float(a) - float(b)) <= 0.001 * max(1, abs(float(b)))
    except (TypeError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--photo", type=int, default=0,
                    help="1-3: turn the order screen into a simulated phone photo of a monitor")
    args = ap.parse_args()
    mode = "vision" if args.vision else "ocr"
    if args.photo:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from PIL import Image
        from simulate_photo import simulate
    start_server()
    from playwright.sync_api import sync_playwright

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.route("**/fonts.*/**", lambda r: r.abort())
        page.goto("http://127.0.0.1:8798/login")
        page.fill("#password", os.environ["MEDICHECK_PASSWORD"])
        page.click("button[type=submit]")
        page.wait_for_load_state("networkidle")

        for folder in sorted(SC.glob("scenario_*")):
            exp = json.loads((folder / "expected.json").read_text())
            def b64(n, folder=folder):
                if args.photo and n == "1_orders.jpg":
                    buf = io.BytesIO()
                    simulate(Image.open(folder / n), args.photo, seed=int(folder.name[-2:])).save(buf, "JPEG", quality=80)
                    return base64.b64encode(buf.getvalue()).decode()
                return base64.b64encode((folder / n).read_bytes()).decode()
            routes = {o["generic"]: o["route"] for o in exp["orders"]}
            out = page.evaluate(STEP_JS, [mode, b64("1_orders.jpg"), b64("2_packages.jpg"), b64("3_prepared.jpg"),
                                          exp["patient_weight_kg"], routes])

            rows = []
            for o in exp["orders"]:
                g = o["generic"]
                read = next((r for r in out.get("ordersRead", []) if r["drug"] == g), None)
                s1 = bool(read and close(read["dose"], o["dose"]) and (read["unit"] or "").lower() == o["unit"].lower())
                plan = next((p for p in out.get("plan", []) if p["drug"] == g), None)
                if plan is None:
                    s2_actual, s2 = "ההוראה לא נבחרה (לא נקראה)", False
                elif plan["ok"]:
                    s2_actual = plan["value"]
                    want = o["prep"]
                    s2 = want["status"] == "ok" and close(str(plan["value"]).split()[0], want["amount"])
                else:
                    s2_actual = "STOP: " + plan["explanation"]
                    s2 = o["prep"]["status"] == "stop" and reason_kind(plan["explanation"]) == o["prep"]["reason"]
                ver = next((v for v in out.get("verify", []) if v["drug"] == g), None)
                s3_actual = (ver["status"] + (f" ({ver['detected']})" if ver and ver["detected"] is not None else "")) if ver else "-"
                s3 = bool(ver and ver["status"] == o["verify"])
                want_prep = (f"{o['prep']['amount']} {o['prep']['unit']}" if o["prep"]["status"] == "ok"
                             else "STOP: " + o["prep"]["reason"])
                rows.append(dict(drug=g, order=f"{o['dose']} {o['unit']} {o['route']}",
                                 s1=s1, s1_actual=(f"{read['dose']} {read['unit']} {read['route'] or '(ללא דרך מתן)'}" if read else "לא נקרא"),
                                 s2=s2, s2_expected=want_prep, s2_actual=s2_actual,
                                 warnings=(plan or {}).get("warnings", []),
                                 s3=s3, s3_expected=o["verify"] + (f" ({o.get('verify_note')})" if o.get("verify_note") else ""),
                                 s3_actual=s3_actual, route_tapped=g in out.get("routeTapped", [])))
            results.append(dict(folder=folder.name, title=exp["title"], weight=exp["patient_weight_kg"],
                                error=out.get("error"), rows=rows, raw=out,
                                images=[thumb(folder / n) for n in ("1_orders.jpg", "2_packages.jpg", "3_prepared.jpg")]))
        browser.close()

    total = sum(len(r["rows"]) for r in results)
    ok = {k: sum(1 for r in results for x in r["rows"] if x[k]) for k in ("s1", "s2", "s3")}
    print(f"=== End-to-end scenarios ({mode}{', photo level ' + str(args.photo) if args.photo else ''}) ===")
    print(f"Step 1 order read correctly: {ok['s1']}/{total} · Step 2 calculation/STOP correct: {ok['s2']}/{total} · "
          f"Step 3 verification correct: {ok['s3']}/{total}")
    for r in results:
        for x in r["rows"]:
            bad = [n for n, k in (("1", "s1"), ("2", "s2"), ("3", "s3")) if not x[k]]
            if bad:
                print(f"  {r['folder']} {x['drug']}: wrong at step {','.join(bad)} · read {x['s1_actual']} · "
                      f"calc {x['s2_actual']} · verify {x['s3_actual']}")

    write_report(results, mode + (f'_photo{args.photo}' if args.photo else ''), total, ok)


def write_report(results, mode, total, ok):
    e = html.escape
    mark = lambda b: '<span class="ok">✓</span>' if b else '<span class="bad">✗</span>'
    parts = []
    for i, r in enumerate(results, 1):
        trs = "".join(
            f"<tr><td class='drug'>{e(x['drug'])}</td><td dir='ltr'>{e(x['order'])}</td>"
            f"<td>{mark(x['s1'])} <span dir='ltr'>{e(x['s1_actual'])}</span>{'<div class=note>דרך המתן נבחרה ידנית</div>' if x['route_tapped'] else ''}</td>"
            f"<td>{mark(x['s2'])} {e(str(x['s2_actual']))}<div class='exp'>צפוי: {e(x['s2_expected'])}</div>"
            + "".join(f"<div class='warn'>⚠ {e(w)}</div>" for w in x["warnings"]) +
            f"</td><td>{mark(x['s3'])} {e(x['s3_actual'])}<div class='exp'>צפוי: {e(x['s3_expected'])}</div></td></tr>"
            for x in r["rows"])
        imgs = "".join(f"<figure><img src='{src}'><figcaption>{cap}</figcaption></figure>"
                       for src, cap in zip(r["images"], ["1. מסך ההוראות", "2. האריזות", "3. מה שהוכן"]))
        err = f"<div class='bad'>שגיאה: {e(r['error'])}</div>" if r["error"] else ""
        parts.append(f"<section><h2>תרחיש {i}: {e(r['title'])}</h2><div class='meta'>משקל {r['weight']} ק\"ג</div>{err}"
                     f"<div class='imgs'>{imgs}</div><table><tr><th>תרופה</th><th>הוראה</th><th>שלב 1: קריאת ההוראה</th>"
                     f"<th>שלב 2: התאמה וחישוב</th><th>שלב 3: אימות לפני מתן</th></tr>{trs}</table></section>")

    engine = "OCR מקומי" if mode == "ocr" else "מודל ראייה"
    doc = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MediCheck – דוח תרחישים</title>
<style>
body {{ margin:0; background:#fff7ef; color:#280003; font-family: "Open Sans", Arial, sans-serif; }}
header {{ background: linear-gradient(120deg,#ff5400,#e8457a 65%,#a24bc9); color:#fff; padding:28px 24px; }}
header h1 {{ margin:0 0 6px; font-size:28px; }} .kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:14px; }}
.kpi {{ background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.5); border-radius:16px; padding:10px 16px; }}
.kpi b {{ display:block; font-size:24px; }}
main {{ max-width:1100px; margin:0 auto; padding:16px; }}
section {{ background:#fff; border:1px solid #f0dccb; border-radius:20px; padding:16px; margin:18px 0; box-shadow:0 8px 24px rgba(40,0,3,.06); }}
h2 {{ margin:0 0 4px; font-size:20px; }} .meta {{ color:rgba(40,0,3,.62); font-size:14px; }}
.imgs {{ display:flex; gap:10px; overflow-x:auto; margin:12px 0; }} figure {{ margin:0; flex:0 0 auto; }}
img {{ width:300px; border-radius:12px; border:1px solid #eee; }} figcaption {{ font-size:13px; color:rgba(40,0,3,.62); }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }} th {{ text-align:right; background:#feefe0; padding:8px; }}
td {{ padding:8px; border-top:1px solid #f3e6da; vertical-align:top; }} td.drug {{ font-weight:700; }}
.ok {{ color:#12805c; font-weight:800; }} .bad {{ color:#d92d20; font-weight:800; }}
.exp {{ color:rgba(40,0,3,.62); font-size:12px; margin-top:3px; }} .warn {{ color:#7a4a00; font-size:12px; }}
.note {{ font-size:12px; color:#a24bc9; }}
</style></head><body>
<header><h1>MediCheck · דוח תרחישים מקצה לקצה</h1><div>מנוע זיהוי: {engine} · {len(results)} תרחישים · {total} תרופות.
המטופל והאריזות בדויים, והמינונים הם ערכי בדיקה בלבד.</div>
<div class="kpis"><div class="kpi">שלב 1 · קריאת ההוראה<b>{ok['s1']}/{total}</b></div>
<div class="kpi">שלב 2 · חישוב או STOP<b>{ok['s2']}/{total}</b></div>
<div class="kpi">שלב 3 · אימות לפני מתן<b>{ok['s3']}/{total}</b></div></div></header>
<main>{''.join(parts)}</main></body></html>"""
    path = Path(__file__).parent / f"scenario_report_{mode}.html"
    path.write_text(doc, encoding="utf-8")
    (Path(__file__).parent / f"scenario_results_{mode}.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "images"} for r in results], ensure_ascii=False, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
