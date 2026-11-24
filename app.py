import hmac
import os
import secrets
import tempfile
import threading
from datetime import timedelta
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    session,
    url_for,
)

# Keep the OCR runtime lightweight and deterministic on hosted CPU services.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# RapidOCR runs the PaddleOCR detection/recognition models with ONNX Runtime.
# The models ship inside the pip package, so nothing is downloaded at startup,
# and peak memory stays well under 512 MB.
from rapidocr_onnxruntime import RapidOCR

import vision


BASE_DIR = Path(__file__).resolve().parent

# Do not expose the repository directory as a Flask static folder.
# MediCheck currently serves a single self-contained index.html.
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Authentication secrets must come from the deployment environment, never GitHub.
APP_PASSWORD = os.environ.get("MEDICHECK_PASSWORD", "")
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=True,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
)

_ocr = None
_ocr_lock = threading.Lock()
_ocr_loading = False
_ocr_error = None

# Only one OCR inference at a time: parallel runs double peak memory on small instances.
_predict_lock = threading.Lock()

# Larger images are downscaled before OCR to bound memory use and CPU time.
OCR_MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", "1280"))


LOGIN_HTML = """
<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="theme-color" content="#ff5400">
  <title>MATAMED – כניסה</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&family=Open+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --ink: #280003;
      --ink-soft: rgba(40, 0, 3, .62);
      --paper: #fff7ef;
      --brand: #ff5400;
      --sun: #ffd23f;
      --grad: linear-gradient(135deg, #ff5400 0%, #ff6a2b 28%, #e8457a 66%, #a24bc9 100%);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      min-height: 100dvh;
      display: flex;
      flex-direction: column;
      background: var(--paper);
      color: var(--ink);
      font-family: "Open Sans", Arial, sans-serif;
    }
    .hero {
      background: var(--grad);
      color: #fff;
      padding: 56px 24px 120px;
      text-align: right;
      position: relative;
      overflow: hidden;
    }
    .hero::after {
      content: "";
      position: absolute;
      left: -80px;
      bottom: -140px;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      background: rgba(255, 210, 63, .35);
      filter: blur(10px);
    }
    .hero-inner { max-width: 430px; margin: 0 auto; position: relative; z-index: 1; }
    .tag {
      display: inline-block;
      background: var(--sun);
      color: var(--ink);
      border-radius: 999px;
      padding: 3px 12px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .6px;
      text-transform: uppercase;
    }
    h1 {
      font-family: "Rubik", "Open Sans", sans-serif;
      font-weight: 600;
      font-size: 44px;
      letter-spacing: -1px;
      margin: 14px 0 6px;
      direction: ltr;
      text-align: right;
    }
    .wordmark {
      display: flex;
      gap: 0;
      align-items: baseline;
      justify-content: flex-end;
    }
    .wordmark b { font-weight: 700; color: #fff; }
    .wordmark span { font-weight: 300; color: rgba(255, 255, 255, .78); }
    .claim {
      direction: ltr;
      text-align: right;
      margin: 0 0 14px;
      font-size: 15px;
      font-weight: 400;
      letter-spacing: .3px;
      color: rgba(255, 255, 255, .9);
    }
    .claim b { font-weight: 800; color: #fff; }
    .tagline {
      font-family: "Rubik", "Open Sans", sans-serif;
      font-size: 22px;
      font-weight: 500;
      line-height: 1.3;
      margin: 0;
    }
    .tagline b { font-weight: 700; }
    main {
      width: min(430px, calc(100% - 32px));
      margin: -84px auto 24px;
      background: #fff;
      border: 1px solid #f0dccb;
      border-radius: 24px;
      box-shadow: 0 20px 50px rgba(40, 0, 3, .12);
      padding: 26px 22px;
      position: relative;
      z-index: 2;
    }
    .subtitle { color: var(--ink-soft); margin: 0 0 18px; line-height: 1.5; }
    label { display: block; font-weight: 700; margin-bottom: 8px; }
    input {
      width: 100%;
      min-height: 56px;
      border: 1px solid #ead5c3;
      border-radius: 14px;
      padding: 10px 14px;
      font-size: 19px;
      direction: ltr;
      color: var(--ink);
      background: #fff;
    }
    input:focus { outline: 3px solid rgba(255, 84, 0, .25); border-color: var(--brand); }
    button {
      width: 100%;
      min-height: 56px;
      margin-top: 16px;
      border: 0;
      border-radius: 999px;
      background: var(--ink);
      color: #fff;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: .3px;
      cursor: pointer;
      box-shadow: 0 10px 22px rgba(40, 0, 3, .2);
    }
    button:active { transform: translateY(1px); }
    .error {
      margin-top: 14px;
      padding: 10px 14px;
      border-radius: 14px;
      background: #fef3f2;
      color: #912018;
      border: 1px solid #f5b5ae;
    }
    .prototype {
      width: min(430px, calc(100% - 32px));
      margin: 0 auto 28px;
      color: var(--ink-soft);
      font-size: 13px;
      line-height: 1.5;
      text-align: center;
    }
  </style>
</head>
<body>
  <section class="hero">
    <div class="hero-inner">
      <span class="tag">Prototype</span>
      <h1 class="wordmark"><b>MATA</b><span>MED</span></h1>
      <p class="claim"><b>M</b>edication <b>A</b>ssurance <b>T</b>hrough <b>A</b>I</p>
      <p class="tagline">בטיחות חכמה בהכנת תרופות.<br><b>בדיוק לפי ההוראה.</b></p>
    </div>
  </section>

  <main>
    <p class="subtitle">גישה למערכת מוגנת בסיסמה.</p>

    <form method="post" autocomplete="off">
      <label for="password">סיסמה</label>
      <input
        id="password"
        name="password"
        type="password"
        required
        autofocus
        autocomplete="current-password"
      >
      <button type="submit">כניסה</button>
    </form>

    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
  </main>

  <div class="prototype">
    Prototype בלבד. אין להשתמש במערכת לצורך החלטה קלינית או מתן תרופה ללא אימות אנושי מלא.
  </div>
</body>
</html>
"""


CONFIG_HTML = """
<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MATAMED – הגדרה נדרשת</title>
</head>
<body style="font-family:Arial,sans-serif;max-width:760px;margin:50px auto;padding:20px;direction:rtl">
  <h1>MATAMED אינו מוגדר עדיין</h1>
  <p>יש להגדיר את <code>MEDICHECK_PASSWORD</code> ב־Environment Variables של השרת.</p>
</body>
</html>
"""


def get_ocr():
    global _ocr, _ocr_loading, _ocr_error

    if _ocr is not None:
        return _ocr

    with _ocr_lock:
        if _ocr is not None:
            return _ocr

        _ocr_loading = True
        _ocr_error = None

        try:
            print("Loading RapidOCR models...", flush=True)
            _ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
            print("RapidOCR ready.", flush=True)
            return _ocr
        except Exception as exc:
            _ocr_error = f"{type(exc).__name__}: {exc}"
            print(f"RapidOCR failed to load: {_ocr_error}", flush=True)
            raise
        finally:
            _ocr_loading = False


def save_upload_to_temp(upload):
    suffix = Path(upload.filename or "image.jpg").suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    upload.save(temp.name)
    temp.close()
    return temp.name


def _plain_list(value):
    if value is None:
        return []

    try:
        return value.tolist()
    except AttributeError:
        return list(value)


def _box_from_polygon(poly):
    if poly is None:
        return None

    try:
        pts = poly.tolist()
    except AttributeError:
        pts = poly

    try:
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        if not xs or not ys:
            return None

        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        return None


def _plain_box(box):
    if box is None:
        return None

    try:
        values = box.tolist()
    except AttributeError:
        values = box

    try:
        if len(values) >= 4 and not isinstance(values[0], (list, tuple)):
            return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]
    except Exception:
        pass

    return _box_from_polygon(values)


def _collect_lines(results, scale):
    """results: RapidOCR output, a list of [polygon, text, score] or None."""
    texts = []
    scores = []
    lines = []

    for item in results or []:
        try:
            polygon, text, score = item[0], item[1], item[2]
        except (TypeError, IndexError):
            continue

        text = str(text or "").strip()
        if not text:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        box = _box_from_polygon(polygon)

        # Map boxes back to the full-size image the browser holds.
        if box is not None and scale != 1.0:
            box = [v / scale for v in box]

        texts.append(text)
        scores.append(score)
        lines.append({"text": text, "score": score, "box": box})

    return texts, scores, lines


def _quality(texts, scores):
    # Rewards long, confidently read text. Used to pick the best rotation.
    return sum(len(t) * s for t, s in zip(texts, scores))


def _mostly_vertical(lines):
    boxes = [
        line["box"] for line in lines
        if line.get("box") and len(line.get("text", "")) >= 3
    ]

    if len(boxes) < 2:
        return False

    tall = sum(1 for b in boxes if (b[3] - b[1]) > 1.3 * (b[2] - b[0]))
    return tall >= len(boxes) * 0.6


def _predict(ocr, image):
    with _predict_lock:
        result, _elapse = ocr(image, use_cls=True, text_score=0.3)
        return result


def run_ocr(upload, try_rotations=False):
    # RapidOCR reads vertical label text directly, so rotation retries are off by
    # default; they triple OCR time on a small CPU.
    import cv2

    if not upload:
        raise ValueError("No image received")

    temp_path = save_upload_to_temp(upload)

    try:
        ocr = get_ocr()

        image = cv2.imread(temp_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read the uploaded image")

        height, width = image.shape[:2]
        scale = 1.0

        if max(height, width) > OCR_MAX_SIDE:
            scale = OCR_MAX_SIDE / float(max(height, width))
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        texts, scores, lines = _collect_lines(_predict(ocr, image), scale)
        rotation = 0

        # Vials and bottles often lie on their side, so the label text is vertical.
        # Re-read the image rotated and keep the orientation that reads best.
        if try_rotations and _mostly_vertical(lines):
            best = (_quality(texts, scores), 0, texts, scores, lines)

            for degrees, code in ((90, cv2.ROTATE_90_CLOCKWISE), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)):
                rotated = cv2.rotate(image, code)
                r_texts, r_scores, r_lines = _collect_lines(_predict(ocr, rotated), scale)
                quality = _quality(r_texts, r_scores)

                if quality > best[0] * 1.15:
                    best = (quality, degrees, r_texts, r_scores, r_lines)

                # Good enough: skip the second rotation to save time.
                if best[1] == degrees and r_scores and sum(r_scores) / len(r_scores) > 0.85:
                    break

            _, rotation, texts, scores, lines = best

        combined = " ".join(texts).strip()
        confidence = sum(scores) / len(scores) if scores else 0.0

        return {
            "text": combined,
            "confidence": confidence,
            "segments": texts,
            "lines": lines,
            "rotation_degrees": rotation,
        }

    finally:
        # Uploaded images exist only as temporary files during OCR.
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@app.before_request
def require_authentication():
    # Keep only the login route and a non-sensitive health endpoint public.
    if request.endpoint in {"login", "health"}:
        return None

    if not APP_PASSWORD:
        return CONFIG_HTML, 503

    if session.get("authenticated") is True:
        return None

    if request.path.startswith("/api/"):
        return jsonify({"error": "authentication_required"}), 401

    return redirect(url_for("login"))


@app.after_request
def add_security_headers(response):
    # Prevent sensitive OCR/API results from being cached by browsers/proxies.
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return CONFIG_HTML, 503

    error = None

    if request.method == "POST":
        candidate = request.form.get("password", "")

        if hmac.compare_digest(candidate, APP_PASSWORD):
            session.clear()
            session["authenticated"] = True
            session.permanent = True
            return redirect(url_for("home"))

        error = "סיסמה שגויה."

    return render_template_string(LOGIN_HTML, error=error)


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
def home():
    return send_file(BASE_DIR / "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "engine": "rapidocr",
        "vision_available": vision.vision_available(),
        "vision_model": vision.VISION_MODEL if vision.vision_available() else None,
        "ocr_loaded": _ocr is not None,
        "ocr_loading": _ocr_loading,
        "ocr_error": _ocr_error,
    })


@app.post("/api/ocr-warmup")
def ocr_warmup():
    try:
        get_ocr()
        return jsonify({"ok": True, "ocr_loaded": True})
    except Exception as exc:
        app.logger.exception("OCR warmup failed")
        return jsonify({
            "ok": False,
            "error": str(exc),
            "ocr_error": _ocr_error,
        }), 500


@app.post("/api/ocr-order")
def ocr_order():
    try:
        result = run_ocr(request.files.get("image"))
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Order OCR failed")
        return jsonify({"error": str(exc)}), 500


@app.post("/api/ocr-medication")
def ocr_medication():
    try:
        result = run_ocr(request.files.get("image"))
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Medication OCR failed")
        return jsonify({"error": str(exc)}), 500


def _vision_endpoint(reader):
    if not vision.vision_available():
        return jsonify({"error": "vision_disabled"}), 503

    upload = request.files.get("image")
    if not upload:
        return jsonify({"error": "No image received"}), 400

    data = upload.read()
    if not data:
        return jsonify({"error": "Empty image"}), 400
    if len(data) > 8 * 1024 * 1024:
        return jsonify({"error": "Image too large"}), 413

    media_type = upload.mimetype if upload.mimetype in ("image/jpeg", "image/png", "image/webp") else "image/jpeg"

    try:
        # The image is held in memory only for the duration of the request.
        return jsonify(reader(data, media_type))
    except Exception as exc:
        app.logger.exception("Vision request failed")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 502


@app.post("/api/vision-orders")
def vision_orders():
    return _vision_endpoint(vision.read_orders)


@app.post("/api/vision-packages")
def vision_packages():
    return _vision_endpoint(vision.read_packages)


if __name__ == "__main__":
    print("Starting MATAMED on port 8000")
    print("OCR models load on the first request (about one second).")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
