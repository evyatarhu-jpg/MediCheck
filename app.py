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
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from paddleocr import PaddleOCR


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
  <title>MediCheck – כניסה</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #f4f7fb;
      color: #123a63;
      font-family: Arial, Helvetica, sans-serif;
      padding: 20px;
    }
    .login-card {
      width: min(430px, 100%);
      background: #fff;
      border: 1px solid #d8e1ea;
      border-radius: 20px;
      box-shadow: 0 14px 40px rgba(21, 45, 74, .10);
      padding: 28px;
    }
    h1 { margin: 0 0 8px; font-size: 32px; }
    .subtitle { color: #667085; margin-bottom: 24px; line-height: 1.5; }
    label { display: block; font-weight: 800; margin-bottom: 8px; }
    input {
      width: 100%;
      min-height: 54px;
      border: 1px solid #b9c7d6;
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 19px;
      direction: ltr;
    }
    button {
      width: 100%;
      min-height: 56px;
      margin-top: 14px;
      border: 0;
      border-radius: 12px;
      background: #0d6efd;
      color: #fff;
      font-size: 19px;
      font-weight: 800;
      cursor: pointer;
    }
    .error {
      margin-top: 14px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #fff1f1;
      color: #a42323;
      border: 1px solid #efc2c2;
    }
    .prototype {
      margin-top: 18px;
      color: #7a5b12;
      background: #fff8df;
      border: 1px solid #ead79c;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <main class="login-card">
    <h1>MediCheck</h1>
    <div class="subtitle">גישה למערכת מוגנת בסיסמה.</div>

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

    <div class="prototype">
      Prototype בלבד. אין להשתמש במערכת לצורך החלטה קלינית או מתן תרופה ללא אימות אנושי מלא.
    </div>
  </main>
</body>
</html>
"""


CONFIG_HTML = """
<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MediCheck – הגדרה נדרשת</title>
</head>
<body style="font-family:Arial,sans-serif;max-width:760px;margin:50px auto;padding:20px;direction:rtl">
  <h1>MediCheck אינו מוגדר עדיין</h1>
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
            print("Loading PaddleOCR mobile models...", flush=True)
            _ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine="onnxruntime",
            )
            print("PaddleOCR ready.", flush=True)
            return _ocr
        except Exception as exc:
            _ocr_error = f"{type(exc).__name__}: {exc}"
            print(f"PaddleOCR failed to load: {_ocr_error}", flush=True)
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
    texts = []
    scores = []
    lines = []

    for result in results:
        payload = result.json
        res = payload.get("res", payload)

        rec_texts = _plain_list(res.get("rec_texts"))
        rec_scores = _plain_list(res.get("rec_scores"))
        rec_boxes = _plain_list(res.get("rec_boxes"))
        dt_polys = _plain_list(res.get("dt_polys"))

        for i, text in enumerate(rec_texts):
            if text is None:
                continue

            text = str(text).strip()
            if not text:
                continue

            score = 0.0

            if i < len(rec_scores):
                try:
                    score = float(rec_scores[i])
                except Exception:
                    score = 0.0

            box = None

            if i < len(rec_boxes):
                box = _plain_box(rec_boxes[i])

            if box is None and i < len(dt_polys):
                box = _box_from_polygon(dt_polys[i])

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
        return ocr.predict(image, text_rec_score_thresh=0.10)


def run_ocr(upload, try_rotations=False):
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
        result = run_ocr(request.files.get("image"), try_rotations=True)
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Medication OCR failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("Starting MediCheck on port 8000")
    print("The first OCR request downloads/loads the PaddleOCR models and may take longer.")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
