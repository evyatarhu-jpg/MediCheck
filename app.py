import os
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# PaddleOCR can use Hugging Face by default. If that source is unavailable,
# uncomment the next line to use Paddle's BOS model source instead:
# os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"

from paddleocr import PaddleOCR


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

_ocr = None
_ocr_lock = threading.Lock()


def get_ocr():
    global _ocr

    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                print("Loading PaddleOCR models for the first time...")
                _ocr = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    engine="onnxruntime",
                )
                print("PaddleOCR ready.")

    return _ocr


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


def run_ocr(upload):
    if not upload:
        raise ValueError("No image received")

    temp_path = save_upload_to_temp(upload)

    try:
        ocr = get_ocr()

        results = ocr.predict(
            temp_path,
            text_rec_score_thresh=0.10,
        )

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

                texts.append(text)
                scores.append(score)

                lines.append({
                    "text": text,
                    "score": score,
                    "box": box,
                })

        combined = " ".join(texts).strip()

        confidence = (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        return {
            "text": combined,
            "confidence": confidence,
            "segments": texts,
            "lines": lines,
        }

    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "ocr_loaded": _ocr is not None,
    })


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


if __name__ == "__main__":
    print("Starting MediCheck on port 8000")
    print("The first OCR request downloads/loads the PaddleOCR models and may take longer.")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
