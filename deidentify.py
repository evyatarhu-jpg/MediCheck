"""
Black out patient identifiers in an image before it leaves the server.

Runs the local OCR, finds rows that carry identifiers, and paints a solid
black band across the full width of each such row. The original image is
never sent anywhere; only the masked copy goes to the vision model.

What is masked
  * any row with an identifier keyword (patient, name, ID, MRN, date of
    birth, case/admission number, phone, address) in Hebrew or English
  * any row with a run of 7 or more digits (ID, MRN, phone, case numbers)
  * on order screens, everything above the first row that holds a dose:
    the patient banner sits there, and names cannot be recognised reliably

Fail closed: if the image cannot be read or OCR finds no text, the caller
gets an error and must not send the image.
"""

import re

import numpy as np

# Keywords are matched against OCR text with spaces and punctuation removed,
# so "ת . ז" and "ת״ז" both become "תז".
_KEYWORDS_HE = [
    "מטופל", "מטופלת", "שםפרטי", "שםמשפחה", "שםהחולה", "שם:", "תז", "תעודתזהות",
    "מספרזהות", "מסזהות", "דרכון", "תאריךלידה", "תלידה", "גיל:", "מספרמקרה",
    "מסמקרה", "מספראשפוז", "מסאשפוז", "מספרתיק", "מסתיק", "טלפון", "נייד",
    "כתובת", "קופתחולים",
]
_KEYWORDS_EN = [
    "patient", "name", "mrn", "dob", "dateofbirth", "birth", "idno", "id:",
    "passport", "admission", "caseno", "encounter", "phone", "address",
]
_DIGIT_RUN = re.compile(r"\d{7,}")
# OCR often reads 0 as O/Q/D and 1 as I/l in ID numbers.
_LOOKALIKE = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "|": "1"})
_DOSE = re.compile(r"\d\s*(mcg|µg|μg|ug|mg|g|ml|units?|iu|meq|mmol)\b", re.I)
# Anything that marks a medication row: a dose, a frequency ("6 h", "q8h") or a form prefix ("I.", "TAB.").
_ROW = re.compile(r"\d\s*(mcg|µg|μg|ug|mg|g|ml|units?|iu|meq|mmol)\b|^\s*q?\d{1,2}\s*h\b|"
                  r"^\s*(i|tab|cap|syr|susp|inj|amp|vial|sol|drops?|inh|neb|supp)\s*\.", re.I)
# Drug names on the order screen are upper-case Latin; patient details are Hebrew.
_DRUG_WORD = re.compile(r"[A-Z]{5,}")
# Column headers of the order table.
_HEADER = ("פרטהוראה", "כמות", "ערכים", "מתן", "תדירות", "תדירו")


def _starts_table(text):
    if _ROW.search(text) or _DRUG_WORD.search(text):
        return True
    sq = _squash(text)
    return any(h in sq for h in _HEADER)


def _squash(text):
    t = str(text or "").lower()
    t = t.replace("״", "").replace('"', "").replace("'", "").replace("׳", "")
    return re.sub(r"[\s.\-_/\\|·•,]+", "", t)


def is_identifier_text(text):
    raw = str(text or "")
    digits = re.sub(r"[\s\-./]", "", raw).translate(_LOOKALIKE)
    if _DIGIT_RUN.search(digits):
        return True
    s = _squash(raw)
    if any(k in s for k in _KEYWORDS_HE):
        return True
    # English keywords only as whole words, so "Amikacin" etc. never trip "name".
    words = re.findall(r"[a-z]+:?", raw.lower())
    bare = [w.rstrip(":") for w in words]
    for k in _KEYWORDS_EN:
        if k.endswith(":"):
            if k in words:
                return True
        elif k in bare or (len(k) > 4 and k in s):
            return True
    return False


def _quad(line):
    """Four corners (tl, tr, br, bl) of an OCR line, from its polygon or its box."""
    poly = line.get("poly")
    if poly is not None and len(poly) == 4:
        pts = sorted(((float(x), float(y)) for x, y in poly), key=lambda p: p[0])
        left = sorted(pts[:2], key=lambda p: p[1])
        right = sorted(pts[2:], key=lambda p: p[1])
        return left[0], right[0], right[1], left[1]
    x0, y0, x1, y1 = line["box"]
    return (x0, y0), (x1, y0), (x1, y1), (x0, y1)


def _slope(q):
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = q
    dx = ((trx - tlx) + (brx - blx)) / 2.0
    if dx <= 1:
        return 0.0
    return ((try_ - tly) + (bry - bly)) / 2.0 / dx


def find_mask_polygons(lines, width, height, orders_screen):
    """Polygons to fill black, in full-image coordinates. Each identifier row is
    masked as a slanted band that follows the row across the whole width."""
    rows = [l for l in lines if l.get("box")]
    heights = sorted(abs(_quad(l)[3][1] - _quad(l)[0][1]) for l in rows) or [20]
    line_h = max(8.0, heights[len(heights) // 2])
    pad = max(4.0, line_h * 0.3)
    slopes = sorted(_slope(_quad(l)) for l in rows if len(l.get("text", "")) >= 4) or [0.0]
    tilt = slopes[len(slopes) // 2]
    polys = []

    for line in rows:
        if not is_identifier_text(line["text"]):
            continue
        tl, tr, br, bl = _quad(line)
        k = _slope((tl, tr, br, bl))
        # Top and bottom edges extended to x = 0 and x = width.
        top0 = tl[1] - k * tl[0] - pad
        bot0 = bl[1] - k * bl[0] + pad
        polys.append([(0, top0), (width, top0 + k * width), (width, bot0 + k * width), (0, bot0)])

    if orders_screen:
        firsts = [l for l in rows if _starts_table(l["text"]) and not is_identifier_text(l["text"])]
        if firsts:
            # Everything above the table (its header, or the first row that looks like
            # a medication), along the page tilt, with a row of margin. The patient
            # banner lives up there.
            def edge(l):
                tl = _quad(l)[0]
                return tl[1] - tilt * tl[0]
            c = min(edge(l) for l in firsts) - line_h
            polys.append([(0, -1), (width, -1), (width, c + tilt * width), (0, c)])

    return polys


def mask_image(image_bytes, ocr_lines_fn, orders_screen=True, max_side=1280):
    """
    image_bytes: the uploaded JPEG/PNG.
    ocr_lines_fn(bgr_image) -> list of {"text", "box"} in that image's coordinates.
    Returns (masked_jpeg_bytes, info). Raises ValueError when it cannot vouch
    for the result, so the caller does not send the image.
    """
    import cv2

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("deidentify: could not read image")

    h, w = image.shape[:2]
    scale = 1.0
    small = image
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    lines = ocr_lines_fn(small) or []
    if not lines:
        raise ValueError("deidentify: no text found, image not sent")

    full = []
    for l in lines:
        box = l.get("box")
        poly = l.get("poly")
        if scale != 1.0:
            box = [v / scale for v in box] if box is not None else None
            poly = [(x / scale, y / scale) for x, y in poly] if poly is not None else None
        full.append({"text": l.get("text", ""), "box": box, "poly": poly})

    polys = find_mask_polygons(full, w, h, orders_screen)
    for p in polys:
        cv2.fillPoly(image, [np.array(p, dtype=np.int32)], (0, 0, 0))
    masked_fraction = float((image.max(axis=2) == 0).mean())

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("deidentify: could not encode masked image")

    return buf.tobytes(), {
        "masked_regions": len(polys),
        "masked_fraction": round(masked_fraction, 3),
    }
