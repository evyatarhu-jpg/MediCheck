"""
Structured reading of medication orders and package labels with a vision model
(Claude, Anthropic API).

Enabled only when ANTHROPIC_API_KEY is set on the server. Images are sent to
the Anthropic API, so enabling it is an institutional data-governance decision.
If the key is missing or a call fails, the client falls back to local OCR.
"""

import base64
import json
import os

VISION_MODEL = os.environ.get("MEDICHECK_VISION_MODEL", "claude-sonnet-5-5")
VISION_TIMEOUT = float(os.environ.get("MEDICHECK_VISION_TIMEOUT", "90"))

_client = None


def vision_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(timeout=VISION_TIMEOUT, max_retries=1)
    return _client


ROUTES = ["PO", "IV", "IM", "SC", "NG", "PR", "INH", "SL", "TOPICAL", "EYE", "EAR", "NASAL", "OTHER", "UNKNOWN"]
DOSE_UNITS = ["mg", "mcg", "g", "ng", "units", "mEq", "mmol", "mL", "drops", "puffs", "other"]
FORMS = ["tablet", "capsule", "injectable_solution", "powder", "oral_liquid", "drops", "inhalation",
         "suppository", "topical", "other", "unknown"]

ORDERS_TOOL = {
    "name": "record_medication_orders",
    "description": "Record every medication order visible in the photographed order screen.",
    "input_schema": {
        "type": "object",
        "properties": {
            "patient_weight_kg": {
                "type": ["number", "null"],
                "description": "Patient weight in kg only if it is printed on the screen; otherwise null."
            },
            "orders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name_as_written": {"type": "string", "description": "Exactly as printed, including prefixes like GTTS. or I. and any '...' truncation."},
                        "generic_name": {"type": ["string", "null"], "description": "International generic (INN) name in English if you can identify it with certainty; null otherwise. Do not guess a truncated name unless only one drug fits."},
                        "name_truncated": {"type": "boolean"},
                        "route": {"type": "string", "enum": ROUTES, "description": "Route as stated or as implied by an explicit form prefix (I. = injection is NOT enough to choose IV vs IM: use UNKNOWN then)."},
                        "route_evidence": {"type": "string", "description": "The text that shows the route, or empty."},
                        "dose": {"type": ["number", "null"], "description": "Single dose amount as a number. Read decimal points with extreme care. null if not visible."},
                        "dose_unit": {"type": ["string", "null"], "enum": DOSE_UNITS + [None]},
                        "dose_as_written": {"type": "string", "description": "The dose text exactly as printed, e.g. '1.05 mg', '2-5 mcg/kg/h'."},
                        "weight_based_dose": {"type": ["string", "null"], "description": "e.g. '10 mg/kg/dose' if printed."},
                        "frequency": {"type": ["string", "null"], "description": "Normalized: q4h, q6h, q8h, q12h, q24h, x1/day, x2/day, x3/day, x4/day, STAT, every other day, weekly; or as written."},
                        "prn": {"type": "boolean"},
                        "continuous_infusion": {"type": "boolean", "description": "True for rates such as mcg/kg/min, mg/kg/h, mL/h or a continuous drip."},
                        "rate_as_written": {"type": ["string", "null"]},
                        "administration_times": {"type": ["string", "null"], "description": "Clock times if printed, comma separated."},
                        "duration": {"type": ["string", "null"], "description": "Infusion duration such as '30 min' if printed."},
                        "dosage_form": {"type": "string", "enum": FORMS},
                        "uncertain_fields": {"type": "array", "items": {"type": "string"}, "description": "Field names you could not read with certainty."},
                        "confidence": {"type": "number", "description": "0 to 1, overall confidence for this row."}
                    },
                    "required": ["name_as_written", "generic_name", "route", "dose", "dose_unit", "dose_as_written",
                                 "prn", "continuous_infusion", "dosage_form", "uncertain_fields", "confidence"]
                }
            },
            "unreadable_medication_rows": {
                "type": "integer",
                "description": "Number of rows that look like medications but could not be read."
            }
        },
        "required": ["orders", "patient_weight_kg"]
    }
}

ORDERS_PROMPT = """This is a photo of a hospital medication-order screen in a pediatric / pediatric cardiac ICU (Israel; the EMR mixes Hebrew and English).
List ONLY medication orders. Skip nursing tasks, assessments, positioning, eye/mouth care, labs, feeds, parenteral nutrition (TPN, SMOFlipid) and IV fluids without a drug.
Rules:
- Transcribe numbers exactly as printed. Decimal points matter: 1.05 is not 105, 2.1 is not 21. A range like '2-5' must stay a range: put it in dose_as_written, set dose to null and add 'dose' to uncertain_fields.
- Never invent a value. If something is not visible or not readable, use null or UNKNOWN and list it in uncertain_fields.
- Route: use the route only when the screen states it (IV, PO, SC, IM, NG...). A dosage-form prefix (GTTS., TAB., SYR., COLL., I.) describes the form; set dosage_form from it. Choose a route from a prefix only when it is unambiguous (TAB./CAP./SYR. -> PO, COLL. -> EYE); for 'I.' use UNKNOWN.
- Names cut with '...' on screen: set name_truncated true; fill generic_name only if exactly one known drug fits.
- Each order is one row. Do not move a dose from one row to another.
Call the tool once with all orders."""

PACKAGES_TOOL = {
    "name": "record_medication_packages",
    "description": "Record every medication package, vial, ampoule, bottle or box visible in the photo.",
    "input_schema": {
        "type": "object",
        "properties": {
            "packages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name_as_written": {"type": "string"},
                        "generic_name": {"type": ["string", "null"], "description": "Generic (INN) name in English; null if not certain."},
                        "brand_name": {"type": ["string", "null"]},
                        "dosage_form": {"type": "string", "enum": FORMS},
                        "strength_value": {"type": ["number", "null"], "description": "For tablets/capsules: amount per unit, e.g. 500 for 500 mg."},
                        "strength_unit": {"type": ["string", "null"], "enum": DOSE_UNITS + [None]},
                        "concentration_amount": {"type": ["number", "null"], "description": "For liquids: drug amount in the stated volume, e.g. 1 in '1 mg/0.1 mL'."},
                        "concentration_unit": {"type": ["string", "null"], "enum": DOSE_UNITS + [None]},
                        "concentration_volume_ml": {"type": ["number", "null"], "description": "The volume that holds concentration_amount, e.g. 0.1 in '1 mg/0.1 mL'; 1 for 'per mL'."},
                        "concentration_as_written": {"type": ["string", "null"]},
                        "total_amount_as_written": {"type": ["string", "null"], "description": "e.g. '500 mg/10 mL vial', '100 tablets'."},
                        "label_route": {"type": ["string", "null"], "description": "Route written on the label (IV, IM, oral, intraocular...)."},
                        "expiry": {"type": ["string", "null"], "description": "Expiry date as YYYY-MM or YYYY-MM-DD if visible."},
                        "lot": {"type": ["string", "null"]},
                        "bbox": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                            "description": "Approximate box of this package in the image as [x0, y0, x1, y1], each 0..1 relative to image width/height."
                        },
                        "uncertain_fields": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number"}
                    },
                    "required": ["name_as_written", "generic_name", "dosage_form", "strength_value", "strength_unit",
                                 "concentration_amount", "concentration_unit", "concentration_volume_ml",
                                 "uncertain_fields", "confidence"]
                }
            }
        },
        "required": ["packages"]
    }
}

PACKAGES_PROMPT = """This photo shows medication packages (vials, ampoules, bottles, boxes) prepared in a pediatric ICU.
For each package read the label: drug, dosage form, strength (solid forms) or concentration (liquids), label route, expiry and lot.
Rules:
- Read numbers and decimal points exactly. '1 mg/0.1 mL' -> amount 1, unit mg, volume 0.1.
- For '500 mg/10 mL' give amount 500, volume 10. For 'mg/mL' give volume 1. For a percentage (e.g. 1%) leave amount null, put the text in concentration_as_written and add 'concentration' to uncertain_fields.
- Powders for reconstitution: give the total amount in total_amount_as_written and leave concentration null unless the label states the concentration after reconstitution.
- Never invent values; use null and list the field in uncertain_fields.
Call the tool once with all packages."""


def _call(image_bytes, media_type, tool, prompt):
    client = _get_client()
    message = client.messages.create(
        model=VISION_MODEL,
        max_tokens=4000,
        temperature=0,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.standard_b64encode(image_bytes).decode("ascii"),
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    for block in message.content:
        if getattr(block, "type", "") == "tool_use":
            return block.input

    raise RuntimeError("The vision model returned no structured result")


def read_orders(image_bytes, media_type="image/jpeg"):
    return _call(image_bytes, media_type, ORDERS_TOOL, ORDERS_PROMPT)


def read_packages(image_bytes, media_type="image/jpeg"):
    return _call(image_bytes, media_type, PACKAGES_TOOL, PACKAGES_PROMPT)
