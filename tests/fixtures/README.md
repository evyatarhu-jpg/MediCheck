# MediCheck test fixtures

Synthetic, fictional images with known answers. Fictional patient and manufacturer.
Doses are test values only and are not clinical guidance.

- `orders/*.jpg` + `.json`: EMR-style order screens photographed off a monitor
- `packages/*.jpg` + `.json`: vials, ampoules, tablet boxes and oral bottles

Regenerate with `python3 tests/generate_fixtures.py`; score with `python3 tests/eval.py`.
