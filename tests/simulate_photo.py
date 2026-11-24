"""
Turn a clean screenshot into something like a phone photo of a monitor:
perspective tilt, slight rotation, moire from the pixel grid, glare, blur,
uneven light, sensor noise and JPEG compression.

  python3 tests/simulate_photo.py in.jpg out.jpg [--level 1|2|3] [--seed N]
"""
import argparse
import random

import numpy as np
from PIL import Image, ImageFilter


def _coeffs(src, dst):
    m = []
    for (x, y), (u, v) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(m, dtype=float)
    b = np.array(src, dtype=float).reshape(8)
    return np.linalg.solve(a, b)


def simulate(img, level=2, seed=0):
    rnd = random.Random(seed)
    img = img.convert("RGB")
    w, h = img.size
    s = {1: 0.03, 2: 0.06, 3: 0.10}[level]

    # Pad with a dark "monitor bezel / desk" so the warp has somewhere to go.
    pad = int(max(w, h) * 0.08)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (25, 25, 28))
    canvas.paste(img, (pad, pad))
    W, H = canvas.size

    # Moire: a fine interference pattern from the monitor pixel grid.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rnd.uniform(0.2, 0.6)
    freq = rnd.uniform(0.18, 0.32)
    moire = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * freq * 2 * np.pi * 0.25)
    arr = np.asarray(canvas).astype(np.float32)
    arr *= (1 + 0.05 * level * moire)[..., None]

    # Uneven light and a glare blob.
    gx, gy = rnd.uniform(0.2, 0.8) * W, rnd.uniform(0.2, 0.8) * H
    r = max(W, H) * rnd.uniform(0.15, 0.3)
    glare = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r * r)))
    arr = arr * (0.85 + 0.15 * (xx / W)[..., None]) + 70 * level * 0.4 * glare[..., None]
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Perspective tilt.
    def j(v):
        return v + rnd.uniform(-s, s) * max(W, H)
    src = [(0, 0), (W, 0), (W, H), (0, H)]
    dst = [(j(0), j(0)), (j(W), j(0)), (j(W), j(H)), (j(0), j(H))]
    canvas = canvas.transform((W, H), Image.PERSPECTIVE, _coeffs(src, dst), Image.BICUBIC, fillcolor=(20, 20, 20))
    canvas = canvas.rotate(rnd.uniform(-2.5, 2.5) * level / 2, resample=Image.BICUBIC, fillcolor=(20, 20, 20))

    # Phone resolution, slight defocus, noise, JPEG.
    target = 2400 if level < 3 else 1800
    k = target / max(W, H)
    canvas = canvas.resize((int(W * k), int(H * k)), Image.LANCZOS)
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.5 + 0.4 * level))
    arr = np.asarray(canvas).astype(np.float32)
    arr += np.random.default_rng(seed).normal(0, 3 + 2 * level, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    simulate(Image.open(a.src), a.level, a.seed).save(a.dst, quality=80)
