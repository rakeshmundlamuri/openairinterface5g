#!/usr/bin/env python3
"""Decode the UE's post-equalization capture back into an image.

Mirrors scripts/custom_re/compare_iq.py's approach but calibrates the complex
scale/rotation from the known PILOT REs only (not the payload) - this is genuine
reference-based calibration a real receiver could do, not "cheating" with the secret
image data.

Usage:
    decode_image.py --meta META_JSON --rx-m CUSTOM_RX_IQ_M --out RECONSTRUCTED_PNG
"""
import argparse
import json
import re
from pathlib import Path

import torch
from torchvision.utils import save_image

from model import Decoder
from pilot import pilot_symbols

_LINE_RE = re.compile(r"(-?\d+)\s*\+\s*j\*\((-?\d+)\)")


def parse_m_file(path):
    values = []
    with open(path) as f:
        for line in f:
            m = _LINE_RE.search(line)
            if m:
                values.append(complex(int(m.group(1)), int(m.group(2))))
    return values


def fit_scale(rx, ref):
    num = sum(r * t.conjugate() for r, t in zip(rx, ref))
    den = sum(abs(t) ** 2 for t in ref)
    return num / den if den else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--rx-m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--original", default=None, help="original.png, for reporting PSNR (evaluation only)")
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    received = parse_m_file(args.rx_m)
    if len(received) != meta["num_re"]:
        raise SystemExit(f"expected {meta['num_re']} REs, got {len(received)} from {args.rx_m}")

    num_pilot = meta["num_pilot"]
    rx_pilot = received[:num_pilot]
    rx_payload = received[num_pilot:]

    ref_pilot = pilot_symbols(num_pilot)
    a = fit_scale(rx_pilot, ref_pilot)
    if a == 0:
        raise SystemExit("pilot calibration failed (zero scale) - check the capture")

    descaled = [r / a for r in rx_payload]  # back to peak-normalized [-1,1]-ish units
    payload = [v * meta["peak_scale"] for v in descaled]  # undo peak-normalization

    c = meta["c"]
    h, w = meta["spatial"]
    real = torch.tensor([v.real for v in payload], dtype=torch.float32).reshape(c, h, w)
    imag = torch.tensor([v.imag for v in payload], dtype=torch.float32).reshape(c, h, w)
    z = torch.cat([real, imag], dim=0).unsqueeze(0)  # (1, 2c, h, w)

    ckpt = torch.load(meta["checkpoint"], map_location="cpu")
    decoder = Decoder(c)
    state = {k[len("decoder."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("decoder.")}
    decoder.load_state_dict(state)
    decoder.eval()

    with torch.no_grad():
        recon = decoder(z)[0]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(recon, args.out)
    print(f"reconstructed image saved to {args.out}")
    print(f"pilot-fitted calibration scale a={a:.6g}")

    if args.original:
        from PIL import Image
        from torchvision import transforms
        orig = transforms.ToTensor()(Image.open(args.original).convert("RGB"))
        mse = torch.mean((recon - orig) ** 2).item()
        psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item() if mse > 0 else float("inf")
        print(f"PSNR vs original: {psnr:.2f} dB (mse={mse:.6f}) - evaluation only, not available to a real receiver")


if __name__ == "__main__":
    main()
