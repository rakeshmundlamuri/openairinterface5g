#!/usr/bin/env python3
"""Encode an image to a custom-RE IQ file for injection into real PDSCH data REs.

Runs the DeepJSCC encoder on a 32x32x3 image, flattens the (2c,8,8) output into c*64
complex channel symbols, peak-normalizes to fit our transport's assumed [-1,1] range,
prepends a small deterministic pilot prefix (see pilot.py) for receiver-side
calibration, and writes:
  - <out-dir>/custom_iq.txt : "re,im" per line, num_pilot + c*64 lines total -
    the exact format nr_custom_signal_load_file() reads.
  - <out-dir>/meta.json     : everything decode_image.py needs to invert this.
  - <out-dir>/original.png  : the 32x32 input image, for later PSNR comparison.

Usage:
    encode_image.py --checkpoint CKPT --out-dir DIR [--image PATH | --cifar-index N] [--num-pilot N]
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from model import Encoder
from pilot import pilot_symbols

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "cifar10_c8_awgn10.pth"


def load_image(image_path, cifar_index):
    if image_path is not None:
        img = Image.open(image_path).convert("RGB").resize((32, 32))
        return transforms.ToTensor()(img)
    from cifar_dataset import HFCifar10
    ds = HFCifar10(root=SCRIPT_DIR / "data", split="test", transform=transforms.ToTensor())
    img, _label = ds[cifar_index]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.exists() else None,
                     help=f"default: {DEFAULT_CHECKPOINT} if it exists (train one with train.py otherwise)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image", default=None, help="path to an image file (resized to 32x32)")
    ap.add_argument("--cifar-index", type=int, default=0, help="CIFAR-10 test-set index if --image not given")
    ap.add_argument("--num-pilot", type=int, default=8)
    args = ap.parse_args()
    if args.checkpoint is None:
        ap.error(f"no --checkpoint given and no default checkpoint found at {DEFAULT_CHECKPOINT} - "
                 f"train one first with train.py, or pass --checkpoint explicitly")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    c = ckpt["c"]
    encoder = Encoder(c)
    state = {k[len("encoder."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(state)
    encoder.eval()

    img = load_image(args.image, args.cifar_index)
    save_image(img, out_dir / "original.png")

    with torch.no_grad():
        z = encoder(img.unsqueeze(0))  # (1, 2c, 8, 8)

    spatial = z.shape[-2:]
    real = z[0, :c].flatten()
    imag = z[0, c:].flatten()
    payload = torch.complex(real, imag)  # (c*H*W,) complex

    peak_scale = payload.abs().max().item()
    payload_norm = payload / peak_scale

    pilots = pilot_symbols(args.num_pilot)
    all_symbols = pilots + [complex(v.real, v.imag) for v in payload_norm]

    iq_path = out_dir / "custom_iq.txt"
    with open(iq_path, "w") as f:
        for s in all_symbols:
            f.write(f"{s.real:.6f},{s.imag:.6f}\n")

    meta = {
        "c": c,
        "spatial": list(spatial),
        "num_pilot": args.num_pilot,
        "num_payload": payload.numel(),
        "num_re": len(all_symbols),
        "peak_scale": peak_scale,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(all_symbols)} REs ({args.num_pilot} pilot + {payload.numel()} payload) to {iq_path}")
    print(f"meta: {meta}")


if __name__ == "__main__":
    main()
