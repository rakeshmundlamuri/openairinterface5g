#!/usr/bin/env python3
"""End-to-end Deep-JSCC-over-real-PDSCH demo.

image -> encode_image.py -> custom_iq.txt -> send_recv_signal.send_recv() (real gNB/UE
rfsim, auto-discovers where the signal fits in the live PDSCH allocation) -> decode_image.py
-> reconstructed image + PSNR.

This script only handles the JSCC-specific parts (encode/decode); all RF transport -
launching gNB/UE, finding a placement that fits, retrying on fading-channel sync failure -
is scripts/custom_re/send_recv_signal.py's job, kept as a separate general-purpose tool
(see doc/CUSTOM_SIGNAL_AI.md) so it can be reused for payloads other than this image codec.

Usage:
    run_demo.py --nr-softmodem PATH --nr-uesoftmodem PATH --gnb-conf PATH \
                --checkpoint PATH [--image PATH | --cifar-index N] \
                [--channel-type TYPE] [--out-dir DIR]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "custom_re"))
from send_recv_signal import send_recv  # noqa: E402


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr-softmodem", required=True)
    ap.add_argument("--nr-uesoftmodem", required=True)
    ap.add_argument("--gnb-conf", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--cifar-index", type=int, default=0)
    ap.add_argument("--num-pilot", type=int, default=8)
    ap.add_argument("--channel-type", default=None, help="e.g. AWGN, Rayleigh8 (default: no channel model)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-retries", type=int, default=4, help="retries on RF sync failure - "
                     "see send_recv_signal.py")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="jscc_out_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir={out_dir}")

    # 1. Encode the image.
    run([sys.executable, str(SCRIPT_DIR / "encode_image.py"),
         "--checkpoint", args.checkpoint, "--out-dir", str(out_dir),
         *(["--image", args.image] if args.image else ["--cifar-index", str(args.cifar_index)]),
         "--num-pilot", str(args.num_pilot)])
    iq_file = out_dir / "custom_iq.txt"
    lines = iq_file.read_text().splitlines()
    iq_values = [complex(*map(float, l.split(","))) for l in lines if l.strip()]
    print(f"encoded payload: {len(iq_values)} REs total")

    # 2. Send it over the real gNB->rfsim->UE PDSCH pipeline and get back what was received.
    send_recv(iq_values, args.nr_softmodem, args.nr_uesoftmodem, args.gnb_conf,
              channel_type=args.channel_type, max_retries=args.max_retries, work_dir=out_dir)
    rx_m = out_dir / "custom_rx_iq.m"

    # 3. Decode the received capture back into an image.
    run([sys.executable, str(SCRIPT_DIR / "decode_image.py"),
         "--meta", str(out_dir / "meta.json"), "--rx-m", str(rx_m),
         "--out", str(out_dir / "reconstructed.png"), "--original", str(out_dir / "original.png")])

    print(f"\ndone. original: {out_dir / 'original.png'}  reconstructed: {out_dir / 'reconstructed.png'}")


if __name__ == "__main__":
    main()
