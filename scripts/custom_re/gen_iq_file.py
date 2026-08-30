#!/usr/bin/env python3
"""Generate an IQ file for the OAI custom-RE test signal.

Writes one "re,im" pair per line (floats in [-1,1]), which
nr_custom_signal_load_file() reads and scales by the gNB's TX_AMP.

Usage:
    gen_iq_file.py <output_file> [num_re]
"""
import cmath
import sys

QPSK_SYMBOLS = [cmath.rect(1.0, cmath.pi / 4 + k * cmath.pi / 2) for k in range(4)]


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <output_file> [num_re]", file=sys.stderr)
        return 1
    out_path = sys.argv[1]
    num_re = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    with open(out_path, "w") as f:
        for i in range(num_re):
            s = QPSK_SYMBOLS[i % len(QPSK_SYMBOLS)]
            f.write(f"{s.real:.6f},{s.imag:.6f}\n")

    print(f"wrote {num_re} IQ values to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
