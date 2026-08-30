#!/usr/bin/env python3
"""Compare a captured IQ dump (gNB's custom_tx_iq.m or UE's custom_rx_iq.m) against
the original custom_iq.txt file the values were generated from.

custom_iq.txt holds one "re,im" pair per line, floats in [-1,1].
The .m dump is produced by LOG_M() with format=1 (complex 16-bit), i.e. lines of
"<varname> = [" followed by one "<re> + j*(<im>)" per RE and a closing "];".

The dump is scaled by the gNB's TX amplitude and rotated by rfsim/the per-symbol
phase correction relative to custom_iq.txt. A single complex scalar (least-squares
best fit) is used to remove that scale/rotation, so the dumped values can be shown
and compared directly against custom_iq.txt in the same units.

Usage:
    compare_iq.py <custom_iq.txt> <captured .m file> [tolerance_fraction]
"""
import re
import sys

LINE_RE = re.compile(r"(-?\d+)\s*\+\s*j\*\((-?\d+)\)")


def parse_iq_txt(path):
    values = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            re_str, im_str = line.split(",")
            values.append(complex(float(re_str), float(im_str)))
    return values


def parse_m_file(path):
    values = []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                values.append(complex(int(m.group(1)), int(m.group(2))))
    return values


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <custom_iq.txt> <captured .m file> [tolerance_fraction]", file=sys.stderr)
        return 1

    original = parse_iq_txt(sys.argv[1])
    captured = parse_m_file(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    if len(original) != len(captured):
        print(f"FAIL: custom_iq.txt has {len(original)} REs, captured file has {len(captured)} REs")
        return 1
    if len(original) == 0:
        print("FAIL: no REs parsed from input files")
        return 1

    # Best-fit complex scalar a minimizing sum |captured - a*original|^2 (the TX
    # amplitude scaling plus any channel/rotation applied along the way). Dividing
    # captured by a removes that scale, bringing it back to custom_iq.txt's units.
    num = sum(c * o.conjugate() for c, o in zip(captured, original))
    den = sum(abs(o) ** 2 for o in original)
    a = num / den if den else 0

    max_rel_err = 0.0
    for i, (o, c) in enumerate(zip(original, captured)):
        descaled = c / a if a else complex(0, 0)
        err = abs(descaled - o)
        ref = abs(o) or 1.0
        rel_err = err / ref
        max_rel_err = max(max_rel_err, rel_err)
        print(f"RE {i}: original={o:.4f} captured={c} descaled={descaled:.4f} rel_err={rel_err:.3f}")

    print(f"\nscale removed a={a:.6g}, max relative error={max_rel_err:.3f} (tolerance={tol})")
    if max_rel_err <= tol:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
