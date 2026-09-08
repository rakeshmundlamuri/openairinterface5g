"""Deterministic pilot pattern, known to both TX and RX without any side channel.

Same fixed QPSK-like repeating pattern as scripts/custom_re/gen_iq_file.py, factored
out here so encode_image.py (which writes it) and decode_image.py (which uses it as a
calibration reference) can't drift apart.
"""
import cmath

_QPSK_SYMBOLS = [cmath.rect(1.0, cmath.pi / 4 + k * cmath.pi / 2) for k in range(4)]


def pilot_symbols(n):
    return [_QPSK_SYMBOLS[i % len(_QPSK_SYMBOLS)] for i in range(n)]
