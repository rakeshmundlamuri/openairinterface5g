#!/usr/bin/env python3
"""Reproduce THE-TRAIN-LAB/Semantic-QAM's own SNR-sweep analysis
(evaluate_and_report_snr_sweep() / measure_semantic_quality() in its notebook), but
driven by the real gNB->rfsim->UE PDSCH pipeline instead of a Sionna-simulated channel,
and comparing its two exported constellations (criticality-aware "semantic" vs
Gray-coded "standard") against each other at each noise level instead of sweeping its
adaptive-K agent.

Caveat carried over honestly from run_demo.py: the x-axis here is rfsim's channel-model
noise_power_dB knob (see scripts/custom_re/send_recv_signal.py), not a calibrated Eb/No -
it is a real, physical channel severity control, but the numeric value is not directly
comparable to the notebook's Sionna ebnodb2no()-based "SNR (dB)".

For each (noise level, scheme, image) this sends that image's already-exported QAM
symbols over the real pipeline, demodulates/reconstructs, and records: symbol error
rate, bit error rate (ber), PSNR, and the notebook's task-based semantic_quality /
label_preservation (classification accuracy) - then averages over images and plots
both schemes against noise level side by side, plus a text summary table.

Usage:
    snr_sweep.py --nr-softmodem PATH --nr-uesoftmodem PATH --gnb-conf PATH \
                 [--qam-order {4,16,64,256,1024}] [--image-indices 0,1,2|all] \
                 [--noise-db-values -10,-5,0,5,10,15,20] [--channel-type AWGN] \
                 [--out-dir DIR]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fetch_bundle import fetch_qam_bundle
from run_demo import evaluate_image_batch, load_scheme_resources

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "custom_re"))
from send_recv_signal import find_placement  # noqa: E402

DEFAULT_CACHE_DIR = SCRIPT_DIR / "data"
SCHEMES = ["semantic", "standard"]
SCHEME_STYLE = {"semantic": dict(color="tab:purple", marker="o", label="semantic (criticality-aware)"),
                "standard": dict(color="tab:orange", marker="s", label="standard (Gray-coded)")}


def parse_list(s, cast):
    return [cast(x) for x in s.split(",") if x.strip() != ""]


def discover_shared_placement(nr_softmodem, gnb_conf, num_re, work_dir):
    """All (scheme, noise level, batch) combinations send the same number of REs per
    launch (batch_size * (num_pilot + payload length), the largest batch used) -
    discover the placement once, sized for the largest batch, and reuse it for the
    whole sweep (smaller trailing batches, using fewer REs at the same start_sc, are
    still valid within an already-validated larger range). Content doesn't matter for
    discovery, only the count, so a throwaway QPSK pattern is fine."""
    print(f"discovering a shared placement for {num_re} REs (reused across the whole sweep)...")
    probe_dir = work_dir / "placement_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    dummy_iq_file = probe_dir / "dummy_iq.txt"
    qpsk = [complex(1, 1) / 2 ** 0.5, complex(-1, 1) / 2 ** 0.5, complex(-1, -1) / 2 ** 0.5, complex(1, -1) / 2 ** 0.5]
    with open(dummy_iq_file, "w") as f:
        for i in range(num_re):
            v = qpsk[i % len(qpsk)]
            f.write(f"{v.real:.6f},{v.imag:.6f}\n")
    return find_placement(Path(nr_softmodem).resolve(), Path(gnb_conf).resolve(), [],
                           dummy_iq_file, num_re, probe_dir)


def run_sweep(qam_dir, classifier_path, image_indices, noise_db_values, num_pilot,
              nr_softmodem, nr_uesoftmodem, gnb_conf, channel_type, max_retries, out_dir,
              extra_images=0, mnist_root=None, batch_size=8):
    resources = {scheme: load_scheme_resources(scheme, qam_dir, classifier_path,
                                                extra_images=extra_images, mnist_root=mnist_root)
                 for scheme in SCHEMES}
    cfg = resources["semantic"]["cfg"]
    k = int(resources["semantic"]["data"]["k"])

    sample_len = len(resources["semantic"]["data"]["semantic_qam_symbols_iq"][image_indices[0]])
    chunks = [image_indices[i:i + batch_size] for i in range(0, len(image_indices), batch_size)]
    max_batch_re = max(len(c) for c in chunks) * (num_pilot + sample_len)
    placement = discover_shared_placement(nr_softmodem, gnb_conf, max_batch_re, out_dir)

    all_results = []
    total_launches = len(noise_db_values) * len(SCHEMES) * len(chunks)
    done = 0
    for noise_db in noise_db_values:
        for scheme in SCHEMES:
            per_image = []
            for chunk in chunks:
                done += 1
                print(f"\n[launch {done}/{total_launches}] noise_power_dB={noise_db} scheme={scheme} "
                      f"images={list(chunk)} ({len(chunk)} in one real rfsim send)")
                per_image.extend(evaluate_image_batch(
                    resources[scheme], chunk, num_pilot, nr_softmodem, nr_uesoftmodem, gnb_conf,
                    channel_type, max_retries, out_dir / "runs", noise_power_db=noise_db, placement=placement))

            agg = {
                "noise_power_db": noise_db,
                "scheme": scheme,
                "symbol_error_rate": float(np.mean([r["symbol_error_rate"] for r in per_image])),
                "ber": float(np.mean([r["ber"] for r in per_image])),
                "psnr": float(np.mean([r["psnr"] for r in per_image])),
                "avg_k": k,
            }
            if "semantic_quality" in per_image[0]:
                for key in ("semantic_quality", "label_preservation", "confidence_preservation",
                            "distribution_similarity"):
                    agg[key] = float(np.mean([r[key] for r in per_image]))
            all_results.append(agg)
            print(f"  -> avg over {len(image_indices)} image(s): "
                  f"SER={agg['symbol_error_rate']:.3f} BER={agg['ber']:.4f} PSNR={agg['psnr']:.2f}dB"
                  + (f" label_preservation={agg['label_preservation']:.3f}"
                     f" semantic_quality={agg['semantic_quality']:.3f}" if "semantic_quality" in agg else ""))
    return all_results, cfg


def plot_sweep(all_results, cfg, out_path):
    """2x3 grid: two lines per subplot (one per scheme) vs noise_power_dB - the same
    layout as THE-TRAIN-LAB/Semantic-QAM's evaluate_and_report_snr_sweep(), adapted to
    compare the two constellations instead of sweeping one system's adaptive K."""
    M = cfg["qam_order"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    def series(scheme, key):
        rows = sorted([r for r in all_results if r["scheme"] == scheme], key=lambda r: r["noise_power_db"])
        return np.array([r["noise_power_db"] for r in rows]), np.array([r[key] for r in rows])

    panels = [
        (axes[0, 0], "label_preservation", "Label Preservation vs Noise", "Accuracy [0, 1]", False, [0, 1.05]),
        (axes[0, 1], "semantic_quality", "Semantic Quality vs Noise", "Quality Score [0, 1]", False, [0, 1.05]),
        (axes[0, 2], "psnr", "Reconstruction PSNR vs Noise", "PSNR (dB)", False, None),
        (axes[1, 0], "ber", f"BER vs Noise ({M}-QAM)", "BER (log scale)", True, None),
        (axes[1, 1], "symbol_error_rate", f"Symbol Error Rate vs Noise ({M}-QAM)", "SER", False, [0, 1.05]),
    ]
    for ax, key, title, ylabel, is_log, ylim in panels:
        if key not in all_results[0]:
            ax.text(0.5, 0.5, "classifier not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontweight="bold")
            continue
        for scheme in SCHEMES:
            x, y = series(scheme, key)
            style = SCHEME_STYLE[scheme]
            if is_log:
                ax.semilogy(x, np.maximum(y, 1e-7), f"{style['marker']}-", color=style["color"],
                            linewidth=2, markersize=6, label=style["label"])
            else:
                ax.plot(x, y, f"{style['marker']}-", color=style["color"], linewidth=2, markersize=6,
                        label=style["label"])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("rfsim channel noise_power_dB")
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=8)

    axes[1, 2].axis("off")
    fixed_k = all_results[0]["avg_k"]
    bits_per_symbol = int(np.log2(M))
    axes[1, 2].text(0.05, 0.7, f"Fixed configuration:\n{M}-QAM ({bits_per_symbol} bits/symbol)\n"
                                f"k={fixed_k} concepts (no adaptive-K agent)\n"
                                f"{fixed_k} symbols/image, both schemes",
                     transform=axes[1, 2].transAxes, fontsize=10,
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    fig.suptitle(f"Criticality-aware vs standard {M}-QAM over real rfsim", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")


def print_summary_table(all_results, cfg):
    M = cfg["qam_order"]
    has_quality = "semantic_quality" in all_results[0]
    print("\n" + "=" * 105)
    print(f"UNIFIED PERFORMANCE SUMMARY TABLE ({M}-QAM, semantic vs standard, real rfsim)")
    print("=" * 105)
    headers = ["Scheme", "Noise (dB)", "SER", "BER"]
    if has_quality:
        headers += ["Label Pres.", "Sem. Quality"]
    headers += ["PSNR (dB)"]

    rows = sorted(all_results, key=lambda r: (r["scheme"], r["noise_power_db"]))
    table = []
    for r in rows:
        row = [r["scheme"], f"{r['noise_power_db']:.1f}", f"{r['symbol_error_rate']:.4f}", f"{r['ber']:.4f}"]
        if has_quality:
            row += [f"{r['label_preservation']:.4f}", f"{r['semantic_quality']:.4f}"]
        row += [f"{r['psnr']:.2f}"]
        table.append(row)

    col_widths = [max(len(h), max((len(row[i]) for row in table), default=0)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    for row in table:
        print(fmt.format(*row))
    print("=" * 105 + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nr-softmodem", required=True)
    ap.add_argument("--nr-uesoftmodem", required=True)
    ap.add_argument("--gnb-conf", required=True)
    ap.add_argument("--qam-order", type=int, default=64, choices=[4, 16, 64, 256, 1024])
    ap.add_argument("--image-indices", default="0,1,2",
                     help="comma-separated indices, or 'all' for the full 10-image set plus any "
                          "--extra-images (default: 0,1,2)")
    ap.add_argument("--extra-images", type=int, default=0,
                     help="generate this many additional MNIST images beyond OAI_Demo's fixed 10 "
                          "(indices 10, 11, ... - see extra_images.py). Default 0.")
    ap.add_argument("--mnist-root", default=None, help="cache dir for the MNIST download (default: data/mnist_raw)")
    ap.add_argument("--noise-db-values", default="-10,-5,0,5,10,15,20",
                     help="rfsim channelmod noise_power_dB values to sweep (comma-separated)")
    ap.add_argument("--channel-type", default="AWGN", help="rfsim channel model type (default: AWGN)")
    ap.add_argument("--num-pilot", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8,
                     help="images per real rfsim launch (batched into one send/receive - see "
                          "send_batch_over_rfsim in run_demo.py). Default 8, the most this PDSCH "
                          "config's single-symbol allocation (600 REs) fits at num_pilot=8/k=64 "
                          "(8*(8+64)=576). Lower it if a different --gnb-conf has less room.")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    image_indices = (list(range(10 + args.extra_images)) if args.image_indices == "all"
                     else parse_list(args.image_indices, int))
    noise_db_values = parse_list(args.noise_db_values, float)

    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "_out" / f"{args.qam_order}QAM_snr_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    qam_dir, classifier_path = fetch_qam_bundle(args.qam_order, args.cache_dir)

    all_results, cfg = run_sweep(qam_dir, classifier_path, image_indices, noise_db_values, args.num_pilot,
                                  args.nr_softmodem, args.nr_uesoftmodem, args.gnb_conf, args.channel_type,
                                  args.max_retries, out_dir, extra_images=args.extra_images,
                                  mnist_root=args.mnist_root, batch_size=args.batch_size)

    print_summary_table(all_results, cfg)
    plot_sweep(all_results, cfg, out_dir / "snr_sweep.png")
    print(f"all outputs in {out_dir}")


if __name__ == "__main__":
    main()
