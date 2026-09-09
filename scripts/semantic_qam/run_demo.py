#!/usr/bin/env python3
"""Send THE-TRAIN-LAB/OAI_Demo's pre-computed criticality-aware ("semantic") QAM
symbols - or its standard-constellation baseline - for one of its 10 exported MNIST
images over the real gNB->rfsim->UE PDSCH pipeline, and see whether the real channel
still reconstructs the image / preserves its classification label.

This does NOT re-run OAI_Demo's own training or its Sionna-simulated channel - it
takes symbols that repo already computed and exported (semantic_qam_symbols_iq /
standard_qam_symbols_iq, both complex64, shape (10, 64)) and substitutes our real RF
transport (scripts/custom_re/send_recv_signal.py) for Sionna's simulated one, then
demodulates/decodes with an independent reimplementation of the same VQ-VAE
architecture (model.py) loaded from OAI_Demo's own exported state dicts.

Usage:
    run_demo.py --nr-softmodem PATH --nr-uesoftmodem PATH --gnb-conf PATH \
                [--qam-order {4,16,64,256,1024}] [--image-index 0-9] \
                [--scheme semantic|standard|both] [--channel-type TYPE] [--out-dir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from fetch_bundle import fetch_qam_bundle
from model import (ImportanceVQVAE, LearnedConstellation, TaskClassifier, bit_error_rate,
                    demap_and_reconstruct, measure_semantic_quality)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "custom_re"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "jscc"))
from send_recv_signal import send_recv  # noqa: E402
from pilot import pilot_symbols  # noqa: E402

DEFAULT_CACHE_DIR = SCRIPT_DIR / "data"


def fit_pilot_scale(rx_pilot, ref_pilot):
    num = sum(r * t.conjugate() for r, t in zip(rx_pilot, ref_pilot))
    den = sum(abs(t) ** 2 for t in ref_pilot)
    return num / den if den else 0


def send_scheme_over_rfsim(iq_ground_truth, num_pilot, nr_softmodem, nr_uesoftmodem, gnb_conf,
                            channel_type, max_retries, work_dir, noise_power_db=-10, placement=None):
    """Peak-normalize, prepend a pilot, send over real rfsim, pilot-calibrate the
    receive, and return the recovered complex symbols in the ORIGINAL constellation's
    scale (i.e. directly comparable to iq_ground_truth / usable for hard-decision
    demodulation against the true constellation points)."""
    peak_scale = max(abs(v) for v in iq_ground_truth) or 1.0
    normalized = [v / peak_scale for v in iq_ground_truth]
    pilots = pilot_symbols(num_pilot)
    tx = pilots + normalized

    received = send_recv(tx, nr_softmodem, nr_uesoftmodem, gnb_conf, channel_type=channel_type,
                          max_retries=max_retries, work_dir=work_dir, noise_power_db=noise_power_db,
                          placement=placement)
    rx_pilot, rx_payload = received[:num_pilot], received[num_pilot:]
    scale = fit_pilot_scale(rx_pilot, pilots)
    if scale == 0:
        raise RuntimeError("pilot calibration failed (zero scale)")
    calibrated = [(r / scale) * peak_scale for r in rx_payload]
    return calibrated


def load_scheme_resources(scheme, qam_dir, classifier_path):
    """Load everything needed to evaluate `scheme` (semantic/standard) once - reuse the
    returned dict across many evaluate_image() calls (many images / noise levels /
    schemes) instead of re-reading config/npz/state-dicts from disk every time."""
    cfg = json.loads((qam_dir / "config.json").read_text())
    data = np.load(qam_dir / "semantic_qam_dataset.npz")

    vqvae = ImportanceVQVAE(cfg["latent_dim"], cfg["num_embeddings"], cfg["num_concepts"])
    vqvae.load_state_dict(torch.load(qam_dir / "models" / "vqvae_state.pt", map_location="cpu", weights_only=True))
    vqvae.eval()

    constellation = LearnedConstellation(cfg["num_bits_per_symbol"])
    constellation.load_state_dict(
        torch.load(qam_dir / "models" / f"{scheme}_constellation_state.pt", map_location="cpu", weights_only=True))
    constellation.eval()
    with torch.no_grad():
        points = constellation()

    classifier = None
    if classifier_path.exists():
        classifier = TaskClassifier()
        classifier.load_state_dict(torch.load(classifier_path, map_location="cpu", weights_only=True))
        classifier.eval()

    return {"scheme": scheme, "cfg": cfg, "data": data, "vqvae": vqvae, "points": points, "classifier": classifier}


def evaluate_image(resources, image_index, num_pilot, nr_softmodem, nr_uesoftmodem, gnb_conf,
                    channel_type, max_retries, work_dir, noise_power_db=-10, placement=None):
    """Send one image's already-exported QAM symbols for `resources['scheme']` over real
    rfsim and return a flat dict of metrics (no printing, no image saving - callers that
    want those, e.g. run_one_scheme, do it themselves with the returned tensors)."""
    scheme, cfg, data = resources["scheme"], resources["cfg"], resources["data"]
    vqvae, points, classifier = resources["vqvae"], resources["points"], resources["classifier"]

    ground_truth_iq = data[f"{scheme}_qam_symbols_iq"][image_index]
    k = int(data["k"])
    bits_per_concept = int(data["bits_per_concept"])
    bits_per_symbol = int(data["bits_per_symbol"])
    payload_bit_length = k * bits_per_concept
    selected_slots = data["selected_slots"][image_index:image_index + 1]
    true_symbol_indices = data["symbol_indices"][image_index]
    label = int(data["labels"][image_index])

    received_iq = send_scheme_over_rfsim(list(ground_truth_iq), num_pilot, nr_softmodem, nr_uesoftmodem,
                                          gnb_conf, channel_type, max_retries, work_dir,
                                          noise_power_db=noise_power_db, placement=placement)

    recon, recovered_indices, detected_symbols = demap_and_reconstruct(
        np.array(received_iq, dtype=np.complex64), points, selected_slots, k,
        bits_per_concept, bits_per_symbol, payload_bit_length, vqvae)

    symbol_error_rate = float(np.mean(detected_symbols[0] != true_symbol_indices))
    ber = bit_error_rate(detected_symbols[0], true_symbol_indices, bits_per_symbol)
    sionna_ser = float(np.mean(data["detected_symbol_indices"][image_index] != true_symbol_indices))

    original = torch.as_tensor(data["images"][image_index]).reshape(1, 28, 28)
    recon_t = torch.as_tensor(recon[0]).reshape(1, 28, 28)
    mse = float(torch.mean((recon_t - original) ** 2))
    psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item() if mse > 0 else float("inf")
    sionna_decoded = torch.as_tensor(data["decoded_images"][image_index]).reshape(1, 28, 28)

    result = {
        "scheme": scheme, "image_index": image_index, "label": label, "k": k,
        "symbol_error_rate": symbol_error_rate, "sionna_symbol_error_rate": sionna_ser,
        "ber": ber, "psnr": psnr,
        "original": original, "recon": recon_t, "sionna_decoded": sionna_decoded,
    }
    if classifier is not None:
        with torch.no_grad():
            pred = classifier(recon_t.reshape(1, 784)).argmax(1).item()
        result["classifier_prediction"] = pred
        result["classifier_correct"] = int(pred == label)
        quality = measure_semantic_quality(
            original.reshape(1, 784), recon_t.reshape(1, 784), classifier, torch.tensor([label]))
        result.update(quality)
    return result


def run_one_scheme(scheme, qam_dir, classifier_path, image_index, num_pilot,
                    nr_softmodem, nr_uesoftmodem, gnb_conf, channel_type, max_retries, out_dir):
    resources = load_scheme_resources(scheme, qam_dir, classifier_path)
    cfg = resources["cfg"]
    print(f"[{scheme}] sending {cfg['qam_order']}-QAM symbols "
          f"(image {image_index}, label {int(resources['data']['labels'][image_index])}) over real rfsim...")
    r = evaluate_image(resources, image_index, num_pilot, nr_softmodem, nr_uesoftmodem,
                        gnb_conf, channel_type, max_retries, out_dir / scheme)

    accuracy_note = ""
    if "classifier_prediction" in r:
        accuracy_note = (f", classifier prediction={r['classifier_prediction']} "
                          f"({'correct' if r['classifier_correct'] else 'WRONG'})")

    save_image(r["recon"], out_dir / f"reconstructed_{scheme}.png")
    save_image(r["original"], out_dir / "original.png")
    save_image(r["sionna_decoded"], out_dir / f"reconstructed_{scheme}_sionna_simulated.png")

    print(f"[{scheme}] symbol error rate over real rfsim: {r['symbol_error_rate']:.3f} "
          f"(Sionna-simulated bundle's own SER: {r['sionna_symbol_error_rate']:.3f})")
    print(f"[{scheme}] reconstruction PSNR: {r['psnr']:.2f}dB (label={r['label']}{accuracy_note})")
    return {"scheme": scheme, "symbol_error_rate": r["symbol_error_rate"], "psnr": r["psnr"], "label": r["label"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr-softmodem", required=True)
    ap.add_argument("--nr-uesoftmodem", required=True)
    ap.add_argument("--gnb-conf", required=True)
    ap.add_argument("--qam-order", type=int, default=64, choices=[4, 16, 64, 256, 1024])
    ap.add_argument("--image-index", type=int, default=0, help="0-9, index into OAI_Demo's fixed 10-image set")
    ap.add_argument("--scheme", default="both", choices=["semantic", "standard", "both"],
                     help="semantic = criticality-aware learned constellation, standard = Sionna Gray-coded baseline")
    ap.add_argument("--num-pilot", type=int, default=8)
    ap.add_argument("--channel-type", default=None, help="e.g. AWGN, Rayleigh8 (default: no channel model)")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "_out" / f"{args.qam_order}QAM_img{args.image_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    qam_dir, classifier_path = fetch_qam_bundle(args.qam_order, args.cache_dir)

    schemes = ["semantic", "standard"] if args.scheme == "both" else [args.scheme]
    results = [run_one_scheme(scheme, qam_dir, classifier_path, args.image_index, args.num_pilot,
                               args.nr_softmodem, args.nr_uesoftmodem, args.gnb_conf,
                               args.channel_type, args.max_retries, out_dir)
               for scheme in schemes]

    print(f"\ndone. outputs in {out_dir}")
    if len(results) == 2:
        sem, std = (r for r in results if r["scheme"] == "semantic"), (r for r in results if r["scheme"] == "standard")
        sem, std = next(sem), next(std)
        print(f"criticality-aware vs standard, same real rfsim conditions: "
              f"SER {sem['symbol_error_rate']:.3f} vs {std['symbol_error_rate']:.3f}, "
              f"PSNR {sem['psnr']:.2f}dB vs {std['psnr']:.2f}dB")


if __name__ == "__main__":
    main()
