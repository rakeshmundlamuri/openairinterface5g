"""Generate additional MNIST test images beyond THE-TRAIN-LAB/OAI_Demo's fixed 10, and
encode them into QAM symbols with an already-loaded bundle's own VQ-VAE - so a sweep
isn't limited to n=10. See encode_image_to_symbols() in model.py for why this is a
valid, self-consistent round trip without needing to replicate OAI_Demo's own
importance-based slot ordering.

The one-time cost here is the MNIST test set download (~10MB, ~20s); after that,
encoding each additional image is a single CPU forward pass (a few tens of ms).
"""
from pathlib import Path

import numpy as np

from model import encode_image_to_symbols

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MNIST_ROOT = SCRIPT_DIR / "data" / "mnist_raw"


def pick_one_per_digit(mnist_test, num_extra, start_search=0):
    """Pick num_extra images cycling through digits 0-9 (same one-per-class balance as
    OAI_Demo's own fixed 10), searching forward from start_search so repeated calls
    (e.g. num_extra=20) don't just re-pick the same first 10."""
    picked = []
    remaining = set(range(10))
    next_wanted = list(range(10))
    idx = start_search
    while len(picked) < num_extra and idx < len(mnist_test):
        img, label = mnist_test[idx]
        cycle_pos = len(picked) % 10
        if label == next_wanted[cycle_pos]:
            picked.append((idx, img, label))
        idx += 1
    if len(picked) < num_extra:
        raise RuntimeError(f"ran out of MNIST test images looking for {num_extra} balanced samples")
    return picked


def build_extra_rows(num_extra, vqvae, scheme_points, mnist_root=DEFAULT_MNIST_ROOT, start_search=0):
    """Returns a dict of numpy arrays (images, labels, selected_slots, symbol_indices,
    {scheme}_qam_symbols_iq for each scheme in scheme_points, decoded_images,
    detected_symbol_indices) for num_extra NEW images, in the same shapes as one
    QAM-order bundle's own npz - one row per image, ready to concatenate onto it.
    scheme_points: dict {scheme_name: constellation_points_tensor}."""
    from torchvision.datasets import MNIST
    import torch

    mnist_root.mkdir(parents=True, exist_ok=True)
    test_set = MNIST(root=str(mnist_root), train=False, download=True)
    picked = pick_one_per_digit(test_set, num_extra, start_search=start_search)

    images, labels, selected_slots_list, symbol_indices_list = [], [], [], []
    per_scheme_symbols = {scheme: [] for scheme in scheme_points}
    for _, img, label in picked:
        arr = (np.asarray(img, dtype=np.float32) / 255.0).reshape(784)
        images.append(arr)
        labels.append(label)
        first_scheme = next(iter(scheme_points))
        symbols_iq, symbol_indices, selected_slots = encode_image_to_symbols(
            vqvae, scheme_points[first_scheme], arr)
        selected_slots_list.append(selected_slots)
        symbol_indices_list.append(symbol_indices)
        per_scheme_symbols[first_scheme].append(symbols_iq)
        for scheme, points in scheme_points.items():
            if scheme == first_scheme:
                continue
            with torch.no_grad():
                gathered = points[torch.as_tensor(symbol_indices, dtype=torch.long)].numpy()
            per_scheme_symbols[scheme].append(gathered)

    # no real Sionna-simulated reference for new images - the clean/no-channel round
    # trip through the same vqvae (decoding the exact indices just encoded) is used as
    # the substitute reference (SER=0 by construction, since these ARE ground truth).
    with torch.no_grad():
        indices_t = torch.as_tensor(np.stack(symbol_indices_list), dtype=torch.long)
        embeddings = torch.nn.functional.embedding(indices_t, vqvae.vq_layer.embeddings)
        decoded = vqvae.decode(embeddings).numpy()

    rows = {
        "images": np.stack(images).astype(np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "selected_slots": np.stack(selected_slots_list).astype(np.int64),
        "symbol_indices": np.stack(symbol_indices_list).astype(np.int64),
        "detected_symbol_indices": np.stack(symbol_indices_list).astype(np.int64),
        "decoded_images": decoded.astype(np.float32),
    }
    for scheme in scheme_points:
        rows[f"{scheme}_qam_symbols_iq"] = np.stack(per_scheme_symbols[scheme]).astype(np.complex64)
    return rows
