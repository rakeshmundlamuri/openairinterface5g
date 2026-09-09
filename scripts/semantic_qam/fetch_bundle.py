"""Download one QAM order's exported bundle from THE-TRAIN-LAB/OAI_Demo into a local
cache - config.json, semantic_qam_dataset.npz, and the vqvae/constellation state dicts
needed to decode it. classifier_state.pt (shared across QAM orders) is fetched once.
"""
from pathlib import Path
from urllib.request import urlretrieve

_BASE_URL = "https://raw.githubusercontent.com/THE-TRAIN-LAB/OAI_Demo/main/Semantic_QAM_10_Image_Dataset_PyTorch"
_VALID_ORDERS = (4, 16, 64, 256, 1024)


def fetch_qam_bundle(qam_order, cache_dir):
    """Returns the local directory containing config.json, semantic_qam_dataset.npz,
    and models/{vqvae,standard_constellation,semantic_constellation}_state.pt."""
    if qam_order not in _VALID_ORDERS:
        raise ValueError(f"qam_order must be one of {_VALID_ORDERS}, got {qam_order}")

    cache_dir = Path(cache_dir)
    qam_dir = cache_dir / f"{qam_order}QAM"
    (qam_dir / "models").mkdir(parents=True, exist_ok=True)

    files = [
        "config.json",
        "semantic_qam_dataset.npz",
        "models/vqvae_state.pt",
        "models/standard_constellation_state.pt",
        "models/semantic_constellation_state.pt",
    ]
    for rel in files:
        local = qam_dir / rel
        if local.exists():
            continue
        url = f"{_BASE_URL}/{qam_order}QAM/{rel}"
        print(f"downloading {url} -> {local}")
        urlretrieve(url, local)

    classifier_path = cache_dir / "classifier_state.pt"
    if not classifier_path.exists():
        url = f"{_BASE_URL}/classifier_state.pt"
        print(f"downloading {url} -> {classifier_path}")
        urlretrieve(url, classifier_path)

    return qam_dir, classifier_path
