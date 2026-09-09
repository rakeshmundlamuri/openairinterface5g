<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# `semantic_qam/` — criticality-aware QAM over a real PDSCH link

Another application built on top of [`custom_re/send_recv_signal.py`](../custom_re/send_recv_signal.py)
(see [`../README.md`](../README.md) for how these directories relate): instead of a
neural image codec (that's [`jscc/`](../jscc/)), this one carries pre-computed QAM
symbols exported by [THE-TRAIN-LAB/Semantic-QAM](https://github.com/THE-TRAIN-LAB/Semantic-QAM)
and [THE-TRAIN-LAB/OAI_Demo](https://github.com/THE-TRAIN-LAB/OAI_Demo) — a VQ-VAE
with a learned per-slot "criticality"/importance score, mapped onto a constellation
whose point positions were trained to protect the most important bits.

The question this answers: when THE-TRAIN-LAB/OAI_Demo's criticality-aware
("semantic") constellation and its Gray-coded ("standard") baseline are sent through
the *same real gNB→rfsim→UE PDSCH pipeline* instead of a Sionna-simulated channel,
does the criticality-aware one actually preserve meaning better under noise?

## Files

- **`fetch_bundle.py`** — downloads one QAM order's exported bundle from
  THE-TRAIN-LAB/OAI_Demo (`config.json`, the MNIST dataset + pre-computed QAM symbols,
  the VQ-VAE and both constellations' state dicts, the shared classifier) into a local
  cache (`data/`, gitignored). Skips files already downloaded.
- **`model.py`** — inference-only PyTorch classes matching OAI_Demo's architecture
  (`ImportanceVQVAE`, `LearnedConstellation`, `TaskClassifier`), independently
  reimplemented from its public description (no LICENSE file there, so this isn't a
  copy — see the module docstring). Validated bit-exact against the bundle's own
  exported decode outputs before ever touching real RF. Also has the demod/reconstruct
  pipeline (`demap_and_reconstruct`) and a port of Semantic-QAM's own evaluation
  metrics (`measure_semantic_quality`, `bit_error_rate`).
- **`run_demo.py`** — sends one scheme's symbols for one image over real rfsim once,
  prints SER/PSNR/classifier-correctness, and saves the original/reconstructed images.
  Also holds the shared building blocks (`load_scheme_resources`, `evaluate_image`,
  `send_scheme_over_rfsim`) that `snr_sweep.py` reuses.
- **`snr_sweep.py`** — the actual analysis tool: sweeps rfsim's AWGN
  `noise_power_dB` channel-model knob, evaluating both schemes across all 10 images at
  each noise level, then prints a summary table and saves a 6-panel comparison plot
  (label preservation, semantic quality, PSNR, BER, SER vs noise). Reproduces
  THE-TRAIN-LAB/Semantic-QAM's own notebook analysis
  (`measure_semantic_quality()` / `evaluate_and_report_snr_sweep()`), but driven by the
  real pipeline instead of a simulated channel, and comparing the two constellations
  against each other at each noise level instead of sweeping an adaptive-K agent.

## Setup

Needs a built `nr-softmodem`/`nr-uesoftmodem` (rfsim-enabled) and Python with the
packages in `requirements.txt` (PyTorch/torchvision, numpy, matplotlib):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run of any script also downloads that QAM order's bundle (a few MB) from
THE-TRAIN-LAB/OAI_Demo into `data/` (gitignored) - needs network access once, then it's
cached.

## Usage

Run these from inside `scripts/semantic_qam/` (or adjust the relative paths below) with
the venv above active.

One image, no channel model, both schemes:

```bash
python3 run_demo.py \
  --nr-softmodem <path> --nr-uesoftmodem <path> --gnb-conf <conf> \
  --qam-order 64 --image-index 0 --scheme both
```

Full noise sweep, all 10 images, both schemes:

```bash
python3 snr_sweep.py \
  --nr-softmodem <path> --nr-uesoftmodem <path> --gnb-conf <conf> \
  --qam-order 64 --image-indices all --noise-db-values="-20,-17,-14,-11,-8,-5" \
  --channel-type AWGN --out-dir _out/64QAM_sweep
```

`--image-indices` also takes a comma-separated subset (e.g. `0,1,2`) for a quick
smoke test before committing to the full 10-image run. There's no single noise range
that works for every QAM order — a higher order packs more points into the same
unit-power constellation, so the useful transition band (where errors start appearing
but haven't collapsed everything yet) sits in a different place. In practice: probe a
wide range with 1-2 images first, then run the full 10-image sweep zoomed into
whatever band actually shows a transition. Measured examples so far:

| QAM order | Useful noise band (`noise_power_dB`) |
|---|---|
| 64  | &minus;20 to &minus;5  |
| 256 | &minus;12 to &minus;4  |
| 1024 | &minus;12 to &minus;7 |

Both scripts share `--num-pilot` (default 8, prepended for pilot-based receiver
calibration — same approach as `jscc/decode_image.py`), `--max-retries` (rfsim
fading-channel retry count), and `--cache-dir` (defaults to `data/` here).

## What's *not* reproduced here

THE-TRAIN-LAB/Semantic-QAM's notebook also trains a DQN agent to adaptively choose how
many of the 64 concepts (`k`) to transmit per image, based on channel conditions. This
tool only evaluates the bundles OAI_Demo already exported at their fixed, trained `k`
(64 for all QAM orders tested so far) — it isolates the constellation-geometry effect
(does moving the points help?) from the adaptive-rate question (should fewer concepts
be sent when the channel is bad?), which is a separate experiment this doesn't run.
