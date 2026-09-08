<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Deep-JSCC image transmission over the real OAI PDSCH pipeline

[[_TOC_]]

## What this is

An end-to-end demo that replaces a simulated channel with the **real OAI PHY**: a
small CNN ([Deep Joint Source-Channel Coding](https://ieeexplore.ieee.org/abstract/document/8723589),
Bourtsoulatze et al. 2019) encodes an image directly into complex channel symbols —
no separate bit-level source/channel coding step — those symbols are injected into
real PDSCH data REs via the [custom RE test signal](CUSTOM_SIGNAL_AI.md) feature, sent
through a real gNB → rfsim → real UE chain (optionally with a real channel model:
AWGN, Rayleigh, ...), read back **after** genuine DMRS-based channel estimation and
equalization, and decoded back into an image by the CNN's other half.

This is independently implemented (see `scripts/jscc/model.py`'s header) following the
architecture described in the paper above, cross-checked against
[chunbaobao/Deep-JSCC-PyTorch](https://github.com/chunbaobao/Deep-JSCC-PyTorch) (which
has no LICENSE file, so its code isn't reused directly — only its publicly-documented
architecture and default hyperparameters).

Needs no GPU and no RF hardware: rfsim is pure software (proven throughout this
project's testing), and training this small model is CPU-tractable. Runs on any VM
with a handful of CPU cores — see the `Dockerfile` in `scripts/jscc/` for a
self-contained, cloud-deployable build.

## Architecture

For a 32×32×3 CIFAR-10 image, the encoder (5 conv layers, strides 2/2/1/1/1) produces
a `(2c, 8, 8)` real tensor, normalized so the whole vector has a fixed average power.
This is read as `c` complex symbols at each of the 64 spatial positions (first `c`
channels = real parts, next `c` = imaginary parts) — `c*64` complex channel symbols
total. `c=8` gives the paper's "ratio ≈ 1/6" operating point: 512 complex symbols for a
3072-real-value image. That comfortably fits in **one** OFDM symbol's worth of PDSCH
data REs (a 106-PRB allocation has up to 1272 REs on a non-DMRS symbol) — this first
version doesn't need to split an image across multiple symbols/slots.

```
image (3,32,32) --Encoder--> (2c,8,8) --peak-normalize--> c*64 complex symbols
                                                                |
                                              [pilot prefix prepended]
                                                                |
                                                   custom_iq.txt (this project's
                                                   existing IQ-file format)
                                                                |
                                    gNB: nr_generate_custom_signal() overwrites
                                    PDSCH data REs on a non-DMRS symbol (real DMRS
                                    elsewhere in the slot drives real chan-est)
                                                                |
                                                     rfsim (+ optional channel model)
                                                                |
                                    UE: nr_channel_compensation() -> nr_extract_
                                    custom_signal() reads the same REs POST-equalization
                                                                |
                                            pilot-based scale/rotation calibration
                                                                |
                                                        Decoder --> reconstructed image
```

## Why a pilot prefix

`compare_iq.py` (used to verify the custom-RE feature itself) fits the residual
TX-amplitude scale and any channel rotation by comparing the *received* values against
the *known original* — fine for a test harness, but a real receiver doesn't have the
original image to compare against. So `encode_image.py` prepends a small (8 RE)
deterministic pilot pattern (`pilot.py` — the same fixed QPSK-like sequence
`scripts/custom_re/gen_iq_file.py` already used) before the actual image payload.
`decode_image.py` fits the complex scale/rotation from **only those 8 known REs**
(same least-squares math as `compare_iq.py`) and applies it to the rest — genuine
reference-based calibration, not "cheating" with the secret payload.

## Setup

```bash
cd scripts/jscc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls PyTorch/torchvision's CPU wheels. Build the gNB/UE binaries as
usual (see [`CUSTOM_SIGNAL_AI.md`](CUSTOM_SIGNAL_AI.md)):
```bash
cd ../../cmake_targets
./build_oai --gNB --nrUE -P --build-tool-opt "-j$(nproc)"
```

**Note on CIFAR-10**: `cifar_dataset.py` pulls CIFAR-10 from HuggingFace's parquet
mirror rather than torchvision's default host (`www.cs.toronto.edu`), which was
throttled to a few hundred bytes/sec in the environment this was developed in — if
your network reaches the original host fine, you can swap back to
`torchvision.datasets.CIFAR10` freely; the two are equivalent data.

## Usage

### 1. Train a model

```bash
python train.py --epochs 15 --inner-channels 8 --channel AWGN --snr-db 10
```
Saves `checkpoints/cifar10_c8_awgn10.pth`. This is a genuinely tiny model on 32×32
images — a few minutes per epoch on a modern CPU core, no GPU needed — but training
time scales with epoch count and CPU speed; time your own first epoch before committing
to a long run. The `--channel`/`--snr-db` here is the *training-time* simulated channel
(`model.py`'s `Channel` class, matching the paper's own training setup) used only to
make the model robust to noise — it has no direct relationship to whatever `rfsim`
channel condition you test with later (see the caveat below).

### 2. Run the end-to-end demo

```bash
cd cmake_targets/ran_build/build
python3 ../../../scripts/jscc/run_demo.py \
  --nr-softmodem ./nr-softmodem --nr-uesoftmodem ./nr-uesoftmodem \
  --gnb-conf ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.pci0.rfsim.conf \
  --checkpoint ../../../scripts/jscc/checkpoints/cifar10_c8_awgn10.pth \
  --cifar-index 3 \
  --out-dir /tmp/jscc_out
```
Add `--channel-type AWGN` or `--channel-type Rayleigh8` (any type from
`CHANNELMOD_MAP_INIT` in `openair1/SIMULATION/TOOLS/sim.h`) to exercise a real noisy/
fading channel instead of rfsim's default near-ideal passthrough — same flag semantics
as `scripts/custom_re/run_rfsim_test.sh --channel-type`. Use `--image <path>` instead of
`--cifar-index N` to encode an arbitrary image file (resized to 32×32).

Output: `original.png`, `reconstructed.png`, and the reported PSNR in `--out-dir`.

### 3. Encode/decode standalone (no RF pipeline)

Useful for iterating on the model without spending time on the RF chain each time:
```bash
python encode_image.py --checkpoint checkpoints/cifar10_c8_awgn10.pth --out-dir /tmp/enc --cifar-index 3
# ... inject /tmp/enc/custom_iq.txt however you like, capture a custom_rx_iq.m ...
python decode_image.py --meta /tmp/enc/meta.json --rx-m /tmp/enc/custom_rx_iq.m --out /tmp/enc/reconstructed.png --original /tmp/enc/original.png
```

## Measured results (15 epochs, `c=8`, trained AWGN/10dB)

Diagnostic methodology: compare (a) the "ideal ceiling" — encoder→decoder in pure
float, no RF, no quantization at all — against (b) the real gNB→rfsim→UE pipeline, on
the *same* frozen checkpoint. If they match, the transport is lossless and PSNR is
purely a training question; if (b) is much worse than (a), there's real transport loss
to chase.

| | PSNR |
|---|---|
| Training-set PSNR, epoch 1 → 15 (with training-time AWGN/10dB channel) | 18.6dB → 26.5dB |
| Ideal ceiling, held-out test images (mean of 6) | 28.6dB |
| Real pipeline, no channel model | 30.1dB (0.2dB from ideal, same image) |
| Real pipeline, `--channel-type AWGN` | 29.3dB (~1dB realistic noise penalty) |

Measuring the real transport's *actual* delivered SNR (fit a complex scale from the
gNB's `custom_tx_iq.m` vs the UE's `custom_rx_iq.m`, then residual-vs-signal power)
under `--channel-type AWGN` gave **~11.4dB** — close to the 10dB training assumption,
so train/test channel mismatch turned out to be a small effect in practice for this
config, not the dominant source of error. **The transport is essentially lossless; PSNR
is set by how well the model is trained**, not by anything RF-side. The training curve
was still climbing (not fully plateaued) at epoch 15 — more epochs (practically
speaking, on a GPU) should improve it further.
- **Single-symbol size limit**: this version carries the whole image in one OFDM
  symbol's worth of REs (plus 8 pilot REs) in a single firing — it does not yet split
  a larger image or lower compression ratio across multiple symbols/slots.
- **Fading channels can fail to sync, not just degrade**: as observed testing the
  underlying custom-RE feature, a bad enough random fade (e.g. `Rayleigh8`) can prevent
  the UE from ever acquiring sync for that run. `run_demo.py` handles this itself:
  `--max-retries` (default 4) redraws a fresh channel realization and retries the whole
  gNB+UE launch automatically, reporting each attempt ("UE never captured (sync failure
  is possible with fading channels...)" per failed attempt) rather than silently
  returning a garbage image or requiring a manual rerun. In testing, one Rayleigh8 run
  out of several needed a second attempt; none needed a third.
- **Occasional unrelated UE startup crash**: once during testing the UE hit
  `Assertion (ret == 0) failed! ... Error in pthread_setname_np()` at startup and never
  got as far as syncing - a transient OAI thread-startup race, not reproducible, and
  unrelated to this feature (a plain retry succeeded immediately). If `run_demo.py`
  reports the UE never captured, check its log for this before assuming a real issue.
- **RF transport is not `run_demo.py`'s job** — it delegates entirely to
  `scripts/custom_re/send_recv_signal.py`, which auto-discovers a `(slot, symbol,
  start_sc)` placement that fits the live PDSCH allocation (by reading the same
  `LOG_W ... doesn't fit this slot's PDSCH allocation` validation this feature already
  produces) rather than assuming the slot 1 / symbol 13 default that was hard-coded here
  in an earlier version. See [`scripts/README.md`](../scripts/README.md) for why these
  are two separate scripts.

## Cloud deployment

`scripts/jscc/Dockerfile` builds a self-contained image (OAI's `build_oai -I` installs
its own build/runtime dependencies; the Python venv installs `requirements.txt`). The
RF side (rfsim/gNB/UE) needs only CPU, no RF hardware, regardless of platform:
```bash
docker build -f scripts/jscc/Dockerfile -t oai-jscc-demo .
docker run --rm -it oai-jscc-demo bash
# inside: train.py / run_demo.py as above
```
This wasn't deployed to an actual cloud VM/account as part of this work (none was
available), but nothing in the image or the demo scripts assumes anything beyond a
generic Linux VM with a few CPU cores — the same property already demonstrated by
every rfsim test in this project running in a sandboxed container with no GPU.

**Moving `train.py` to a GPU cloud VM** (the RF pipeline itself gets no benefit from a
GPU — only training does): `train.py` already auto-selects `cuda` when available
(`torch.cuda.is_available()`). Build with CUDA wheels instead of the default CPU ones:
```bash
docker build -f scripts/jscc/Dockerfile \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t oai-jscc-demo-gpu .
```
The base image (`ubuntu:noble`) has no CUDA runtime itself — either switch the
Dockerfile's `FROM` to a matching `nvidia/cuda` base image, or run against a host/
runtime that provides one (`docker run --gpus all ...` with the NVIDIA Container
Toolkit installed). Not validated end-to-end here (no GPU in this environment) — the
CPU path is what's actually been tested throughout this document.
