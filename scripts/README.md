<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Scripts

Two related but independent tools live here, built on top of the
[custom RE test signal](../doc/CUSTOM_SIGNAL_AI.md) PHY feature
(`openair1/PHY/NR_TRANSPORT/nr_custom_signal.{h,c}`), which writes/reads arbitrary
complex IQ values into real PDSCH data REs on a real OAI gNB/UE over rfsim.

## `custom_re/` — general-purpose signal injection

Not tied to any particular payload. The main entry point is
**`custom_re/send_recv_signal.py`**: give it a signal (a list of complex values, or an
IQ file) and it sends it over the real gNB→rfsim→UE pipeline and returns what was
received — automatically finding a `(slot, symbol, start_sc)` placement that fits the
live PDSCH allocation, rather than requiring one to be hand-picked.

```bash
python3 custom_re/send_recv_signal.py \
  --nr-softmodem <path> --nr-uesoftmodem <path> --gnb-conf <conf> \
  --length 20   # or --iqfile <path>
```
```python
from send_recv_signal import send_recv
received = send_recv(iq_values, nr_softmodem, nr_uesoftmodem, gnb_conf, channel_type="AWGN")
```

Also here: `gen_iq_file.py` (generates a test IQ pattern), `compare_iq.py` (offline
tx/rx comparison), `run_rfsim_test.sh` (the original manual test harness for the raw
feature itself, with `--chanmod`/`--channel-type` support). Full details, the pilot/
calibration story, and troubleshooting: **[`doc/CUSTOM_SIGNAL_AI.md`](../doc/CUSTOM_SIGNAL_AI.md)**.

## `jscc/` — Deep-JSCC image codec demo

A specific application built on top of the tool above: a small neural image codec
(independently implemented, architecture from Bourtsoulatze et al.) whose channel
symbols *are* the payload sent through `send_recv_signal.py` — an image goes in one
end, real DMRS-based channel estimation and equalization happen in the middle, a
reconstructed image comes out the other end. The main entry point is
**`jscc/run_demo.py`**, which only handles the JSCC-specific parts (encode/decode) and
delegates all RF transport to `custom_re/send_recv_signal.py`.

```bash
python3 jscc/run_demo.py \
  --nr-softmodem <path> --nr-uesoftmodem <path> --gnb-conf <conf> \
  --checkpoint jscc/checkpoints/cifar10_c8_awgn10.pth --cifar-index 3 \
  --channel-type AWGN
```

Also here: `train.py`, `encode_image.py`/`decode_image.py`, `model.py`, `pilot.py`,
`cifar_dataset.py`, a `Dockerfile` for cloud/GPU training. Full details, measured PSNR
results, and honest caveats: **[`doc/JSCC_DEMO.md`](../doc/JSCC_DEMO.md)**.

## `semantic_qam/` — criticality-aware QAM analysis

A third application on the same transport: instead of a payload we train ourselves,
this one carries pre-computed criticality-aware ("semantic") and Gray-coded
("standard") QAM symbols exported by
[THE-TRAIN-LAB/Semantic-QAM](https://github.com/THE-TRAIN-LAB/Semantic-QAM) /
[THE-TRAIN-LAB/OAI_Demo](https://github.com/THE-TRAIN-LAB/OAI_Demo), and asks whether
the criticality-aware constellation actually preserves meaning better under noise once
it's sent over this real pipeline instead of a simulated channel. The main entry point
is **`semantic_qam/snr_sweep.py`**, which sweeps the channel noise level and compares
both schemes on label preservation / semantic quality (not just raw symbol error rate
or PSNR). Full details: **[`semantic_qam/README.md`](semantic_qam/README.md)**.

## Why the transport is separate

`send_recv_signal.py` knows nothing about images, models, or pilots — it moves opaque
complex numbers through the real PHY and back. Each payload's own code (`jscc/run_demo.py`,
`semantic_qam/snr_sweep.py`) knows nothing about RF launch mechanics, placement
discovery, or fading-channel retries — it just encodes/loads symbols, calls the
transport, and decodes. Keeping them separate is why `semantic_qam/` could reuse the
transport as-is: a new payload (someone else's exported QAM symbols, in this case)
without touching `send_recv_signal.py` at all.
