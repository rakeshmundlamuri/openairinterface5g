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

## Why two scripts

`send_recv_signal.py` knows nothing about images, models, or pilots — it moves opaque
complex numbers through the real PHY and back. `run_demo.py` knows nothing about RF
launch mechanics, placement discovery, or fading-channel retries — it just encodes,
calls the transport, and decodes. Keeping them separate means the transport tool is
reusable for any future payload (a different codec, a raw test signal, criticality-aware
QAM symbols, ...) without dragging in JSCC-specific code, and the JSCC demo stays
readable as "the ML part" without RF plumbing mixed in.
