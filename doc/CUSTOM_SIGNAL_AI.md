<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Custom RE test signal

[[_TOC_]]

> A real application built on top of this feature: [`JSCC_DEMO.md`](JSCC_DEMO.md) sends
> an actual image through it, encoded/decoded by a small neural net (Deep-JSCC), as a
> demonstration of transporting an arbitrary complex payload rather than a synthetic
> test pattern.

## What this is

A debug/test feature for the NR PHY that:

1. overwrites a small, fixed set of **PDSCH data resource elements (REs)** within a
   real, scheduled PDSCH allocation with known complex IQ values loaded from a plain
   text file — on an OFDM symbol that carries no DMRS, so real DMRS elsewhere in the
   same slot still drives genuine channel estimation, and
2. lets the UE read those same REs back **after** channel estimation and equalization
   (i.e. from the compensated PDSCH symbols), so the whole downlink pipeline —
   resource-grid mapping, OFDM modulation, rfsim channel, FFT, real DMRS-based channel
   estimation, equalization/compensation, RE extraction — is exercised and verified
   end to end.

It is not a 3GPP channel or reference signal — it's a stepping stone for future
custom-signal work, and a standalone tool for sanity-checking the DL PDSCH pipeline
(e.g. after touching PHY resource-mapping or channel-estimation code).

The core logic lives in `openair1/PHY/NR_TRANSPORT/nr_custom_signal.{h,c}`:

- `nr_custom_signal_load_file()` — parses the IQ file into memory, scaled to the gNB's
  TX amplitude convention.
- `nr_generate_custom_signal()` — writes those values into `txdataF` at the configured
  slot/symbol/subcarriers. Called from `phy_procedures_gNB_TX()`
  (`openair1/SCHED_NR/phy_procedures_nr_gNB.c`) **after** `nr_generate_pdsch()` has run
  for the slot, overwriting the data REs it just wrote — DMRS on other symbols of the
  same slot is untouched.
- `nr_extract_custom_signal()` — reads the same REs back out of the UE's
  **channel-compensated** PDSCH symbol buffer (`rxdataF_comp`, populated by
  `nr_channel_compensation()`), not the raw pre-equalization grid. Called from
  `pdsch_processing()` (`openair1/SCHED_NR_UE/phy_procedures_nr_ue.c`), right after
  `nr_ue_pdsch_procedures()` returns for the codeword.

Both hooks are gated behind `--custom-re-enable` and are no-ops otherwise. Both also
**validate the chosen (symbol, subcarrier range) against the live PDSCH allocation**
before writing/reading — if it doesn't fit (wrong symbol, a DMRS-occasion symbol, or
outside the scheduled RB range), they log a warning and skip that slot rather than
silently writing/reading garbage. See "Troubleshooting" below.

### Why a non-DMRS symbol

Within the PDSCH allocation, one or more OFDM symbols carry real DMRS (interleaved with
data on specific subcarriers per `dmrsConfigType`); the remaining symbols are pure
data — every subcarrier in those symbols is a data RE. This feature only targets a pure
**data symbol** (checked via the PDU's `dlDmrsSymbPos` bitmap): it avoids replicating
the exact per-subcarrier DMRS-vs-data pattern on both TX and RX, while still landing on
genuine PDSCH data REs, with genuine DMRS elsewhere in the slot driving the channel
estimate used to equalize them. Interleaving the custom signal with DMRS on the *same*
symbol is out of scope for this version.

## IQ file format

Plain text, one RE per line, `re,im` as floats in `[-1, 1]`:

```
0.707107,0.707107
-0.707107,0.707107
-0.707107,-0.707107
0.707107,-0.707107
```

`gen_iq_file.py` (see below) generates one for you; you can also hand-write one if you
want a specific pattern.

On load, each value is scaled by the gNB's TX amplitude (`c16mulRealShift`), the same
convention used by PSS/SSS/PBCH/PRS/PDSCH, so the injected signal sits at a comparable
power level to the rest of the downlink.

## CLI options

Shared between `nr-softmodem` and `nr-uesoftmodem` (both must be given the **same**
values so they agree on where to write/read):

| Option | Meaning | Default |
|---|---|---|
| `--custom-re-enable` | turn the feature on | off |
| `--custom-re-iqfile <path>` | path to the IQ file (gNB loads it; UE just needs it for symmetry/logging) | — |
| `--custom-re-frame <n>` | firing period in frames: fires when `frame % n == 0`; **`0` means every frame** | `0` |
| `--custom-re-slot <n>` | slot within the frame — must be a slot your scheduler actually assigns a PDSCH to | `0` |
| `--custom-re-symbol <n>` | OFDM symbol within the slot — must be a **non-DMRS** symbol of that PDSCH's allocation | `13` |
| `--custom-re-startsc <n>` | first subcarrier (logical RE index from subcarrier 0 of the channel) — must fall inside the PDSCH's RB allocation | `0` |
| `--custom-re-numre <n>` | number of consecutive REs | `12` |

Unlike a generic "write anywhere" test signal, these values are **validated against the
live PDSCH allocation on every firing** (see below) — if your config doesn't actually
schedule a PDSCH in the chosen slot, or the symbol/subcarrier range doesn't fit that
PDSCH's allocation, the gNB/UE log a warning and skip instead of writing/reading
garbage. Run once with logging on and look at the warning if you're not sure what
values to pick for your setup (see Troubleshooting).

`--custom-re-frame 0` (the default) is deliberately "every frame, not a single exact
frame": the UE takes several frames to synchronize, so a one-shot exact-frame match is
easy to miss entirely by the time the UE is actually running PDSCH processing. With
period `0`, the signal is present continuously on the chosen slot, so there's no timing
race to get right — just start the UE and it will land on the very next occurrence.

## Logs and dumps produced

When the condition fires and the allocation check passes:

- gNB: logs `CUSTOM_RE_TX_CAPTURED frame <f> slot <s>` and writes `custom_tx_iq.m`
  (in its current working directory) with the values it just wrote.
- UE: logs `CUSTOM_RE_RX_CAPTURED frame <f> slot <s>` and writes `custom_rx_iq.m` with
  the channel-compensated (post-equalization) values it read back.

If the allocation check fails, each side instead logs (at warning level, always
visible) either `no PDSCH scheduled in frame <f> slot <s>` or `symbol/subcarrier
config doesn't fit this slot's PDSCH allocation`, with the actual allocation bounds it
found, so you can adjust the CLI options accordingly.

The capture markers use `LOG_A` (analysis level) — pass `--log_config.global_log_level
analysis` if you don't otherwise see them (the skip warnings use `LOG_W`, which is
visible by default).

`custom_tx_iq.m`/`custom_rx_iq.m` are plain MATLAB/Octave-style dumps (one complex value
per line, `re + j*(im)`), the same format `LOG_M` uses elsewhere in the codebase. Note
the RX dump's values are on a different absolute scale than the TX dump's — they've
been through real channel estimation/equalization, which renormalizes amplitude
relative to the channel estimate. `compare_iq.py` (below) fits out that scale before
comparing, so this is expected, not a bug.

## Testing it yourself

### 1. Build

```bash
cd cmake_targets
./build_oai --gNB --nrUE -P --build-tool-opt "-j16"
```

This builds `nr-softmodem`, `nr-uesoftmodem`, and the phy-simulators into
`cmake_targets/ran_build/build/`.

### 2. Easiest: one-command script

```bash
cd cmake_targets/ran_build/build
../../../scripts/custom_re/run_rfsim_test.sh \
  ./nr-softmodem ./nr-uesoftmodem \
  ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.pci0.rfsim.conf
```

This generates the IQ file, starts gNB+UE over rfsim in `--phy-test` mode (no core
network needed), waits for both capture markers, then compares **both** the gNB's TX
capture and the UE's RX capture against the original IQ file, printing `PASS`/`FAIL`
for each. Takes ~15-20s. Ctrl-C is handled — it stops both processes on exit.

The script uses `--custom-re-slot 1 --custom-re-symbol 13`: slot 1 is where
`--phy-test`'s default scheduler (`dlsch_slot_bitmap = 1<<1`, see
`openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_phytest.c`) places PDSCH by default, and
symbol 13 is a non-DMRS symbol of that allocation (single front-loaded DMRS symbol,
mapping type A). If you use a different scheduler config, adjust these — the `LOG_W`
skip warnings will tell you the real allocation bounds if your first guess doesn't fit.

Example output:

```
comparing gNB TX capture against /tmp/tmp.xxx/custom_iq.txt...
RE 0: original=0.7071+0.7071j captured=(366+366j) descaled=0.7061+0.7061j rel_err=0.001
...
scale removed a=518.309+0j, max relative error=0.001 (tolerance=0.1)
PASS

comparing UE RX capture against /tmp/tmp.xxx/custom_iq.txt...
RE 0: original=0.7071+0.7071j captured=(261+260j) descaled=0.7040+0.7026j rel_err=0.005
...
scale removed a=370.406-0.353553j, max relative error=0.014 (tolerance=0.1)
PASS
```

(Note the UE's `captured` magnitude, ~260, differs from the gNB's, ~366 — that's the
channel-compensation renormalization mentioned above, not an error; both still descale
back to the same ~0.7071 original values.)

### 3. Manual, step by step (if you want to watch it / poke around)

**Generate the IQ file:**

```bash
cd cmake_targets/ran_build/build
python3 ../../../scripts/custom_re/gen_iq_file.py /tmp/custom_iq.txt 12
```

**Terminal 1 — start the gNB** (`--phy-test` mode needs no core network; the extra
`min_rxtxtime` is required by this reference conf/mode combination):

```bash
mkdir -p /tmp/gnb_run && cd /tmp/gnb_run
/path/to/cmake_targets/ran_build/build/nr-softmodem \
  -O /path/to/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.pci0.rfsim.conf \
  --rfsim --phy-test --gNBs.[0].min_rxtxtime 3 \
  --custom-re-enable --custom-re-iqfile /tmp/custom_iq.txt \
  --custom-re-frame 0 --custom-re-slot 1 --custom-re-symbol 13 --custom-re-startsc 0 --custom-re-numre 12 \
  --log_config.global_log_level analysis
```

Wait until `rbconfig.raw` and `reconfig.raw` appear in `/tmp/gnb_run/` — `--phy-test`
writes these instead of doing a real RRC exchange, and the UE needs to read them.
Within a second or two you should also see `CUSTOM_RE_TX_CAPTURED` scrolling by; if you
instead see a `custom RE signal: ... skipping` warning, the allocation info it prints
tells you what `--custom-re-*` values will actually fit.

**Terminal 2 — start the UE**, pointing at those two files:

```bash
mkdir -p /tmp/ue_run && cd /tmp/ue_run
/path/to/cmake_targets/ran_build/build/nr-uesoftmodem \
  --rfsim --phy-test \
  --reconfig-file /tmp/gnb_run/reconfig.raw --rbconfig-file /tmp/gnb_run/rbconfig.raw \
  --rfsimulator.serveraddr 127.0.0.1 \
  --custom-re-enable --custom-re-iqfile /tmp/custom_iq.txt \
  --custom-re-frame 0 --custom-re-slot 1 --custom-re-symbol 13 --custom-re-startsc 0 --custom-re-numre 12 \
  --log_config.global_log_level analysis
```

You should see `CUSTOM_RE_RX_CAPTURED` scroll by in terminal 2 within a second or two
of the UE syncing (repeating every frame, since the period is `0`).

**Terminal 3 — compare** (once both files exist):

```bash
python3 scripts/custom_re/compare_iq.py /tmp/custom_iq.txt /tmp/gnb_run/custom_tx_iq.m
python3 scripts/custom_re/compare_iq.py /tmp/custom_iq.txt /tmp/ue_run/custom_rx_iq.m
```

Each prints per-RE `original`/`captured`/`descaled`/`rel_err` and a final
`PASS`/`FAIL`. `compare_iq.py` fits a single complex scale factor (removing the gNB's
TX amplitude scaling, or the UE's channel-compensation renormalization, plus any
residual rotation) before comparing, so what you see in the `descaled` column is
directly comparable to `custom_iq.txt`'s original values.

Ctrl-C both processes when done.

### 4. Testing with a real channel model (AWGN, Rayleigh, ...)

By default rfsim is a near-ideal passthrough (no noise, no multipath) — the flow above
runs that way, giving sub-1% error. To instead exercise the feature over a real channel,
pass `--chanmod` (AWGN, the default type) or `--channel-type <TYPE>` (implies `--chanmod`)
to the script:

```bash
cd cmake_targets/ran_build/build

# AWGN (noise only, no fading)
../../../scripts/custom_re/run_rfsim_test.sh \
  ./nr-softmodem ./nr-uesoftmodem \
  ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.pci0.rfsim.conf \
  --chanmod

# 8-tap Rayleigh fading (real frequency-selective multipath)
../../../scripts/custom_re/run_rfsim_test.sh \
  ./nr-softmodem ./nr-uesoftmodem \
  ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.pci0.rfsim.conf \
  --channel-type Rayleigh8
```

`AWGN` is literally a fixed, deterministic single path plus additive Gaussian noise — no
fading, no multipath (the sim code's own comment: Ricean factor `0` = AWGN, `1` =
Rayleigh). `--channel-type` accepts any of rfsim's channel model types — `Rayleigh8`,
`Rayleigh1`, `Rice8`, `TDL_A`/`TDL_B`/`TDL_C`/`TDL_D`/`TDL_E`, `EPA`/`EVA`/`ETU`, etc.
(full list: `CHANNELMOD_MAP_INIT` in `openair1/SIMULATION/TOOLS/sim.h`) — the script
generates a `channelmod_rfsimu.conf` with that type, `noise_power_dB = -10`,
`ploss_dB = 0` for both link directions and includes it via `-O` on both gNB and UE.

The gNB's TX capture is always unaffected (~0.1% error — the channel only touches the RF
link, not what the gNB itself wrote). The UE's RX capture reflects the real channel:
AWGN shows small, evenly-scattered per-RE error (a few percent), so the script uses a
`0.15` comparison tolerance for it; anything with real fading (Rayleigh/Rice/TDL/...) can
scatter individual REs much further on an unlucky channel draw (observed up to ~17% on
one run), so the script uses `0.2` for those. This is expected behavior of a noisy/fading
channel, not a sign the write/read pipeline is broken.

Because fading models draw a **random** channel realization each run, occasionally the
draw is bad enough that **sync itself fails** (repeated `synch Failed` in the UE log,
never reaching `CUSTOM_RE_RX_CAPTURED` at all) — this happened on one run of `Rayleigh8`
during testing. That's the channel model doing its job (a real receiver would lose sync
on a deep enough fade too), not a script or feature bug — just rerun.

Doing this manually (outside the script) needs two extra pieces: a `channelmod_rfsimu.conf`
(see [`channel_simulation.md`](../openair1/SIMULATION/TOOLS/DOC/channel_simulation.md)
for the format) alongside a `-O` conf file that ends with
`@include "channelmod_rfsimu.conf"` for **both** the gNB and the UE (a UE conf containing
only that one line is enough in `--phy-test` mode), and `--rfsimulator.[0].options
chanmod` on both command lines. Also note that once you pass any `--rfsimulator.[0].*`
option, keep all rfsimulator options in that same indexed form (e.g.
`--rfsimulator.[0].serveraddr`, not the unindexed `--rfsimulator.serveraddr`) — mixing the
two forms in one invocation makes the config module reject the unindexed one as an
"unknown option".

### 5. Faster iteration: standalone `nr_dlsim` loopback

`openair1/SIMULATION/NR_PHY/dlsim.c` calls the same `phy_procedures_gNB_TX()` and
`pdsch_processing()` functions directly, in a single process, with no rfsim networking
— useful for quick iteration on the PHY code itself. Build it with:

```bash
cd cmake_targets/ran_build/build
cmake --build . --target nr_dlsim -- -j16
```

Note: `nr_dlsim` has its own bespoke single-character `getopt` CLI parser
(`openair1/SIMULATION/NR_PHY/dlsim.c`), separate from the shared config-module options
`nr-softmodem`/`nr-uesoftmodem` use — the `--custom-re-*` long options above are **not**
recognized by `nr_dlsim` as-is. Wiring them in would need extending that parser; until
then, use the two-process rfsim flow above for anything involving these flags.

## Troubleshooting

- **`custom RE signal: no PDSCH scheduled in frame <f> slot <s> — nothing to
  overwrite`**: your `--custom-re-slot` doesn't get a PDSCH from your scheduler config.
  For `--phy-test` mode, the default PDSCH slot is controlled by `dlsch_slot_bitmap`
  (default `1<<1`, i.e. slot 1) and the `-D`/`-Dmod` CLI options
  (`openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_phytest.c`) — either match your
  `--custom-re-slot` to that, or pass `-D`/`-Dmod` to change which slot gets PDSCH.
- **`custom RE signal: symbol/subcarrier config doesn't fit this slot's PDSCH
  allocation (...)`**: the warning prints the actual allocation's symbol range,
  subcarrier range, and (gNB side) layer count it found — pick
  `--custom-re-symbol`/`--custom-re-startsc`/`--custom-re-numre` to fit inside those
  bounds, and make sure the chosen symbol isn't a DMRS-occasion symbol (see "Why a
  non-DMRS symbol" above).
- **No `CUSTOM_RE_*_CAPTURED` log lines but no skip warning either**: add
  `--log_config.global_log_level analysis` (the marker uses `LOG_A`, gated at the
  `analysis` level; the skip warnings use `LOG_W` and are visible regardless). The IQ
  dump files (`custom_tx_iq.m`/`custom_rx_iq.m`) are written unconditionally whenever
  the capture actually happens, regardless of log level.
- **UE never captures anything even though the gNB does**: the UE needs several frames
  to synchronize before PDSCH processing runs at all; with the default
  `--custom-re-frame 0` (every frame) this isn't a timing issue, but if you set a
  nonzero period, make sure it's large enough that the UE will still be running when
  the next multiple comes around.
- **`Assertion (k2 >= GET_DURATION_RX_TO_TX(...))` at UE startup in `--phy-test` mode**:
  unrelated to this feature — a config mismatch between the reference conf's
  `min_rxtxtime` and `--phy-test` mode's scheduling. Add
  `--gNBs.[0].min_rxtxtime 3` to the gNB command line.
