<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Custom RE test signal

[[_TOC_]]

## What this is

A debug/test feature for the NR PHY that:

1. reserves a small, fixed set of resource elements (REs) in a chosen downlink slot,
2. fills them with known complex IQ values loaded from a plain text file, and
3. lets the UE read those same REs back, so the whole downlink OFDM TX/RX chain
   (resource-grid mapping, OFDM modulation, rfsim channel, FFT, RE extraction) can be
   verified end to end without needing PDSCH/decoding to succeed.

It is not a 3GPP channel or reference signal — it's a stepping stone for future
custom-signal work, and a standalone tool for sanity-checking the DL grid pipeline
(e.g. after touching PHY resource-mapping code, or when bringing up a new RF/channel
path).

The core logic lives in `openair1/PHY/NR_TRANSPORT/nr_custom_signal.{h,c}`:

- `nr_custom_signal_load_file()` — parses the IQ file into memory, scaled to the gNB's
  TX amplitude convention.
- `nr_generate_custom_signal()` — writes those values into `txdataF` at the configured
  slot/symbol/subcarriers. Called from `phy_procedures_gNB_TX()`
  (`openair1/SCHED_NR/phy_procedures_nr_gNB.c`) **after** all standard DL channel
  generation for the slot (PSS/SSS/PBCH/PDCCH/PDSCH/CSI-RS/PRS), so nothing else in
  that slot can overwrite the reserved REs.
- `nr_extract_custom_signal()` — reads the same REs back out of `rxdataF` on the UE
  side. Called from `pdsch_processing()`
  (`openair1/SCHED_NR_UE/phy_procedures_nr_ue.c`).

Both hooks are gated behind `--custom-re-enable` and are no-ops otherwise.

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
convention used by PSS/SSS/PBCH/PRS, so the injected signal sits at a comparable power
level to the rest of the downlink.

## CLI options

Shared between `nr-softmodem` and `nr-uesoftmodem` (both must be given the **same**
values so they agree on where to write/read):

| Option | Meaning | Default |
|---|---|---|
| `--custom-re-enable` | turn the feature on | off |
| `--custom-re-iqfile <path>` | path to the IQ file (gNB loads it; UE just needs it for symmetry/logging) | — |
| `--custom-re-frame <n>` | firing period in frames: fires when `frame % n == 0`; **`0` means every frame** | `0` |
| `--custom-re-slot <n>` | slot within the frame to use — must be a DL slot in your TDD/FDD config | `0` |
| `--custom-re-symbol <n>` | OFDM symbol index within the slot | `13` |
| `--custom-re-startsc <n>` | first subcarrier (logical RE index from subcarrier 0 of the channel) | `0` |
| `--custom-re-numre <n>` | number of consecutive REs | `12` |

Pick `--custom-re-slot`/`--custom-re-symbol` so they land in a DL slot your scheduler
config actually uses — the write-after-everything-else ordering means it's safe even if
PDSCH/PDCCH share the slot, but don't point it at an UL slot in a TDD config or nothing
will ever fire on the gNB side.

`--custom-re-frame 0` (the default) is deliberately "every frame, not a single exact
frame": the UE takes several frames to synchronize, so a one-shot exact-frame match is
easy to miss entirely by the time the UE is actually running `pdsch_processing()`. With
period `0`, the signal is present continuously on the chosen slot, so there's no timing
race to get right — just start the UE and it will land on the very next occurrence.

## Logs and dumps produced

When the condition fires:

- gNB: logs `CUSTOM_RE_TX_CAPTURED frame <f> slot <s>` and writes `custom_tx_iq.m`
  (in its current working directory) with the values it just wrote.
- UE: logs `CUSTOM_RE_RX_CAPTURED frame <f> slot <s>` and writes `custom_rx_iq.m` with
  the raw (pre-channel-equalization) values it read back.

Both markers use `LOG_A` (analysis level) — pass `--log_config.global_log_level
analysis` if you don't otherwise see them.

`custom_tx_iq.m`/`custom_rx_iq.m` are plain MATLAB/Octave-style dumps (one complex value
per line, `re + j*(im)`), the same format `LOG_M` uses elsewhere in the codebase.

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

Example output:

```
comparing gNB TX capture against /tmp/tmp.xxx/custom_iq.txt...
RE 0: original=0.7071+0.7071j captured=(366+366j) descaled=0.7061+0.7061j rel_err=0.001
...
scale removed a=518.309+0j, max relative error=0.001 (tolerance=0.1)
PASS

comparing UE RX capture against /tmp/tmp.xxx/custom_iq.txt...
RE 0: original=0.7071+0.7071j captured=(364+362j) descaled=0.7060+0.7008j rel_err=0.006
...
scale removed a=516.07+0.471404j, max relative error=0.013 (tolerance=0.1)
PASS
```

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
  --custom-re-frame 0 --custom-re-slot 6 --custom-re-symbol 13 --custom-re-startsc 0 --custom-re-numre 12 \
  --log_config.global_log_level analysis
```

Wait until `rbconfig.raw` and `reconfig.raw` appear in `/tmp/gnb_run/` — `--phy-test`
writes these instead of doing a real RRC exchange, and the UE needs to read them.

**Terminal 2 — start the UE**, pointing at those two files:

```bash
mkdir -p /tmp/ue_run && cd /tmp/ue_run
/path/to/cmake_targets/ran_build/build/nr-uesoftmodem \
  --rfsim --phy-test \
  --reconfig-file /tmp/gnb_run/reconfig.raw --rbconfig-file /tmp/gnb_run/rbconfig.raw \
  --rfsimulator.serveraddr 127.0.0.1 \
  --custom-re-enable --custom-re-iqfile /tmp/custom_iq.txt \
  --custom-re-frame 0 --custom-re-slot 6 --custom-re-symbol 13 --custom-re-startsc 0 --custom-re-numre 12 \
  --log_config.global_log_level analysis
```

You should see `CUSTOM_RE_TX_CAPTURED` scroll by in terminal 1 and
`CUSTOM_RE_RX_CAPTURED` in terminal 2 within a second or two of the UE syncing (repeating
every frame, since the period is `0`).

**Terminal 3 — compare** (once both files exist):

```bash
python3 scripts/custom_re/compare_iq.py /tmp/custom_iq.txt /tmp/gnb_run/custom_tx_iq.m
python3 scripts/custom_re/compare_iq.py /tmp/custom_iq.txt /tmp/ue_run/custom_rx_iq.m
```

Each prints per-RE `original`/`captured`/`descaled`/`rel_err` and a final
`PASS`/`FAIL`. `compare_iq.py` fits a single complex scale factor (removing the gNB's
TX amplitude scaling and any channel/rotation picked up along the way) before
comparing, so what you see in the `descaled` column is directly comparable to
`custom_iq.txt`'s original values.

Ctrl-C both processes when done.

### 4. Faster iteration: standalone `nr_dlsim` loopback

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

- **No `CUSTOM_RE_*_CAPTURED` log lines**: add `--log_config.global_log_level analysis`
  (the marker uses `LOG_A`, gated at the `analysis` level). The IQ dump files
  (`custom_tx_iq.m`/`custom_rx_iq.m`) are written unconditionally regardless of log
  level, so their absence means the condition genuinely never fired — double check
  `--custom-re-slot` is a DL slot in your config's TDD/FDD pattern.
- **UE never captures anything even though the gNB does**: the UE needs several frames
  to synchronize before `pdsch_processing()` runs at all; with the default
  `--custom-re-frame 0` (every frame) this isn't a timing issue, but if you set a
  nonzero period, make sure it's large enough that the UE will still be running when
  the next multiple comes around, not so small/precise that sync delay skips it once.
- **`Assertion (k2 >= GET_DURATION_RX_TO_TX(...))` at UE startup in `--phy-test` mode**:
  unrelated to this feature — a config mismatch between the reference conf's
  `min_rxtxtime` and `--phy-test` mode's scheduling. Add
  `--gNBs.[0].min_rxtxtime 3` to the gNB command line.
