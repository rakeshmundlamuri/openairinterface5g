#!/usr/bin/env python3
"""End-to-end Deep-JSCC-over-real-PDSCH demo.

image -> encode_image.py -> custom_iq.txt -> real gNB/UE rfsim (--phy-test, the same
proven flow as scripts/custom_re/run_rfsim_test.sh) -> custom_rx_iq.m -> decode_image.py
-> reconstructed image + PSNR.

Usage:
    run_demo.py --nr-softmodem PATH --nr-uesoftmodem PATH --gnb-conf PATH \
                --checkpoint PATH [--image PATH | --cifar-index N] \
                [--channel-type TYPE] [--out-dir DIR]
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Same defaults already proven in scripts/custom_re/run_rfsim_test.sh: slot 1 is where
# --phy-test's default scheduler (dlsch_slot_bitmap = 1<<1) places PDSCH; symbol 13 is a
# non-DMRS (pure data) symbol of that allocation; frame period 0 = fires every frame.
CUSTOM_RE_FRAME = 0
CUSTOM_RE_SLOT = 1
CUSTOM_RE_SYMBOL = 13
CUSTOM_RE_STARTSC = 0


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def write_channelmod_conf(path, channel_type):
    path.write_text(f"""channelmod = {{
  max_chan = 10;
  modellist = "modellist_rfsimu_1";
  modellist_rfsimu_1 = (
    {{ model_name = "rfsimu_channel_enB0"; type = "{channel_type}"; ploss_dB = 0;
      noise_power_dB = -10; forgetfact = 0; offset = 0; ds_tdl = 0; }},
    {{ model_name = "rfsimu_channel_ue0"; type = "{channel_type}"; ploss_dB = 0;
      noise_power_dB = -10; forgetfact = 0; offset = 0; ds_tdl = 0; }}
  );
}};
""")


def wait_for(predicate, timeout_s, poll_s=1.0, what=""):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    print(f"FAIL: timed out waiting for {what}", file=sys.stderr)
    return False


def log_has(path, marker):
    return path.exists() and marker in path.read_text(errors="ignore")


def run_rf_once(nr_softmodem, nr_uesoftmodem, gnb_conf, channel_type, custom_re_flags):
    """One attempt at launching gNB+UE and waiting for both captures. Returns
    (ok, gnb_dir, ue_dir). A fading channel (Rayleigh/Rice/...) can draw a bad enough
    realization that the UE never syncs at all - that's the channel model doing its
    job, not a bug (a real receiver would lose sync on a deep enough fade too) - so
    this is meant to be retried by the caller, with a fresh channel draw each time."""
    gnb_dir = Path(tempfile.mkdtemp(prefix="jscc_gnb_"))
    ue_dir = Path(tempfile.mkdtemp(prefix="jscc_ue_"))

    gnb_conf_use = gnb_conf
    channel_flags = []
    ue_conf_flags = []
    if channel_type:
        for f in gnb_conf.parent.glob("*.conf"):
            shutil.copy(f, gnb_dir / f.name)
        write_channelmod_conf(gnb_dir / "channelmod_rfsimu.conf", channel_type)
        write_channelmod_conf(ue_dir / "channelmod_rfsimu.conf", channel_type)
        gnb_conf_use = gnb_dir / gnb_conf.name
        with open(gnb_conf_use, "a") as f:
            f.write('\n@include "channelmod_rfsimu.conf"\n')
        ue_conf = ue_dir / "ue_chanmod.conf"
        ue_conf.write_text('@include "channelmod_rfsimu.conf"\n')
        ue_conf_flags = ["-O", str(ue_conf)]
        channel_flags = ["--rfsimulator.[0].options", "chanmod"]

    gnb_log = gnb_dir / "gnb.log"
    ue_log = ue_dir / "ue.log"
    gnb_proc = ue_proc = None
    try:
        print("starting gNB (phy-test mode, no core network needed)...")
        gnb_proc = subprocess.Popen(
            [str(nr_softmodem), "-O", str(gnb_conf_use), "--rfsim", "--phy-test",
             "--gNBs.[0].min_rxtxtime", "3", *channel_flags, *custom_re_flags,
             "--log_config.global_log_level", "analysis"],
            cwd=gnb_dir, stdout=open(gnb_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)

        if not wait_for(lambda: (gnb_dir / "reconfig.raw").exists(), 20, what="gNB reconfig.raw"):
            return False, gnb_dir, ue_dir

        print("starting UE...")
        ue_proc = subprocess.Popen(
            [str(nr_uesoftmodem), *ue_conf_flags, "--rfsim", "--phy-test",
             "--reconfig-file", str(gnb_dir / "reconfig.raw"), "--rbconfig-file", str(gnb_dir / "rbconfig.raw"),
             "--rfsimulator.[0].serveraddr", "127.0.0.1", *channel_flags, *custom_re_flags,
             "--log_config.global_log_level", "analysis"],
            cwd=ue_dir, stdout=open(ue_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)

        print("waiting for CUSTOM_RE_TX_CAPTURED / CUSTOM_RE_RX_CAPTURED markers...")
        ok = wait_for(lambda: log_has(gnb_log, "CUSTOM_RE_TX_CAPTURED") and log_has(ue_log, "CUSTOM_RE_RX_CAPTURED"),
                      60, what="capture markers")
        if not ok:
            if not log_has(gnb_log, "CUSTOM_RE_TX_CAPTURED"):
                print(f"gNB never captured - see {gnb_log}", file=sys.stderr)
            if not log_has(ue_log, "CUSTOM_RE_RX_CAPTURED"):
                print(f"UE never captured (sync failure is possible with fading channels - see {ue_log})",
                      file=sys.stderr)
        return ok, gnb_dir, ue_dir
    finally:
        # Each was launched with start_new_session=True, so it's its own process-group
        # leader - signal the whole group (not just the direct child) in case it spawned
        # any child processes of its own, and so any signal *it* sends to "its" group on
        # shutdown can't reach back to us or the caller's shell.
        for p in (gnb_proc, ue_proc):
            if p is not None and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in (gnb_proc, ue_proc):
            if p is not None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr-softmodem", required=True)
    ap.add_argument("--nr-uesoftmodem", required=True)
    ap.add_argument("--gnb-conf", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--cifar-index", type=int, default=0)
    ap.add_argument("--num-pilot", type=int, default=8)
    ap.add_argument("--channel-type", default=None, help="e.g. AWGN, Rayleigh8 (default: no channel model)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-retries", type=int, default=4, help="retries on RF sync failure "
                     "(fading channels can draw a bad enough realization that the UE never "
                     "syncs - that's the channel model doing its job, not a bug - so we just "
                     "redraw and try again, same as a real receiver would)")
    args = ap.parse_args()

    nr_softmodem = Path(args.nr_softmodem).resolve()
    nr_uesoftmodem = Path(args.nr_uesoftmodem).resolve()
    gnb_conf = Path(args.gnb_conf).resolve()

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="jscc_out_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir={out_dir}")

    # 1. Encode the image.
    run([sys.executable, str(SCRIPT_DIR / "encode_image.py"),
         "--checkpoint", args.checkpoint, "--out-dir", str(out_dir),
         *(["--image", args.image] if args.image else ["--cifar-index", str(args.cifar_index)]),
         "--num-pilot", str(args.num_pilot)])
    iq_file = out_dir / "custom_iq.txt"
    num_re = len(iq_file.read_text().splitlines())
    print(f"encoded payload: {num_re} REs total")

    custom_re_flags = ["--custom-re-enable", "--custom-re-iqfile", str(iq_file),
                        "--custom-re-frame", str(CUSTOM_RE_FRAME), "--custom-re-slot", str(CUSTOM_RE_SLOT),
                        "--custom-re-symbol", str(CUSTOM_RE_SYMBOL), "--custom-re-startsc", str(CUSTOM_RE_STARTSC),
                        "--custom-re-numre", str(num_re)]

    ok = False
    gnb_dir = ue_dir = None
    for attempt in range(1, args.max_retries + 1):
        print(f"--- RF attempt {attempt}/{args.max_retries} ---")
        ok, gnb_dir, ue_dir = run_rf_once(nr_softmodem, nr_uesoftmodem, gnb_conf, args.channel_type, custom_re_flags)
        if ok:
            break
        shutil.rmtree(gnb_dir, ignore_errors=True)
        shutil.rmtree(ue_dir, ignore_errors=True)
    if not ok:
        print(f"FAIL: no successful RF capture after {args.max_retries} attempts", file=sys.stderr)
        raise SystemExit(1)
    print(f"gnb_dir={gnb_dir} ue_dir={ue_dir}")

    rx_m = ue_dir / "custom_rx_iq.m"
    tx_m = gnb_dir / "custom_tx_iq.m"
    shutil.copy(rx_m, out_dir / "custom_rx_iq.m")
    shutil.copy(tx_m, out_dir / "custom_tx_iq.m")

    # 2. Decode the received capture back into an image.
    run([sys.executable, str(SCRIPT_DIR / "decode_image.py"),
         "--meta", str(out_dir / "meta.json"), "--rx-m", str(rx_m),
         "--out", str(out_dir / "reconstructed.png"), "--original", str(out_dir / "original.png")])

    print(f"\ndone. original: {out_dir / 'original.png'}  reconstructed: {out_dir / 'reconstructed.png'}")


if __name__ == "__main__":
    main()
