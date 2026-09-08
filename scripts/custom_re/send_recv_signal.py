#!/usr/bin/env python3
"""Send an arbitrary complex IQ signal over the real gNB->rfsim->UE PDSCH pipeline and
get back what was received - automatically finding a (slot, symbol, start_sc) placement
that actually fits the live PDSCH allocation, instead of requiring it to be hand-picked.

The custom-RE-signal feature (openair1/PHY/NR_TRANSPORT/nr_custom_signal.{h,c}) already
validates any chosen placement against the live scheduled PDSCH allocation and logs the
real bounds when it doesn't fit (see doc/CUSTOM_SIGNAL_AI.md). This script exploits that:
it makes a cheap gNB-only "discovery" attempt with a starting guess, parses the actual
allocation bounds out of the log when the guess is rejected, and narrows in on a placement
that fits - then does the real gNB+UE send/receive with that placement (with retry on
fading-channel sync failure, same as scripts/jscc/run_demo.py).

Only fits within a single OFDM symbol - a signal longer than one symbol's available REs
raises a clear error rather than silently splitting across symbols/slots.

Usage (as a script):
    send_recv_signal.py --nr-softmodem PATH --nr-uesoftmodem PATH --gnb-conf PATH \
        (--iqfile FILE | --length N) [--channel-type TYPE] [--out-dir DIR]

Usage (as a library):
    from send_recv_signal import send_recv
    received = send_recv(iq_values, nr_softmodem, nr_uesoftmodem, gnb_conf)  # -> list[complex]
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

CUSTOM_RE_FRAME = 0  # fire every frame - see doc/CUSTOM_SIGNAL_AI.md
CANDIDATE_SLOTS = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]  # slot 1 is --phy-test's default PDSCH slot; rest are fallbacks
SYMBOLS_PER_SLOT = 14

_SKIP_RE = re.compile(
    r"symbol (\d+), sc (\d+)\.\.(\d+), alloc symbols (\d+)\.\.(\d+) sc (\d+)\.\.(\d+)")
_NO_PDSCH_RE = re.compile(r"no PDSCH scheduled in frame (\d+) slot (\d+)")
_M_LINE_RE = re.compile(r"(-?\d+)\s*\+\s*j\*\((-?\d+)\)")


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


def wait_for(predicate, timeout_s, poll_s=0.5):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def log_text(path):
    return path.read_text(errors="ignore") if path.exists() else ""


def start_proc(argv, cwd, log_path):
    return subprocess.Popen(argv, cwd=cwd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                             start_new_session=True)


def stop_proc(p):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def custom_re_flags(iq_file, num_re, slot, symbol, start_sc):
    return ["--custom-re-enable", "--custom-re-iqfile", str(iq_file),
            "--custom-re-frame", str(CUSTOM_RE_FRAME), "--custom-re-slot", str(slot),
            "--custom-re-symbol", str(symbol), "--custom-re-startsc", str(start_sc),
            "--custom-re-numre", str(num_re)]


def find_placement(nr_softmodem, gnb_conf, gnb_conf_args, iq_file, num_re, work_dir, max_probes=8):
    """gNB-only discovery: find a (slot, symbol, start_sc) that fits num_re REs, by reading
    back the live allocation bounds this feature already reports when a guess is rejected.
    Returns (slot, symbol, start_sc) or raises RuntimeError."""
    for slot in CANDIDATE_SLOTS:
        symbol_candidates = [13]  # known-good default (non-DMRS symbol in typical configs) - tried first
        start_sc = 0
        tried_symbols = set()
        for _probe in range(max_probes):
            if not symbol_candidates:
                break
            symbol = symbol_candidates.pop(0)
            if symbol in tried_symbols or symbol < 0 or symbol >= SYMBOLS_PER_SLOT:
                continue
            tried_symbols.add(symbol)

            probe_dir = Path(tempfile.mkdtemp(prefix="probe_gnb_"))
            log_path = probe_dir / "gnb.log"
            proc = start_proc(
                [str(nr_softmodem), "-O", str(gnb_conf), "--rfsim", "--phy-test",
                 "--gNBs.[0].min_rxtxtime", "3", *gnb_conf_args,
                 *custom_re_flags(iq_file, num_re, slot, symbol, start_sc),
                 "--log_config.global_log_level", "analysis"],
                probe_dir, log_path)
            try:
                got = wait_for(lambda: ("CUSTOM_RE_TX_CAPTURED" in log_text(log_path))
                                or _SKIP_RE.search(log_text(log_path))
                                or _NO_PDSCH_RE.search(log_text(log_path)), timeout_s=15)
                text = log_text(log_path)
            finally:
                stop_proc(proc)
                shutil.rmtree(probe_dir, ignore_errors=True)

            if not got:
                continue  # slow slot / nothing useful logged yet - move on to next candidate
            if "CUSTOM_RE_TX_CAPTURED" in text:
                print(f"found placement: slot={slot} symbol={symbol} start_sc={start_sc}")
                return slot, symbol, start_sc
            m = _NO_PDSCH_RE.search(text)
            if m:
                break  # this slot never gets a PDSCH at all - try the next slot
            m = _SKIP_RE.search(text)
            if m:
                _, _, _, alloc_sym_lo, alloc_sym_hi, alloc_sc_lo, alloc_sc_hi = (int(g) for g in m.groups())
                if num_re > (alloc_sc_hi - alloc_sc_lo):
                    break  # signal doesn't fit this allocation's subcarrier range at all - try next slot
                start_sc = alloc_sc_lo
                if not symbol_candidates:
                    # step backwards from the top of the allocation's symbol range - the last
                    # symbol is most often the first one tried and non-DMRS in typical configs
                    symbol_candidates = list(range(alloc_sym_hi - 1, alloc_sym_lo - 1, -1))
    raise RuntimeError(f"could not find a placement for a {num_re}-RE signal in any of {CANDIDATE_SLOTS}")


def parse_m_file(path):
    values = []
    with open(path) as f:
        for line in f:
            m = _M_LINE_RE.search(line)
            if m:
                values.append(complex(int(m.group(1)), int(m.group(2))))
    return values


def parse_m_file_stable(path, expected_len, tries=20, delay=0.05):
    """LOG_M truncates-then-rewrites this file on every firing (format=1 uses "w+"), and
    with --custom-re-frame 0 that's roughly every ~10ms - so reading it exactly when the
    capture marker first appears in the log can catch it mid-write (briefly empty/partial).
    Retry a few times; the content is identical every firing for a static payload, so any
    complete snapshot is as good as any other."""
    last_len = -1
    for _ in range(tries):
        values = parse_m_file(path)
        if len(values) == expected_len:
            return values
        last_len = len(values)
        time.sleep(delay)
    raise RuntimeError(f"{path} never stabilized at {expected_len} REs (last read: {last_len})")


def send_recv(iq_values, nr_softmodem, nr_uesoftmodem, gnb_conf, channel_type=None, max_retries=4,
              work_dir=None):
    """Send iq_values (list of complex, magnitude expected in [-1,1]) over the real
    gNB->rfsim->UE PDSCH pipeline. Returns the raw received complex values (same length,
    post-equalization, NOT descaled/calibrated - do that with your own pilot/reference,
    same as scripts/jscc/decode_image.py does)."""
    nr_softmodem = Path(nr_softmodem).resolve()
    nr_uesoftmodem = Path(nr_uesoftmodem).resolve()
    gnb_conf = Path(gnb_conf).resolve()
    num_re = len(iq_values)

    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="send_recv_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    iq_file = work_dir / "custom_iq.txt"
    with open(iq_file, "w") as f:
        for v in iq_values:
            f.write(f"{v.real:.6f},{v.imag:.6f}\n")

    gnb_conf_args = []
    channel_flag = []
    ue_conf_args = []
    if channel_type:
        probe_conf_dir = work_dir / "chanmod"
        probe_conf_dir.mkdir(exist_ok=True)
        for f in gnb_conf.parent.glob("*.conf"):
            shutil.copy(f, probe_conf_dir / f.name)
        write_channelmod_conf(probe_conf_dir / "channelmod_rfsimu.conf", channel_type)
        gnb_conf_chanmod = probe_conf_dir / gnb_conf.name
        with open(gnb_conf_chanmod, "a") as f:
            f.write('\n@include "channelmod_rfsimu.conf"\n')
        gnb_conf = gnb_conf_chanmod
        ue_conf = probe_conf_dir / "ue_chanmod.conf"
        ue_conf.write_text('@include "channelmod_rfsimu.conf"\n')
        ue_conf_args = ["-O", str(ue_conf)]
        channel_flag = ["--rfsimulator.[0].options", "chanmod"]

    print(f"discovering a placement for {num_re} REs...")
    slot, symbol, start_sc = find_placement(nr_softmodem, gnb_conf, gnb_conf_args, iq_file, num_re, work_dir)
    flags = custom_re_flags(iq_file, num_re, slot, symbol, start_sc)

    for attempt in range(1, max_retries + 1):
        print(f"--- send/receive attempt {attempt}/{max_retries} ---")
        gnb_dir = Path(tempfile.mkdtemp(prefix="sr_gnb_"))
        ue_dir = Path(tempfile.mkdtemp(prefix="sr_ue_"))
        gnb_log = gnb_dir / "gnb.log"
        ue_log = ue_dir / "ue.log"
        gnb_proc = ue_proc = None
        try:
            gnb_proc = start_proc(
                [str(nr_softmodem), "-O", str(gnb_conf), "--rfsim", "--phy-test",
                 "--gNBs.[0].min_rxtxtime", "3", *gnb_conf_args, *channel_flag, *flags,
                 "--log_config.global_log_level", "analysis"],
                gnb_dir, gnb_log)
            if not wait_for(lambda: (gnb_dir / "reconfig.raw").exists(), 20):
                print("gNB never wrote reconfig.raw", file=sys.stderr)
                continue
            ue_proc = start_proc(
                [str(nr_uesoftmodem), *ue_conf_args, "--rfsim", "--phy-test",
                 "--reconfig-file", str(gnb_dir / "reconfig.raw"), "--rbconfig-file", str(gnb_dir / "rbconfig.raw"),
                 "--rfsimulator.[0].serveraddr", "127.0.0.1", *channel_flag, *flags,
                 "--log_config.global_log_level", "analysis"],
                ue_dir, ue_log)
            ok = wait_for(lambda: "CUSTOM_RE_TX_CAPTURED" in log_text(gnb_log)
                          and "CUSTOM_RE_RX_CAPTURED" in log_text(ue_log), 60)
            if not ok:
                print(f"capture failed this attempt (gNB log: {gnb_log}, UE log: {ue_log})", file=sys.stderr)
                continue
            try:
                sent = parse_m_file_stable(gnb_dir / "custom_tx_iq.m", num_re)
                received = parse_m_file_stable(ue_dir / "custom_rx_iq.m", num_re)
            except RuntimeError as e:
                print(f"{e} - retrying", file=sys.stderr)
                continue
            shutil.copy(gnb_dir / "custom_tx_iq.m", work_dir / "custom_tx_iq.m")
            shutil.copy(ue_dir / "custom_rx_iq.m", work_dir / "custom_rx_iq.m")
            return received
        finally:
            stop_proc(gnb_proc)
            stop_proc(ue_proc)
            shutil.rmtree(gnb_dir, ignore_errors=True)
            shutil.rmtree(ue_dir, ignore_errors=True)
    raise RuntimeError(f"no successful capture after {max_retries} attempts (see {work_dir})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nr-softmodem", required=True)
    ap.add_argument("--nr-uesoftmodem", required=True)
    ap.add_argument("--gnb-conf", required=True)
    ap.add_argument("--iqfile", default=None, help="existing 're,im' per line IQ file (length auto-detected)")
    ap.add_argument("--length", type=int, default=None, help="generate a test signal of this many REs instead")
    ap.add_argument("--channel-type", default=None, help="e.g. AWGN, Rayleigh8 (default: no channel model)")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if not args.iqfile and not args.length:
        ap.error("need either --iqfile or --length")

    if args.iqfile:
        lines = Path(args.iqfile).read_text().splitlines()
        iq_values = [complex(*map(float, l.split(","))) for l in lines if l.strip()]
    else:
        sys.path.insert(0, str(SCRIPT_DIR))
        from pathlib import Path as _P
        import importlib.util
        spec = importlib.util.spec_from_file_location("gen_iq_file", SCRIPT_DIR / "gen_iq_file.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        iq_values = [gen.QPSK_SYMBOLS[i % len(gen.QPSK_SYMBOLS)] for i in range(args.length)]

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="send_recv_out_"))
    received = send_recv(iq_values, args.nr_softmodem, args.nr_uesoftmodem, args.gnb_conf,
                          channel_type=args.channel_type, max_retries=args.max_retries, work_dir=out_dir)

    print(f"\nsent {len(iq_values)} values, received {len(received)} values (dumped in {out_dir}):")
    for i, (s, r) in enumerate(zip(iq_values, received)):
        print(f"  RE {i}: sent={s:.4f} received={r}")


if __name__ == "__main__":
    main()
