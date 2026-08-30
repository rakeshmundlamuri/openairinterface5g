#!/bin/bash
# Manual rfsim test for the custom RE signal feature: launches gNB and UE softmodems
# in --phy-test mode over rfsim (no core network needed), with matching --custom-re-*
# flags, waits for both to log a CUSTOM_RE_*_CAPTURED marker, then compares the IQ
# they dumped. Validated end-to-end against gnb.sa.band78.fr1.106PRB.pci0.rfsim.conf.
#
# Run from the build directory containing nr-softmodem and nr-uesoftmodem
# (e.g. cmake_targets/ran_build/build/), after building with:
#   ./build_oai --gNB --nrUE --ninja
#
# Usage:
#   run_rfsim_test.sh <path_to_nr-softmodem> <path_to_nr-uesoftmodem> <gnb_conf_file>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NR_SOFTMODEM=$(realpath "${1:?usage: $0 <nr-softmodem> <nr-uesoftmodem> <gnb_conf_file>}")
NR_UESOFTMODEM=$(realpath "${2:?usage: $0 <nr-softmodem> <nr-uesoftmodem> <gnb_conf_file>}")
GNB_CONF=$(realpath "${3:?usage: $0 <nr-softmodem> <nr-uesoftmodem> <gnb_conf_file>}")

GNB_DIR=$(mktemp -d)
UE_DIR=$(mktemp -d)
IQFILE="$GNB_DIR/custom_iq.txt"

# Firing period 0 = every frame, so we don't need to guess when the gNB/UE will first
# reach a specific frame number: it fires on every occurrence of the chosen slot, on
# whichever DL slot the TDD pattern actually gives us (slot 6 is DL in the reference conf).
CUSTOM_RE_FRAME=0
CUSTOM_RE_SLOT=6
CUSTOM_RE_SYMBOL=13
CUSTOM_RE_STARTSC=0
CUSTOM_RE_NUMRE=12

CUSTOM_RE_FLAGS="--custom-re-enable --custom-re-iqfile $IQFILE \
  --custom-re-frame $CUSTOM_RE_FRAME --custom-re-slot $CUSTOM_RE_SLOT \
  --custom-re-symbol $CUSTOM_RE_SYMBOL --custom-re-startsc $CUSTOM_RE_STARTSC \
  --custom-re-numre $CUSTOM_RE_NUMRE"

echo "generating IQ file..."
python3 "$SCRIPT_DIR/gen_iq_file.py" "$IQFILE" "$CUSTOM_RE_NUMRE"

cleanup() {
  echo "stopping gNB/UE..."
  [ -n "${GNB_PID:-}" ] && kill "$GNB_PID" 2>/dev/null
  [ -n "${UE_PID:-}" ] && kill "$UE_PID" 2>/dev/null
}
trap cleanup EXIT

echo "starting gNB (phy-test mode, no core network needed)..."
(cd "$GNB_DIR" && exec "$NR_SOFTMODEM" -O "$GNB_CONF" --rfsim --phy-test --gNBs.[0].min_rxtxtime 3 $CUSTOM_RE_FLAGS) >"$GNB_DIR/gnb.log" 2>&1 &
GNB_PID=$!

echo "waiting for gNB to write rbconfig.raw/reconfig.raw..."
for _ in $(seq 1 20); do
  [ -f "$GNB_DIR/reconfig.raw" ] && break
  sleep 1
done
if [ ! -f "$GNB_DIR/reconfig.raw" ]; then
  echo "FAIL: gNB never wrote reconfig.raw (see $GNB_DIR/gnb.log)"
  exit 1
fi

echo "starting UE..."
(cd "$UE_DIR" && exec "$NR_UESOFTMODEM" --rfsim --phy-test \
    --reconfig-file "$GNB_DIR/reconfig.raw" --rbconfig-file "$GNB_DIR/rbconfig.raw" \
    --rfsimulator.serveraddr 127.0.0.1 $CUSTOM_RE_FLAGS) >"$UE_DIR/ue.log" 2>&1 &
UE_PID=$!

echo "waiting for CUSTOM_RE_TX_CAPTURED / CUSTOM_RE_RX_CAPTURED markers (up to 60s)..."
for _ in $(seq 1 60); do
  grep -q "CUSTOM_RE_TX_CAPTURED" "$GNB_DIR/gnb.log" 2>/dev/null && grep -q "CUSTOM_RE_RX_CAPTURED" "$UE_DIR/ue.log" 2>/dev/null && break
  sleep 1
done

if ! grep -q "CUSTOM_RE_TX_CAPTURED" "$GNB_DIR/gnb.log" 2>/dev/null; then
  echo "FAIL: gNB never logged CUSTOM_RE_TX_CAPTURED (see $GNB_DIR/gnb.log)"
  exit 1
fi
if ! grep -q "CUSTOM_RE_RX_CAPTURED" "$UE_DIR/ue.log" 2>/dev/null; then
  echo "FAIL: UE never logged CUSTOM_RE_RX_CAPTURED (see $UE_DIR/ue.log)"
  exit 1
fi

echo "comparing gNB TX capture against $IQFILE..."
python3 "$SCRIPT_DIR/compare_iq.py" "$IQFILE" "$GNB_DIR/custom_tx_iq.m" || exit 1

echo
echo "comparing UE RX capture against $IQFILE..."
python3 "$SCRIPT_DIR/compare_iq.py" "$IQFILE" "$UE_DIR/custom_rx_iq.m"
