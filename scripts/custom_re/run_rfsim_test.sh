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
#   run_rfsim_test.sh <path_to_nr-softmodem> <path_to_nr-uesoftmodem> <gnb_conf_file> [--chanmod] [--channel-type <TYPE>]
#
# --chanmod enables rfsim's channel model instead of the default near-ideal passthrough.
# --channel-type <TYPE> additionally selects the model type (implies --chanmod); default
# AWGN. Valid TYPE values are rfsim's channelmod types, e.g. AWGN, Rayleigh8, Rayleigh1,
# Rice8, TDL_A, TDL_B, TDL_C, TDL_D, TDL_E, EPA, EVA, ETU (full list: CHANNELMOD_MAP_INIT
# in openair1/SIMULATION/TOOLS/sim.h). Real noise/fading means an occasional RE can
# exceed a tight comparison tolerance even when the feature is working correctly - this
# script loosens it automatically for any non-ideal channel (see doc/CUSTOM_SIGNAL_AI.md).

set -e

USAGE="usage: $0 <nr-softmodem> <nr-uesoftmodem> <gnb_conf_file> [--chanmod] [--channel-type <TYPE>]"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NR_SOFTMODEM=$(realpath "${1:?$USAGE}")
NR_UESOFTMODEM=$(realpath "${2:?$USAGE}")
GNB_CONF=$(realpath "${3:?$USAGE}")
shift 3

CHANMOD=0
CHANNEL_TYPE="AWGN"
while [ $# -gt 0 ]; do
  case "$1" in
    --chanmod) CHANMOD=1; shift ;;
    --channel-type)
      CHANMOD=1
      CHANNEL_TYPE="${2:?$USAGE (--channel-type needs a value)}"
      shift 2
      ;;
    *) echo "$USAGE" >&2; exit 1 ;;
  esac
done

GNB_DIR=$(mktemp -d)
UE_DIR=$(mktemp -d)
IQFILE="$GNB_DIR/custom_iq.txt"
COMPARE_TOL=0.1

if [ "$CHANMOD" = "1" ]; then
  CONF_DIR="$(dirname "$GNB_CONF")"
  # gnb_conf's own includes (e.g. neighbour-config-rfsim.conf) are relative to its
  # original directory, so copy the whole thing alongside our generated channelmod conf.
  cp "$CONF_DIR"/*.conf "$GNB_DIR/" 2>/dev/null || true

  CHANMOD_CONF_BODY="channelmod = {
  max_chan = 10;
  modellist = \"modellist_rfsimu_1\";
  modellist_rfsimu_1 = (
    { # DL, modify on UE side
      model_name     = \"rfsimu_channel_enB0\";
      type           = \"$CHANNEL_TYPE\";
      ploss_dB       = 0;
      noise_power_dB = -10;
      forgetfact     = 0;
      offset         = 0;
      ds_tdl         = 0;
    },
    { # UL, modify on gNB side
      model_name     = \"rfsimu_channel_ue0\";
      type           = \"$CHANNEL_TYPE\";
      ploss_dB       = 0;
      noise_power_dB = -10;
      forgetfact     = 0;
      offset         = 0;
      ds_tdl         = 0;
    }
  );
};"
  echo "$CHANMOD_CONF_BODY" >"$GNB_DIR/channelmod_rfsimu.conf"
  echo "$CHANMOD_CONF_BODY" >"$UE_DIR/channelmod_rfsimu.conf"

  GNB_CONF_CHANMOD="$GNB_DIR/$(basename "$GNB_CONF")"
  printf '\n@include "channelmod_rfsimu.conf"\n' >>"$GNB_CONF_CHANMOD"
  GNB_CONF="$GNB_CONF_CHANMOD"
  UE_CONF_CHANMOD="$UE_DIR/ue_chanmod.conf"
  printf '@include "channelmod_rfsimu.conf"\n' >"$UE_CONF_CHANMOD"

  # AWGN measures ~1% error in practice; anything with real fading (Rayleigh/Rice/TDL/...)
  # measures noticeably more (a single random channel draw can occasionally fade hard on
  # a given RE) - loosen the tolerance accordingly rather than false-FAIL a working run.
  if [ "$CHANNEL_TYPE" = "AWGN" ]; then
    COMPARE_TOL=0.15
  else
    COMPARE_TOL=0.2
  fi
fi

# Firing period 0 = every frame, so we don't need to guess when the gNB/UE will first
# reach a specific frame number: it fires on every occurrence of the chosen slot. Slot 1
# is where --phy-test's default scheduler (dlsch_slot_bitmap = 1<<1) actually places
# PDSCH; symbol 13 is a non-DMRS (pure data) symbol of that allocation.
CUSTOM_RE_FRAME=0
CUSTOM_RE_SLOT=1
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

CHANMOD_FLAG=""
[ "$CHANMOD" = "1" ] && CHANMOD_FLAG="--rfsimulator.[0].options chanmod"

echo "starting gNB (phy-test mode, no core network needed)..."
(cd "$GNB_DIR" && exec "$NR_SOFTMODEM" -O "$GNB_CONF" --rfsim --phy-test --gNBs.[0].min_rxtxtime 3 $CHANMOD_FLAG $CUSTOM_RE_FLAGS) >"$GNB_DIR/gnb.log" 2>&1 &
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

UE_CONF_FLAG=""
[ "$CHANMOD" = "1" ] && UE_CONF_FLAG="-O $UE_CONF_CHANMOD"

echo "starting UE..."
(cd "$UE_DIR" && exec "$NR_UESOFTMODEM" $UE_CONF_FLAG --rfsim --phy-test \
    --reconfig-file "$GNB_DIR/reconfig.raw" --rbconfig-file "$GNB_DIR/rbconfig.raw" \
    --rfsimulator.[0].serveraddr 127.0.0.1 $CHANMOD_FLAG $CUSTOM_RE_FLAGS) >"$UE_DIR/ue.log" 2>&1 &
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
python3 "$SCRIPT_DIR/compare_iq.py" "$IQFILE" "$GNB_DIR/custom_tx_iq.m" "$COMPARE_TOL" || exit 1

echo
echo "comparing UE RX capture against $IQFILE..."
python3 "$SCRIPT_DIR/compare_iq.py" "$IQFILE" "$UE_DIR/custom_rx_iq.m" "$COMPARE_TOL"
