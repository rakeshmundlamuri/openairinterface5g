/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef __NR_CUSTOM_SIGNAL_H__
#define __NR_CUSTOM_SIGNAL_H__

#include "PHY/defs_nr_common.h"

// Test/debug feature: write a small block of known complex IQ values (loaded from a
// file) into a chosen, dedicated slot of the DL resource grid, and read them back on
// the UE side, so the whole DL OFDM TX/RX chain can be verified end to end (e.g. with
// rfsim). Not a 3GPP channel: this is a stepping stone for future custom-signal work.
typedef struct {
  bool enabled;
  int target_frame; // firing period in frames: fires when (frame % target_frame == 0); 0 means every frame
  int target_slot; // slot within the frame reserved for this signal
  int symbol; // OFDM symbol index within the slot
  int start_sc; // first subcarrier index (k), as a logical RE index from subcarrier 0 of the channel
                // (same convention as PSS/PBCH/PDCCH/PRS on TX; nr_extract_custom_signal() converts
                // it to the UE's raw FFT-bin rxdataF indexing internally, see its comment)
  int num_re; // number of consecutive REs starting at start_sc
  int ant; // antenna port index
  c16_t *iq; // num_re values loaded from file, already scaled to the TX amplitude convention
} nr_custom_signal_config_t;

// Parses a text file of "re,im" pairs (one per line, floats in [-1,1]) into cfg->iq,
// scaling each sample by amp via c16mulRealShift(). Returns 0 on success.
int nr_custom_signal_load_file(const char *path, int16_t amp, nr_custom_signal_config_t *cfg);

// Writes cfg->num_re values from cfg->iq into txdataF at (cfg->symbol, cfg->start_sc..+num_re).
// Call after all other DL channel generation for the slot so nothing else can overwrite these REs.
void nr_generate_custom_signal(c16_t *txdataF, const NR_DL_FRAME_PARMS *frame_parms, const nr_custom_signal_config_t *cfg);

// Reads cfg->num_re values from rxdataF at (cfg->symbol, cfg->start_sc..+num_re) into out (caller-allocated, cfg->num_re entries).
void nr_extract_custom_signal(const c16_t *rxdataF, const NR_DL_FRAME_PARMS *frame_parms, const nr_custom_signal_config_t *cfg, c16_t *out);

#endif
