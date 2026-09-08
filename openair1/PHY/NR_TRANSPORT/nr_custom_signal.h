/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef __NR_CUSTOM_SIGNAL_H__
#define __NR_CUSTOM_SIGNAL_H__

#include "PHY/defs_nr_common.h"

// Test/debug feature: overwrite a small block of PDSCH data REs (loaded from a file)
// on a non-DMRS symbol of a real PDSCH allocation, and read them back on the UE side
// after channel estimation/compensation, so the whole DL OFDM TX/RX chain - including
// real DMRS-based channel estimation - can be verified end to end (e.g. with rfsim).
// Not a 3GPP channel: this is a stepping stone for future custom-signal work.
typedef struct {
  bool enabled;
  int target_frame; // firing period in frames: fires when (frame % target_frame == 0); 0 means every frame
  int target_slot; // slot within the frame reserved for this signal
  int symbol; // OFDM symbol index within the slot; must be a non-DMRS symbol of the PDSCH allocation
  int start_sc; // first subcarrier index (k), as a logical RE index from subcarrier 0 of the channel
                // (same convention as PSS/PBCH/PDCCH/PRS/PDSCH on TX); must fall inside the PDSCH
                // allocation's RB range - callers validate this against the live allocation before use
  int num_re; // number of consecutive REs starting at start_sc
  int ant; // antenna port index
  c16_t *iq; // num_re values loaded from file, already scaled to the TX amplitude convention
} nr_custom_signal_config_t;

// Parses a text file of "re,im" pairs (one per line, floats in [-1,1]) into cfg->iq,
// scaling each sample by amp via c16mulRealShift(). Returns 0 on success.
int nr_custom_signal_load_file(const char *path, int16_t amp, nr_custom_signal_config_t *cfg);

// Writes cfg->num_re values from cfg->iq into txdataF at (cfg->symbol, cfg->start_sc..+num_re),
// overwriting whatever PDSCH data nr_generate_pdsch() just wrote there. Call after PDSCH
// generation for the slot so nothing else can overwrite these REs; caller must have already
// validated (symbol, start_sc, num_re) against the live PDSCH allocation.
void nr_generate_custom_signal(c16_t *txdataF, const NR_DL_FRAME_PARMS *frame_parms, const nr_custom_signal_config_t *cfg);

// Reads cfg->num_re values starting at local index j out of a channel-compensated PDSCH
// symbol buffer (i.e. rxdataF_comp for one (symbol, layer), already offset by the caller)
// into out (caller-allocated, cfg->num_re entries). j is the caller-computed offset of
// cfg->start_sc within the PDSCH allocation (valid only because cfg->symbol carries no DMRS,
// so every RE in the allocation on that symbol is data - see caller for the derivation).
void nr_extract_custom_signal(const c16_t *rxdataF_comp_symbol, int j, const nr_custom_signal_config_t *cfg, c16_t *out);

#endif
