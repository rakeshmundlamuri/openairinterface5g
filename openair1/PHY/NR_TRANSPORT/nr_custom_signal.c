/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "PHY/NR_TRANSPORT/nr_custom_signal.h"
#include "PHY/TOOLS/tools_defs.h"
#include "common/utils/LOG/log.h"
#include <stdio.h>
#include <stdlib.h>

int nr_custom_signal_load_file(const char *path, int16_t amp, nr_custom_signal_config_t *cfg)
{
  FILE *f = fopen(path, "r");
  if (!f) {
    LOG_E(PHY, "custom RE signal: could not open IQ file %s\n", path);
    return -1;
  }

  c16_t *iq = malloc16(cfg->num_re * sizeof(*iq));
  AssertFatal(iq != NULL, "custom RE signal: out of memory allocating %d IQ samples\n", cfg->num_re);

  int n = 0;
  char line[256];
  while (n < cfg->num_re && fgets(line, sizeof(line), f)) {
    float re, im;
    if (sscanf(line, "%f,%f", &re, &im) != 2)
      continue;
    c16_t sample = {.r = (int16_t)(re * 32767), .i = (int16_t)(im * 32767)};
    iq[n] = c16mulRealShift(sample, amp, 15);
    n++;
  }
  fclose(f);

  if (n != cfg->num_re) {
    LOG_E(PHY, "custom RE signal: IQ file %s has %d usable samples, expected %d\n", path, n, cfg->num_re);
    free(iq);
    return -1;
  }

  cfg->iq = iq;
  LOG_I(PHY, "custom RE signal: loaded %d IQ samples from %s\n", n, path);
  return 0;
}

void nr_generate_custom_signal(c16_t *txdataF, const NR_DL_FRAME_PARMS *frame_parms, const nr_custom_signal_config_t *cfg)
{
  c16_t *tx = txdataF + cfg->symbol * frame_parms->ofdm_symbol_size;
  for (int i = 0; i < cfg->num_re; i++)
    tx[cfg->start_sc + i] = cfg->iq[i];
}

void nr_extract_custom_signal(const c16_t *rxdataF_comp_symbol, int j, const nr_custom_signal_config_t *cfg, c16_t *out)
{
  for (int i = 0; i < cfg->num_re; i++)
    out[i] = rxdataF_comp_symbol[j + i];
}
