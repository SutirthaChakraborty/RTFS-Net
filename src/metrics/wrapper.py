###
# Author: Kai Li
# Date: 2021-06-22 12:41:36
# LastEditors: Kai Li
# LastEditTime: 2021-08-30 17:44:28
###
import csv
import torch
import logging
import numpy as np

from ..losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pit_sisnr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        self.pit_snr = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")

    def __call__(self, mix, clean, estimate, key):
        # sisnr
        sisnr = self.pit_sisnr(estimate.unsqueeze(0), clean.unsqueeze(0))
        mix = torch.stack([mix] * clean.shape[0], dim=0)
        sisnr_baseline = self.pit_sisnr(mix.unsqueeze(0), clean.unsqueeze(0))
        sisnr_i = sisnr - sisnr_baseline

        # sdr
        sdr = self.pit_snr(
            estimate.unsqueeze(0),
            clean.unsqueeze(0),
        )
        sdr_baseline = self.pit_snr(
            mix.unsqueeze(0),
            clean.unsqueeze(0),
        )
        sdr_i = sdr - sdr_baseline
        row = {
            "snt_id": key,
            "sdr": -sdr.item(),
            "sdr_i": -sdr_i.item(),
            "si-snr": -sisnr.item(),
            "si-snr_i": -sisnr_i.item(),
        }
        self.writer.writerow(row)
        # Metric Accumulation
        self.all_sdrs.append(-sdr.item())
        self.all_sdrs_i.append(-sdr_i.item())
        self.all_sisnrs.append(-sisnr.item())
        self.all_sisnrs_i.append(-sisnr_i.item())

    def get_mean(self):
        return {
            "sdr": np.array(self.all_sdrs).mean(),
            "sdr_i": np.array(self.all_sdrs_i).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "si-snr_i": np.array(self.all_sisnrs_i).mean(),
        }

    def get_std(self):
        return {
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std(),
        }

    def final(
        self,
    ):
        row = {
            "snt_id": "avg",
            "sdr": np.array(self.all_sdrs).mean(),
            "sdr_i": np.array(self.all_sdrs_i).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "si-snr_i": np.array(self.all_sisnrs_i).mean(),
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std(),
        }
        self.writer.writerow(row)
        self.results_csv.close()
