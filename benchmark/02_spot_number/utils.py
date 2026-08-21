from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load_h5ad(path):
    data = ad.read_h5ad(path)
    data.var_names_make_unique()
    if "Truth" not in data.obs:
        raise KeyError(f"{path} does not contain obs['Truth']")
    if "spatial" not in data.obsm:
        raise KeyError(f"{path} does not contain obsm['spatial']")
    if "counts" in data.layers:
        data.X = data.layers["counts"].copy()
    data.obs["ground_truth"] = data.obs["Truth"].astype("category")
    return data


def calculate_metrics(truth, prediction):
    truth = pd.Series(truth).reset_index(drop=True)
    prediction = pd.Series(prediction).reset_index(drop=True)
    valid = truth.notna() & prediction.notna()
    y_true = truth[valid].astype(str)
    y_pred = prediction[valid].astype(str)
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "n_truth_spots": int(valid.sum()),
    }


def update_summary(row, output_path):
    """Append or replace one input's row in a method-level summary CSV."""
    output_path = Path(output_path)
    current = pd.DataFrame([row])
    if output_path.is_file():
        previous = pd.read_csv(output_path)
        if "input_h5ad" in previous.columns:
            previous = previous[
                previous["input_h5ad"].astype(str)
                != str(row["input_h5ad"])
            ]
        current = pd.concat([previous, current], ignore_index=True, sort=False)
    if "n_spots" in current.columns:
        current = current.sort_values("n_spots").reset_index(drop=True)
    temporary_path = output_path.with_suffix(".csv.tmp")
    current.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)


def mclust_labels(embedding, n_clusters, seed=2023):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(ro.default_converter + numpy2ri.converter):
        converted = ro.conversion.py2rpy(np.asarray(embedding, dtype=np.float64))
    ro.r("library(mclust)")
    ro.r("set.seed")(int(seed))
    result = ro.r["Mclust"](converted, G=int(n_clusters), verbose=False)
    return np.asarray(result.rx2("classification"), dtype=str).reshape(-1)


class ResourceMonitor:
    def __init__(self, interval=0.2):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.start_rss = self.peak_rss = self.gpu_peak = 0
        self.pynvml = None
        if importlib.util.find_spec("pynvml"):
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _rss(self):
        total = 0
        try:
            processes = [self.process, *self.process.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            processes = [self.process]
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return total

    def _gpu(self):
        if self.pynvml is None:
            return 0
        try:
            processes = self.pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu_handle)
            return sum(int(p.usedGpuMemory) for p in processes if int(p.pid) == os.getpid())
        except Exception:
            return 0

    def _sample(self):
        while not self.stop_event.wait(self.interval):
            self.peak_rss = max(self.peak_rss, self._rss())
            self.gpu_peak = max(self.gpu_peak, self._gpu())

    def start(self):
        self.start_rss = self.peak_rss = self._rss()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.peak_rss = max(self.peak_rss, self._rss())
        self.gpu_peak = max(self.gpu_peak, self._gpu())
        gib = 1024**3
        return {
            "ram_start_gib": self.start_rss / gib,
            "ram_peak_gib": self.peak_rss / gib,
            "ram_peak_increase_gib": max(0, self.peak_rss - self.start_rss) / gib,
            "gpu_process_peak_gib": self.gpu_peak / gib,
        }
