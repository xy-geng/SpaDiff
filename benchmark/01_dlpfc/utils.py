"""Shared mclust, metric, and resource-monitoring helpers for the benchmarks."""

from __future__ import annotations

import importlib.util
import os
import threading

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def mclust_labels(
    embedding,
    n_domains: int,
    seed: int = 0,
) -> np.ndarray:
    """Run R mclust for methods without a built-in mclust helper."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    embedding = np.asarray(embedding, dtype=np.float64)
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_embedding = ro.conversion.py2rpy(embedding)

    ro.r("library(mclust)")
    ro.r("set.seed")(int(seed))
    result = ro.r["Mclust"](
        r_embedding,
        G=int(n_domains),
        verbose=False,
    )
    return np.asarray(result.rx2("classification"), dtype=str).reshape(-1)


def calculate_ari_nmi(truth, prediction) -> tuple[float, float, int]:
    truth = pd.Series(truth).reset_index(drop=True)
    prediction = pd.Series(prediction).reset_index(drop=True)
    valid = truth.notna() & prediction.notna()
    y_true = truth[valid].astype(str)
    y_pred = prediction[valid].astype(str)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(
        y_true,
        y_pred,
        average_method="arithmetic",
    )
    return float(ari), float(nmi), int(valid.sum())


class ResourceMonitor:
    """Monitor RSS and process-level GPU memory."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread = None
        self.start_rss = 0
        self.peak_rss = 0
        self.gpu_peak = 0

        if importlib.util.find_spec("pynvml") is not None:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self.pynvml = None
            self.gpu_handle = None

    def _gpu_bytes(self):
        if self.pynvml is None:
            return 0

        processes = self.pynvml.nvmlDeviceGetComputeRunningProcesses(
            self.gpu_handle
        )
        return sum(
            int(process.usedGpuMemory)
            for process in processes
            if int(process.pid) == os.getpid()
        )

    def _rss(self):
        processes = [self.process]
        processes.extend(self.process.children(recursive=True))

        total = 0
        for process in processes:
            total += process.memory_info().rss
        return total

    def _sample(self):
        while not self.stop_event.wait(self.interval):
            self.peak_rss = max(self.peak_rss, self._rss())
            self.gpu_peak = max(self.gpu_peak, self._gpu_bytes())

    def start(self):
        self.start_rss = self._rss()
        self.peak_rss = self.start_rss
        self.thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

        self.peak_rss = max(self.peak_rss, self._rss())
        self.gpu_peak = max(self.gpu_peak, self._gpu_bytes())
        gibibyte = 1024**3

        return {
            "ram_start_gib": self.start_rss / gibibyte,
            "ram_peak_gib": self.peak_rss / gibibyte,
            "ram_peak_increase_gib": max(
                0,
                self.peak_rss - self.start_rss,
            )
            / gibibyte,
            "gpu_process_peak_gib": self.gpu_peak / gibibyte,
        }


def gpu_metrics(torch_module):
    if torch_module is None or not torch_module.cuda.is_available():
        return {
            "gpu_peak_allocated_gib": 0.0,
            "gpu_peak_reserved_gib": 0.0,
            "gpu_total_gib": 0.0,
            "device": "CPU",
        }

    gibibyte = 1024**3
    device_properties = torch_module.cuda.get_device_properties(0)
    return {
        "gpu_peak_allocated_gib": (
            torch_module.cuda.max_memory_allocated() / gibibyte
        ),
        "gpu_peak_reserved_gib": (
            torch_module.cuda.max_memory_reserved() / gibibyte
        ),
        "gpu_total_gib": device_properties.total_memory / gibibyte,
        "device": torch_module.cuda.get_device_name(0),
    }
