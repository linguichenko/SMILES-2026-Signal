import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):

    score_filter = helpers["score_filter"]
    fit_tx_prediction = helpers["fit_tx_prediction"]

    pred1 = fit_tx_prediction(rx)

    rx1 = rx - pred1

    extra_terms = []

    for k in range(tx_n.shape[1]):
        x = tx_n[:, k]

        extra_terms.append(x * np.abs(x) ** 2)

    X = np.column_stack([
        score_filter(t)
        for t in extra_terms
    ])

    rx2 = rx1.copy()

    for ch in range(rx.shape[1]):

        y = score_filter(rx1[:, ch])

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

        pred_extra = X @ coef

        rx2[:, ch] -= pred_extra

    band = np.column_stack([
        score_filter(rx2[:, ch])
        for ch in range(rx2.shape[1])
    ])

    cov = band.conj().T @ band
    cov /= band.shape[0]

    eigvals, eigvecs = np.linalg.eigh(cov)

    v = eigvecs[:, -1]

    shared = band @ v

    power = np.vdot(shared, shared).real

    rank1 = np.zeros_like(rx2)

    for ch in range(4):

        alpha = np.vdot(shared, band[:, ch]) / power

        rank1[:, ch] = alpha * shared


    rx_hat = (rx2 - 0.9 * rank1)

    return rx_hat

print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
