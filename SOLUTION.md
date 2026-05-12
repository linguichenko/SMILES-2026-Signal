# SOLUTION.md

## Reproducibility Instructions

### Environment

The solution was developed and tested with:

* Python 3.10+
* numpy
* scipy
* gdown
* json (Python standard library)

Install the required dependencies with:

```bash
pip install numpy scipy gdown
```

### Running the Solution

Run the main entrypoint:

```bash
python applicant_solution.py
```

This command will:

1. Download `challenge.mat`
2. Load the TX/RX data
3. Compute the provided baseline
4. Run the custom canceller implementation
5. Write the resulting metrics into `results.json`

The main reported metric is:

```json
results["yours"]["average_db"]
```

### Important Reproducibility Notes

* The repository is self-contained.
* No modifications to `task_and_baseline.py` or the dataset are required.
* The implementation relies on the provided helper functions:

  * `score_filter`
  * `fit_tx_prediction`
* Small numerical differences may appear across environments due to BLAS/LAPACK implementations.

---

# Final Solution Description

## Overview

The final solution uses a two-stage interference cancellation pipeline:

1. TX-driven nonlinear interference cancellation
2. Spatially coherent rank-1 interference cancellation

The approach was designed to remain consistent with the explainability constraints enforced by the official scorer.

---

## 1. TX-Driven Nonlinear Cancellation

The provided baseline already models several nonlinear TX interaction terms of the form:

$$
x^2 y^*
$$

including multiple lagged versions.

This stage estimates the component of RX that can be explained by nonlinear leakage from the transmitted signals and subtracts it.

The implementation reuses the provided helper:

```python
fit_tx_prediction(...)
```

This produces the first residual:

$$
rx_1 = rx - \hat{I}_{TX}
$$

---

## 2. Additional Memoryless Nonlinear Terms

The baseline model was extended with additional cubic nonlinear features of the form:

$$
x |x|^2
$$

for each TX channel.

These terms are commonly used in RF and power-amplifier distortion models and capture self-induced cubic nonlinearities that are not fully represented by the provided baseline features.

The filtered nonlinear features are stacked into a regression matrix and fitted independently for each RX channel using least squares.

This produces an improved residual:

$$
rx_2
$$

with reduced TX-related interference.

This modification contributed noticeably to the metric improvement over the provided baseline.

---

## 3. Rank-1 Coherent Interference Removal

After TX cancellation, a remaining interference component was observed to be spatially coherent across RX channels, matching the task description of the external interference term.

To estimate this component:

1. The residual signals were bandpass filtered using the provided scoring filter.
2. A covariance matrix across RX channels was computed.
3. Eigenvalue decomposition was applied.
4. The dominant eigenvector was used to estimate the shared rank-1 interference component.

The recovered rank-1 component was then subtracted from the residual signal.

A regularization term was added to the covariance matrix for numerical stability.

---

## 4. Partial Rank-1 Subtraction

Experiments showed that subtracting the full rank-1 estimate was slightly suboptimal.

The final implementation uses:

```python
rx_hat = rx2 - 0.9 * rank1
```

instead of full subtraction.

This likely avoids removing portions of the desired signal that partially overlap with the dominant coherent component.

### Coefficient Experiments

| Coefficient | Average Score (dB) |
| ----------- | ------------------ |
| 0.50        | 7.87               |
| 0.85        | 8.51               |
| 0.90        | 8.52               |
| 0.93        | 8.51               |
| 0.95        | 8.50               |
| 0.99        | 8.47               |
| 1.00        | 8.46               |

The best observed result was obtained with:

```python
0.9
```

---

# Experiments and Failed Attempts

## 1. Initial Working Version (~7 dB)

The first successful approach consisted of:

1. Baseline TX cancellation
2. Single rank-1 coherent interference removal

This implementation already produced a significant improvement over the provided baseline (~7 dB average score).

The method worked because it directly matched the structure described in the task statement:

$$
I = F(TX) + E
$$

where:

* `F(TX)` was handled by the baseline nonlinear model
* `E` was modeled as a coherent rank-1 component across RX channels

However, the remaining TX-related nonlinear residuals limited the achievable score.

---

## 2. Iterative Cancellation Attempts

Several iterative approaches were tested, including repeated alternating stages of:

1. TX cancellation
2. Rank-1 cancellation

These methods frequently produced invalid solutions with a final score of `0 dB`.

The likely reason is that aggressive iterative subtraction began removing components that could no longer be explained as:

* TX-driven nonlinear interference, or
* coherent rank-1 interference

which violated the explainability constraints enforced by the scorer.

As a result, these iterative approaches were discarded.

---

## 3. Full Rank-1 Subtraction

Using:

```python
rx_hat = rx2 - rank1
```

(full subtraction)

worked reasonably well but consistently underperformed slightly compared to partial subtraction.

This suggested that the dominant coherent component also contained some desired signal energy, making conservative subtraction preferable.
