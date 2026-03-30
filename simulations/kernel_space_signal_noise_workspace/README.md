# Feature-Space Signal/Noise Workspace

This folder isolates two different covariance-signal situations using one fixed linear stage-1 kernel. The default run is at dimension `220`, and a higher-dimensional supplementary run is provided at `d=500` with `repeats=3`.

- a large covariance case where KDMLP can beat KFDA
- a small covariance case where covariance estimation noise is larger than the true gap

## Main script

- `kernel_space_signal_noise_demo.py`

Run it with:

```bash
python kernel_space_signal_noise_demo.py
```

Example higher-dimensional run:

```bash
python kernel_space_signal_noise_demo.py --dimension 500 --repeats 3 --output-subdir outputs_d500_r3
```

## Main outputs

- `outputs/strong_bestkernel_accuracy_summary.csv`
- `outputs/weak_bestkernel_accuracy_summary.csv`
- `outputs/feature_space_signal_noise_summary.csv`
- `outputs/signal_noise_case_summary.csv`
- `outputs/accuracy_curves.png`
- `outputs/signal_noise_case_summary.png`
- `outputs/feature_space_signal_vs_noise.png`
- `outputs/projection_views.png`
- `outputs_d500_r3/`

The figures below use the default `d=220` run in `outputs/`, and they now show only the linear stage-1 kernel. This gives one direct comparison where the large covariance case has true linear feature covariance signal far above estimation error and the small covariance case has signal below error. The `d=500` run remains available in `outputs_d500_r3/` as a supplementary check.

| Label | Stage-1 kernel | True feature covariance difference | Feature covariance error | Relation |
| --- | --- | ---: | ---: | --- |
| A | linear | 109.0150 | 33.3802 | signal > error |
| B | linear | 10.4546 | 23.7135 | signal < error |

![Case-level feature covariance signal vs error](outputs/signal_noise_case_summary.png)

![Feature-space signal vs noise](outputs/feature_space_signal_vs_noise.png)

![Accuracy curves](outputs/accuracy_curves.png)

## Interpretation

- The displayed default figures are built at `d=220`; a supplementary `d=500`, `repeats=3` run is also provided in `outputs_d500_r3/`.
- The large covariance case is a high-dimensional same-mean / different-covariance regime.
- The small covariance case is a high-dimensional mean-gap / small-covariance-gap regime.
- The comparison is driven by the linear feature-space diagnostics:
  `ref_D_sigma_feature` versus `cov_gap_error_feature`.
- In the large covariance case, the linear feature-space covariance difference is much larger than its estimation error (`109.0150` vs `33.3802`, ratio `3.266`), and KDMLP improves over KFDA (`0.898` vs `0.781`).
- In the small covariance case, the linear covariance part stays below the noise floor (`10.4546` vs `23.7135`, ratio `0.441`); the final accuracy can still be influenced by the mean-separation part of the method (`0.683` vs `0.648` for KFDA).
- The companion Gaussian sweep in `simulations/redo_projection_workspace` now enforces `r <= p` for every stage-1 kernel and uses a moderate default `p=120`. In that sweep, the weak-gap regime reaches its best KDMLP accuracy at very small `r` (`0.624` at `r=2`, versus `0.518` for `KFDA-1D`), which is consistent with a setting where the mean-separation component matters more than large covariance-sensitive expansions.
- The higher-dimensional supplementary run is included in `outputs_d500_r3`, where `d=500` and `repeats=3`. In that run, the large covariance case still favors KDMLP (`0.579` vs `0.529` for KFDA), while the small covariance case still has feature-space signal below covariance error and is helped by the mean-separation component of `KDMLP large`.
