# Feature-Space Signal/Noise Workspace

This folder isolates two different covariance-signal situations while evaluating the full stage-1 kernel menu. The default run is at dimension `220`, and a higher-dimensional supplementary run is provided at `d=500` with `repeats=3`.

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
- `outputs/representative_kernel_accuracy_curves.png`
- `outputs/signal_noise_case_summary.png`
- `outputs/feature_space_signal_vs_noise.png`
- `outputs/projection_views.png`
- `outputs_d500_r3/`

The figures below use the default `d=220` run in `outputs/`, and they summarize each case by the stage-1 kernel with the strongest feature-space covariance signal/error ratio. This gives one direct comparison where the large covariance case has a nonlinear feature-space covariance signal far above estimation error and the small covariance case stays below it. The `d=500` run remains available in `outputs_d500_r3/` as a supplementary check.

| Label | Stage-1 kernel | True feature covariance difference | Feature covariance error | Relation |
| --- | --- | ---: | ---: | --- |
| A | sigmoid | 0.2135 | 0.0509 | signal > error |
| B | sigmoid | 0.0290 | 0.0649 | signal < error |

![Case-level feature covariance signal vs error](outputs/signal_noise_case_summary.png)

![Feature-space signal vs noise](outputs/feature_space_signal_vs_noise.png)

![Accuracy curves](outputs/representative_kernel_accuracy_curves.png)

## Interpretation

- The displayed default figures are built at `d=220`; a supplementary `d=500`, `repeats=3` run is also provided in `outputs_d500_r3/`.
- The large covariance case is a high-dimensional same-mean / different-covariance regime.
- The small covariance case is a high-dimensional mean-gap / small-covariance-gap regime.
- The comparison is driven by the feature-space diagnostics:
  `ref_D_sigma_feature` versus `cov_gap_error_feature`.
- In the large covariance case, the representative kernel is `sigmoid`, whose feature-space covariance difference is much larger than its estimation error (`0.2135` vs `0.0509`, ratio `4.199`), and KDMLP improves over KFDA (`0.918` vs `0.874`).
- In the small covariance case, even the best representative kernel stays below the noise floor (`sigmoid`, `0.0290` vs `0.0649`, ratio `0.447`); the final accuracy can still be influenced by the mean-separation part of the method (`0.683` vs `0.648` for KFDA).
- The displayed accuracy curves and projection views now use that representative nonlinear kernel (`sigmoid`) rather than the earlier linear-only view.
- The companion Gaussian sweep in `simulations/redo_projection_workspace` now enforces `r <= p` for every stage-1 kernel and uses a moderate default `p=120`. In that sweep, the weak-gap regime reaches its best KDMLP accuracy at very small `r` (`0.624` at `r=2`, versus `0.518` for `KFDA-1D`), which is consistent with a setting where the mean-separation component matters more than large covariance-sensitive expansions.
- The higher-dimensional supplementary run is included in `outputs_d500_r3`, where `d=500` and `repeats=3`. In that run, the large covariance case still favors KDMLP (`0.579` vs `0.529` for KFDA), while the small covariance case still has feature-space signal below covariance error and is helped by the mean-separation component of `KDMLP large`.
