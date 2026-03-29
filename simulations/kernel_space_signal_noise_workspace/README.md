# Kernel-Space Signal/Noise Workspace

This folder isolates two different situations in dimension `220`:

- a strong feature-space covariance-signal case where KDMLP can beat KFDA
- a weak feature-space covariance-signal case where covariance estimation noise is larger than the true gap

## Main script

- `kernel_space_signal_noise_demo.py`

Run it with:

```bash
python kernel_space_signal_noise_demo.py
```

## Main outputs

- `outputs/strong_bestkernel_accuracy_summary.csv`
- `outputs/weak_bestkernel_accuracy_summary.csv`
- `outputs/feature_space_signal_noise_summary.csv`
- `outputs/accuracy_curves.png`
- `outputs/feature_space_signal_vs_noise.png`
- `outputs/projection_views.png`

![Feature-space signal vs noise](outputs/feature_space_signal_vs_noise.png)

![Accuracy curves](outputs/accuracy_curves.png)

## Interpretation

- Both cases are rebuilt at `d=220`.
- The strong case is a dense same-mean / different-covariance regime.
- The weak case is a dense mean-gap / tiny-covariance-gap regime.
- The comparison is driven by the feature-space diagnostics:
  `ref_D_sigma_feature` versus `cov_gap_error_feature`.
- In the strong case, the best feature-space signal-to-error ratio exceeds `1`, and KDMLP improves over KFDA.
- In the weak case, the covariance part is below the noise floor; the final accuracy can still be influenced by the mean-separation part of the method.
- For a cleaner KFDA-favorable tiny-gap example, see `simulations/redo_projection_workspace/outputs/accuracy_summary.csv`, where the verified `diff_mean_tiny_covariance_gap` regime gives `KFDA-1D = 0.6007` and best KDMLP `= 0.5883`.
