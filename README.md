# ICML_-Rebuttal

This repository collects the new simulation and comparison code used for the ICML rebuttal experiments.

## Included Workspaces

- `simulations/redo_projection_workspace`
  Gaussian KDMLP vs KFDA experiments, including the higher-dimensional `d=220` audit.
- `simulations/kernel_space_signal_noise_workspace`
  Kernel-space covariance-signal vs estimation-noise experiments.
- `simulations/supcon_gaussian_workspace`
  Gaussian KDMLP vs supervised contrastive learning comparisons.
- `simulations/supcon_openml_workspace`
  OpenML KDMLP vs supervised contrastive learning comparisons.
- `simulations/supcon_cifar100_workspace`
  CIFAR-100 binary-pair KDMLP vs supervised contrastive learning scans and same-`r` comparisons.

Each workspace keeps its own scripts, README, tables, and generated plots.

## 1. Gaussian KDMLP vs KFDA

Workspace: `simulations/redo_projection_workspace`

This set of experiments studies three dense Gaussian regimes: same mean with different covariance, different mean with different covariance, and different mean with only a tiny covariance gap. The main figure below now shows the higher-dimensional follow-up with `p=500` and a denser grid up to `r=32`; the older `p=20` sweep is kept underneath as a reference baseline. In both settings, covariance-rich regimes benefit from larger `r`, while the tiny-gap regime remains much flatter.

| Regime (`p=500`) | KFDA-1D | Best KDMLP | Best KDMLP acc. |
| --- | ---: | --- | ---: |
| Same mean, different covariance | 0.523 | `MY-large_mu`, `r=32` | 0.697 |
| Different mean, different covariance | 0.646 | `MY-large_mu`, `r=32` | 0.686 |
| Different mean, tiny covariance gap | 0.663 | `MY-large_mu`, `r=1` | 0.677 |

The smoother `p=500` grid is
`{1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32}`. This run shows a cleaner trend: the covariance-only case improves steadily as `r` grows, the mixed mean/covariance case gains more gradually, and the tiny-gap regime stays nearly flat and peaks at very small `r`.

![Gaussian KDMLP vs KFDA high-dimensional follow-up](simulations/redo_projection_workspace/outputs_d500_r32_smooth/accuracy_curves_best_kernel.png)

The original baseline sweep uses `p=20` with `r in {1,2,3,4,8,16,32,64}` and gives:

| Regime (`p=20`) | KFDA-1D | Best KDMLP | Best KDMLP acc. |
| --- | ---: | --- | ---: |
| Same mean, different covariance | 0.686 | `MY-small_mu`, `r=64` | 0.866 |
| Different mean, different covariance | 0.738 | `MY-small_mu`, `r=64` | 0.910 |
| Different mean, tiny covariance gap | 0.601 | `MY-large_mu`, `r=32` | 0.617 |

![Gaussian KDMLP vs KFDA baseline sweep](simulations/redo_projection_workspace/outputs/accuracy_curves_best_kernel.png)

## 2. Kernel-Space Signal vs Estimation Noise

Workspace: `simulations/kernel_space_signal_noise_workspace`

This workspace is a diagnostic companion to the Gaussian experiments. KDMLP is designed to capture class-separation information carried by the difference between class covariance matrices, which is not directly exploited by KFDA. Therefore, KDMLP is expected to help when the covariance difference contains meaningful discriminative structure. However, in data-driven settings the true covariance operators are unknown, and any method must rely on finite-sample estimates. Even in the extreme case where the true covariance matrices are identical, the sample covariance matrices will almost surely differ due to estimation noise. A method that explicitly tries to leverage this artifactual difference can therefore generalize worse than FDA or KFDA, which are better aligned with a shared-covariance regime. More broadly, KDMLP should not be expected to uniformly outperform KFDA on arbitrary datasets. It is best suited to problems where covariance differences provide useful class-separation information beyond mean separation, and where the true covariance-gap signal is not overwhelmed by covariance estimation error. In this workspace, that trade-off is summarized by comparing the feature-space covariance-gap magnitude to its estimation error, measured in the Hilbert-Schmidt norm.

| Case | Best feature-space signal / noise | KFDA-1D | Best KDMLP |
| --- | ---: | ---: | ---: |
| Strong kernel covariance signal (`d=500`) | 0.835 | 0.529 | 0.579 |
| Weak kernel covariance signal (`d=500`) | 0.436 | 0.580 | 0.679 |

The plots below now show the higher-dimensional supplementary run at `d=500` with `repeats=3`. In that run, the strong case still favors KDMLP (`KFDA-1D = 0.529`, best KDMLP `= 0.579`), while the weak case remains covariance-noise dominated in the feature-space diagnostic (best signal-to-noise ratio about `0.436`). The complementary weak-gap Gaussian regime in `simulations/redo_projection_workspace` is still much less decisive than the covariance-rich cases: after extending the KDMLP sweep to `r=64`, the best KDMLP accuracy is only `0.617` versus `0.601` for `KFDA-1D`, so the gain is marginal rather than a strong win.

![Kernel-space signal vs noise](simulations/kernel_space_signal_noise_workspace/outputs_d500_r3/feature_space_signal_vs_noise.png)

The corresponding accuracy curves make the same point from the prediction side: when the covariance signal is stable, KDMLP can improve over KFDA, while in the weak-signal setting the behavior is much less robust and should be interpreted together with the covariance-error diagnostic above.

![Kernel-space accuracy curves](simulations/kernel_space_signal_noise_workspace/outputs_d500_r3/accuracy_curves.png)

## 3. Gaussian KDMLP vs SupCon

Workspace: `simulations/supcon_gaussian_workspace`

This comparison uses the dimension-matched protocol: after SupCon training, its representation is compressed to the same final dimension `r` as KDMLP. In this high-dimensional synthetic setting, `MY-large_mu` stays near-perfect at very small `r`, while SupCon benefits only modestly from increasing `r`.

| Final dimension `r` | `MY-large_mu` | SupCon |
| ---: | ---: | ---: |
| 1 | 0.9992 | 0.9471 |
| 4 | 0.9992 | 0.9469 |
| 6 | 0.9990 | 0.9492 |

![Gaussian KDMLP vs SupCon](simulations/supcon_gaussian_workspace/outputs_dimmatch/same_r_accuracy_plot.png)

## 4. OpenML KDMLP vs SupCon

Workspace: `simulations/supcon_openml_workspace`

This workspace extends the same-`r` comparison to OpenML binary classification tasks. The strongest wins appear on datasets where a small number of KDMLP directions remains highly informative, while some datasets remain more sensitive to the chosen `r`.

| Dataset | `r` | Best KDMLP | SupCon |
| --- | ---: | ---: | ---: |
| breast-cancer | 64 | 0.722 | 0.600 |
| Titanic | 64 | 0.950 | 0.899 |
| hypothyroid | 64 | 0.977 | 0.969 |
| sonar | 32 | 0.905 | 0.857 |

![OpenML KDMLP vs SupCon](simulations/supcon_openml_workspace/outputs_same_r/openml_same_r_mean_curve.png)

## 5. CIFAR-100 KDMLP vs SupCon

Workspace: `simulations/supcon_cifar100_workspace`

These experiments search binary CIFAR-100 tasks and then rerun the most interesting ones with the same final dimension `r` for both methods. The most useful pairs show that `MY-large_mu` can outperform SupCon while still using a small or moderate target dimension.

| Task | Best `r` | Best KDMLP | SupCon |
| --- | ---: | ---: | ---: |
| lion vs possum | 4 | 0.992 | 0.930 |
| bee vs spider | 64 | 0.898 | 0.859 |
| beaver vs possum | 2 | 0.906 | 0.844 |

![CIFAR-100 KDMLP vs SupCon](simulations/supcon_cifar100_workspace/outputs_more_examples/cifar100_more_examples_same_r.png)
