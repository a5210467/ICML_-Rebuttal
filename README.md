# ICML_-Rebuttal

This repository collects the new simulation and comparison code used for the ICML rebuttal experiments.

## Included Workspaces

- `simulations/redo_projection_workspace`
  Gaussian KDMLP vs KFDA experiments, including the higher-dimensional `d=220` audit.
- `simulations/kernel_space_signal_noise_workspace`
  Feature-space covariance-signal vs estimation-noise experiments.
- `simulations/supcon_gaussian_workspace`
  Gaussian KDMLP vs supervised contrastive learning comparisons.
- `simulations/supcon_openml_workspace`
  OpenML KDMLP vs supervised contrastive learning comparisons.
- `simulations/supcon_cifar100_workspace`
  CIFAR-100 binary-pair KDMLP vs supervised contrastive learning scans and same-`r` comparisons.

Each workspace keeps its own scripts, README, tables, and generated plots.

## 1. Large Gaussian KDMLP vs KFDA

Workspace: `simulations/redo_projection_workspace`

This set of experiments studies three large Gaussian regimes: same mean with different covariance, different mean with different covariance, and different mean with only a small covariance gap. The main figure below shows the higher-dimensional follow-up with `p=500` and a denser grid up to `r=32`; underneath it we keep a moderate-dimensional sanity baseline at `p=120`. In both settings, covariance-rich regimes benefit from larger `r`, while the small-gap regime behaves differently and should be interpreted together with the mean-separation term in `KDMLP large`.

| Label | Regime (`p=500`) | KFDA-1D | Best KDMLP | Best KDMLP acc. |
| --- | --- | ---: | --- | ---: |
| A | Same mean, different covariance | 0.523 | `KDMLP large`, `r=32` | 0.697 |
| B | Different mean, different covariance | 0.646 | `KDMLP large`, `r=32` | 0.686 |
| C | Different mean, small covariance gap | 0.663 | `KDMLP large`, `r=1` | 0.677 |

The smoother `p=500` grid is
`{1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32}`. This run shows a cleaner trend: the covariance-only case improves steadily as `r` grows, the mixed mean/covariance case gains more gradually, and the small-gap regime stays nearly flat and peaks at very small `r`.

![Gaussian KDMLP vs KFDA high-dimensional follow-up](simulations/redo_projection_workspace/outputs_d500_r32_smooth/accuracy_curves_best_kernel.png)

The moderate-dimensional baseline sweep now uses `p=120`, `repeats=1`, with `r in {1,2,3,4,6,8,10,12,16,24,32}` and gives:

| Label | Regime (`p=120`) | KFDA-1D | Best KDMLP | Best KDMLP acc. |
| --- | --- | ---: | --- | ---: |
| A | Same mean, different covariance | 0.648 | `KDMLP large`, `r=32` | 0.784 |
| B | Different mean, different covariance | 0.680 | `KDMLP small`, `r=24` | 0.715 |
| C | Different mean, small covariance gap | 0.518 | `KDMLP large`, `r=2` | 0.624 |

Here `p` is the original input dimension of the Gaussian data and `r` is the target KDMLP projection dimension. To keep the interpretation as true dimension reduction, the Gaussian workspace now enforces `r <= p` for every stage-1 kernel. The default command uses the moderate `p=120` setup above, while the `p=500` run remains the cleaner large-dimensional presentation.

![Gaussian KDMLP vs KFDA baseline sweep](simulations/redo_projection_workspace/outputs/accuracy_curves_best_kernel.png)

### Running time

We also record wall-clock timing for the moderate `p=120` Gaussian baseline in `simulations/redo_projection_workspace/outputs/timing_summary.csv`. In exact kernel form, `KFDA-1D` is still often faster, while KDMLP trades extra wall-clock time for better accuracy in the covariance-rich regimes and for a stronger low-`r` gain in the small-gap regime.

| Label | Regime (`p=120`) | KFDA-1D time (s) | Fastest KDMLP time (s) | Best-accuracy KDMLP acc. / time |
| --- | --- | ---: | ---: | --- |
| A | Same mean, different covariance | 1.03 | 0.76 | 0.784 at 1.34s |
| B | Different mean, different covariance | 0.46 | 0.60 | 0.715 at 1.02s |
| C | Different mean, small covariance gap | 0.38 | 0.40 | 0.624 at 0.41s |

This timing pattern matches the corrected statistical story in the plots: KDMLP still pays a higher stage-1 cost because it searches over covariance-sensitive projections, but the extra computation is most justified in the covariance-rich regimes. In the moderate `p=120` weak-gap case, the best setting stays at very small `r`, which is consistent with a regime where large covariance-sensitive expansions are not especially helpful.

## 2. Feature-Space Signal vs Estimation Noise

Workspace: `simulations/kernel_space_signal_noise_workspace`

This workspace is a diagnostic companion to the Gaussian experiments, but the presentation here is intentionally kept to one clean linear-kernel story. KDMLP is designed to use class-separation information carried by the difference between class covariance matrices, whereas KFDA primarily follows the mean-separation direction. The practical issue is that with finite training data, the covariance difference that KDMLP sees is always a mixture of true covariance signal and covariance-estimation error. Therefore, the most meaningful comparison is whether the true linear feature-space covariance difference is larger than its estimation error. If it is, KDMLP has a stable covariance signal to use; if it is not, any covariance-sensitive gain should be interpreted cautiously rather than as a clean covariance effect.

| Label | Case | Linear feature signal / error | KFDA-1D | Best KDMLP |
| --- | ---: | ---: | ---: |
| A | Large covariance case (`d=220`) | 3.266 | 0.781 | 0.898 |
| B | Small covariance case (`d=220`) | 0.441 | 0.648 | 0.683 |

The plots below now show only the linear stage-1 kernel at the default `d=220`, because it gives the cleanest direct comparison between true covariance signal and covariance-estimation error. In Case A, the large covariance case has a linear feature-space covariance difference that is far larger than the error (`109.015` vs `33.380`, ratio `3.266`), and KDMLP improves clearly over KFDA (`0.898` vs `0.781`). In Case B, the linear feature-space covariance difference is smaller than the error (`10.455` vs `23.714`, ratio `0.441`), so the covariance part is below the noise floor; there the best KDMLP result (`0.683` vs `0.648` for KFDA) should be read as a joint mean/covariance effect rather than as a pure covariance win. The higher-dimensional `d=500` run is still kept in the workspace as a supplementary check.

| Label | Stage-1 kernel | True feature covariance difference | Feature covariance error | Relation |
| --- | --- | ---: | ---: | --- |
| A | linear | 109.0150 | 33.3802 | signal > error |
| B | linear | 10.4546 | 23.7135 | signal < error |

This case-level table makes the intended separation explicit in one fixed linear setting: in Case A, the true feature-space covariance difference is larger than the estimation error, whereas in Case B it is smaller.

![Case-level feature covariance signal vs error](simulations/kernel_space_signal_noise_workspace/outputs/signal_noise_case_summary.png)

![Feature-space signal vs noise](simulations/kernel_space_signal_noise_workspace/outputs/feature_space_signal_vs_noise.png)

The corresponding accuracy curves make the same point from the prediction side under the same linear stage-1 kernel: when the covariance signal is stable, KDMLP can improve over KFDA, while in the weak-signal setting the behavior is much less robust and should be interpreted together with the covariance-error diagnostic above.

![Feature-space accuracy curves](simulations/kernel_space_signal_noise_workspace/outputs/accuracy_curves.png)

## 3. Gaussian KDMLP vs SupCon

Workspace: `simulations/supcon_gaussian_workspace`

This comparison uses the dimension-matched protocol: after SupCon training, its representation is compressed to the same final dimension `r` as KDMLP. In this high-dimensional synthetic setting, `KDMLP large` stays near-perfect at very small `r`, while SupCon benefits only modestly from increasing `r`.

| Label | Final dimension `r` | `KDMLP large` | SupCon |
| --- | ---: | ---: | ---: |
| A | 1 | 0.9992 | 0.9471 |
| B | 4 | 0.9992 | 0.9469 |
| C | 6 | 0.9990 | 0.9492 |

![Gaussian KDMLP vs SupCon](simulations/supcon_gaussian_workspace/outputs_dimmatch/same_r_accuracy_plot.png)

## 4. OpenML KDMLP vs SupCon

Workspace: `simulations/supcon_openml_workspace`

This workspace extends the same-`r` comparison to OpenML binary classification tasks. The strongest wins appear on datasets where a small number of KDMLP directions remains highly informative, while some datasets remain more sensitive to the chosen `r`.

| Label | Dataset | `r` | Best KDMLP | SupCon |
| --- | --- | ---: | ---: | ---: |
| A | breast-cancer | 64 | 0.722 | 0.600 |
| B | Titanic | 64 | 0.950 | 0.899 |
| C | hypothyroid | 64 | 0.977 | 0.969 |
| D | sonar | 32 | 0.905 | 0.857 |

![OpenML KDMLP vs SupCon](simulations/supcon_openml_workspace/outputs_same_r/openml_same_r_mean_curve.png)

## 5. CIFAR-100 KDMLP vs SupCon

Workspace: `simulations/supcon_cifar100_workspace`

These experiments compare a selected set of binary CIFAR-100 tasks, using the same final dimension `r` for both methods. The most useful pairs show that `KDMLP large` can outperform SupCon while still using a small or moderate target dimension.

| Label | Task | Best `r` | Best KDMLP | SupCon |
| --- | --- | ---: | ---: | ---: |
| A | lion vs possum | 4 | 0.992 | 0.930 |
| B | bee vs spider | 64 | 0.898 | 0.859 |
| C | beaver vs possum | 2 | 0.906 | 0.844 |

![CIFAR-100 KDMLP vs SupCon](simulations/supcon_cifar100_workspace/outputs_more_examples/cifar100_more_examples_same_r.png)
