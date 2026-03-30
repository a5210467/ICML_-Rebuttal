# Redo Projection Workspace

This folder is a fresh rebuild of the synthetic comparison, using the style of the earlier `1D_projection.ipynb` experiment but with:

- nested kernel selection for stage 1
- stage-2 SVM cross-validation
- paper-style block `A, B` stage-1 solves for `KDMLP large` and `KDMLP small`
- KFDA baseline
- original-space and feature-space covariance diagnostics
- an extended KDMLP target-dimension sweep with the rule `r <= p`
- a moderate default sweep at `p=120`
- a higher-dimensional follow-up with configurable `p`, including a `p=500` smooth-curve run up to `r=32`

## Main script

- `redo_projection_cv_experiment.py`

Run it with:

```bash
python redo_projection_cv_experiment.py
```

Example higher-dimensional run:

```bash
python redo_projection_cv_experiment.py \
  --dimension 500 \
  --repeats 3 \
  --r-values 1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32 \
  --output-subdir outputs_d500_r32_smooth
```

## Current summary

The main follow-up we now highlight is the smoother high-dimensional run:

- `p=500`
- `repeats=3`
- `r in {1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32}`
- output folder: `outputs_d500_r32_smooth`

That run gives:

| Label | Regime | KFDA-1D | Best KDMLP | Best KDMLP acc. |
| --- | --- | ---: | --- | ---: |
| A | Same mean, different covariance | 0.523 | `KDMLP large`, `r=32` | 0.697 |
| B | Different mean, different covariance | 0.646 | `KDMLP large`, `r=32` | 0.686 |
| C | Different mean, small covariance gap | 0.663 | `KDMLP large`, `r=1` | 0.677 |

The smoother high-dimensional curve is here:

![High-dimensional Gaussian sweep](outputs_d500_r32_smooth/accuracy_curves_best_kernel.png)

The moderate default sweep is kept in `outputs/`, with:

- `p=120`
- `repeats=1`
- `r in {1,2,3,4,6,8,10,12,16,24,32}`

| Label | Regime | KFDA-1D | Best KDMLP | Best KDMLP acc. |
| --- | --- | ---: | --- | ---: |
| A | Same mean, different covariance | 0.648 | `KDMLP large`, `r=32` | 0.784 |
| B | Different mean, different covariance | 0.680 | `KDMLP small`, `r=24` | 0.715 |
| C | Different mean, small covariance gap | 0.518 | `KDMLP large`, `r=2` | 0.624 |

Here `p` is the original Gaussian input dimension and `r` is the target projection dimension. The Gaussian workspace now enforces `r <= p` for every stage-1 kernel. The `p=120` run is the new moderate default, while the `p=500` run is still the cleaner large-dimensional presentation.

## Running time

The script now also records wall-clock time in `outputs/timing_summary.csv` for the default `p=120` sweep. The exact kernel KFDA solve is still often faster, while KDMLP spends more time in the covariance-sensitive stage-1 search and in larger target dimensions.

| Label | Regime | KFDA-1D time (s) | Fastest KDMLP time (s) | Best-accuracy KDMLP acc. / time |
| --- | --- | ---: | ---: | --- |
| A | Same mean, different covariance | 1.03 | 0.76 | 0.784 at 1.34s |
| B | Different mean, different covariance | 0.46 | 0.60 | 0.715 at 1.02s |
| C | Different mean, small covariance gap | 0.38 | 0.40 | 0.624 at 0.41s |

So the practical trade-off is still fairly clean in this corrected synthetic setting: KFDA is usually cheaper, but KDMLP can still buy a substantial accuracy gain in the covariance-rich regimes. In the moderate `p=120` small-gap regime, the best setting remains at very small `r`, which is consistent with a case where aggressive covariance-sensitive expansion is not especially useful.

![Baseline Gaussian sweep](outputs/accuracy_curves_best_kernel.png)

## Main outputs

- `outputs/accuracy_summary.csv`
- `outputs/selection_log.csv`
- `outputs/covariance_diagnostics_summary.csv`
- `outputs/accuracy_curves_best_kernel.png`
- `outputs/covariance_diagnostics.png`
- `outputs/selected_projection_views.png`
- `outputs_d500_r32_smooth/`

## Math note

- `kfda_vs_kdmlp_note.tex`

This note explains:

- what `D_mu` and `D_sigma` mean in RKHS
- why nonlinear kernels can let KFDA benefit from covariance changes too
- a covariance concentration bound in feature space
- when KFDA should be safer than KDMLP
