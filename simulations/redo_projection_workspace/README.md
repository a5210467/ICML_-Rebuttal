# Redo Projection Workspace

This folder is a fresh rebuild of the synthetic comparison, using the style of the earlier `1D_projection.ipynb` experiment but with:

- nested kernel selection for stage 1
- stage-2 SVM cross-validation
- paper-style block `A, B` stage-1 solves for `KDMLP large` and `KDMLP small`
- KFDA baseline
- original-space and feature-space covariance diagnostics
- an extended KDMLP target-dimension sweep with the rule `r <= p`
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

The original baseline sweep is still kept in `outputs/`, with the corrected grid `r in {1,2,3,4,5,6,7,8,10,12,14,16,20}`:

| Label | Regime | KFDA-1D | Best KDMLP | Best KDMLP acc. |
| --- | --- | ---: | --- | ---: |
| A | Same mean, different covariance | 0.686 | `KDMLP small`, `r=10` | 0.820 |
| B | Different mean, different covariance | 0.738 | `KDMLP large`, `r=16` | 0.887 |
| C | Different mean, small covariance gap | 0.601 | `KDMLP large`, `r=16` | 0.602 |

Here `p` is the original Gaussian input dimension and `r` is the target projection dimension. The corrected Gaussian workspace now enforces `r <= p` for every stage-1 kernel, so the `p=20` sweep stops at `r=20`. The `p=500` run is therefore still the cleaner large-dimensional presentation, while the `p=20` run is kept as the original baseline.

## Running time

The script now also records wall-clock time in `outputs/timing_summary.csv` for the baseline `p=20` sweep. The exact kernel KFDA solve is still faster, while KDMLP spends more time in the covariance-sensitive stage-1 search and in larger target dimensions.

| Label | Regime | KFDA-1D time (s) | Fastest KDMLP time (s) | Best-accuracy KDMLP acc. / time |
| --- | --- | ---: | ---: | --- |
| A | Same mean, different covariance | 0.55 | 0.67 | 0.820 at 0.80s |
| B | Different mean, different covariance | 0.36 | 0.48 | 0.887 at 0.70s |
| C | Different mean, small covariance gap | 0.27 | 0.36 | 0.602 at 0.50s |

So the practical trade-off is fairly clean in this corrected synthetic setting: KFDA is cheaper, but KDMLP can still buy a substantial accuracy gain in the covariance-rich regimes, whereas the small-gap regime yields almost no statistical return for the extra runtime.

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
