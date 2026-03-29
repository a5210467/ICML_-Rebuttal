# Redo Projection Workspace

This folder is a fresh rebuild of the synthetic comparison, using the style of the earlier `1D_projection.ipynb` experiment but with:

- nested kernel selection for stage 1
- stage-2 SVM cross-validation
- paper-style block `A, B` stage-1 solves for `MY-large_mu` and `MY-small_mu`
- KFDA baseline
- original-space and feature-space covariance diagnostics
- an extended KDMLP target-dimension sweep `r in {1,2,3,4,8,16,32,64}`
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

The updated Gaussian sweep no longer stops at `r=4`. With the extended grid through `r=64`, the covariance-rich regimes continue to improve:

| Regime | KFDA-1D | Best KDMLP | Best KDMLP acc. |
| --- | ---: | --- | ---: |
| Same mean, different covariance | 0.686 | `MY-small_mu`, `r=64` | 0.866 |
| Different mean, different covariance | 0.738 | `MY-small_mu`, `r=64` | 0.910 |
| Different mean, tiny covariance gap | 0.601 | `MY-large_mu`, `r=32` | 0.617 |

The key qualitative change is that the two covariance-rich regimes keep gaining from larger `r`, while the tiny-gap regime remains much flatter and only marginally exceeds the KFDA baseline.

## Higher-dimensional follow-up (`p=500`)

To make the `r`-curves smoother in a higher original dimension, we also ran:

- `p=500`
- `repeats=3`
- `r in {1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32}`
- output folder: `outputs_d500_r32_smooth`

That run gives:

| Regime | KFDA-1D | Best KDMLP | Best KDMLP acc. |
| --- | ---: | --- | ---: |
| Same mean, different covariance | 0.523 | `MY-large_mu`, `r=32` | 0.697 |
| Different mean, different covariance | 0.646 | `MY-large_mu`, `r=32` | 0.686 |
| Different mean, tiny covariance gap | 0.663 | `MY-large_mu`, `r=1` | 0.677 |

The smoother high-dimensional curve is here:

![High-dimensional Gaussian sweep](outputs_d500_r32_smooth/accuracy_curves_best_kernel.png)

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
