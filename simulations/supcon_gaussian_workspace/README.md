# Gaussian SupCon vs Our Method

Main script:

- `gaussian_supcon_vs_ours.py`
- `gaussian_supcon_vs_ours_hd220.py`
- `gaussian_same_r_dimmatch.py`

Outputs:

- `outputs/gaussian_summary.csv`
- `outputs/gaussian_diagnostics.csv`
- `outputs/gaussian_accuracy_comparison.png`
- `outputs/gaussian_signal_vs_error.png`
- `outputs_hd220/hd220_summary.csv`
- `outputs_hd220/hd220_accuracy_bar.png`

This compares `KFDA-1D`, `KDMLP large`, `KDMLP small`, and a feature-level `SupCon` baseline on the Gaussian synthetic regimes reused from the redo projection experiments.
