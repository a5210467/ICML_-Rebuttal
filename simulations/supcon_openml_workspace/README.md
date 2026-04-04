# OpenML SupCon vs Our Method

Main script:

- `openml_supcon_vs_ours.py`
- `openml_supcon_vs_ours_same_r.py`

Outputs:

- `outputs/openml_summary.csv`
- `outputs/openml_ours_full.csv`
- `outputs/openml_supcon_rows.csv`
- `outputs/openml_margin_vs_supcon.png`
- `outputs/openml_summary_barplot.png`
- `outputs_same_r/openml_same_r_mean_curve.png`
- `outputs_same_r/openml_dim32_64_barplot.png`
- `outputs_same_r/openml_kfda_best_same_r.csv`

This uses the same OpenML dataset list as the earlier OpenML comparison notebook and adds both a feature-level `SupCon` baseline and a same-`r` binary KFDA reference. In the same-`r` outputs, KFDA is repeated across the tested `r` grid because the binary KFDA stage-1 discriminant is intrinsically rank-1.
