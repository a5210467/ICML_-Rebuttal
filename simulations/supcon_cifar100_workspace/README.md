# CIFAR-100 SupCon vs Our Method

Main script:

- `cifar100_scan_supcon_vs_ours.py`
- `cifar100_same_r_supcon_vs_ours.py`
- `cifar100_more_examples_search.py`

Outputs:

- `outputs/cifar100_pair_scan.csv`
- `outputs/cifar100_full_check.csv`
- `outputs/cifar100_margin_plot.png`
- best-pair plots such as `outputs/beaver_vs_possum_accuracy_bar.png`
- `outputs_same_r/cifar100_same_r_mean_curve.png`
- `outputs_same_r/beaver_vs_possum_same_r_summary.csv`
- `outputs_same_r/clock_vs_tractor_same_r_summary.csv`

This workspace compares a selected set of CIFAR-100 binary pairs, then reports the most useful same-`r` comparisons with stronger settings. In the same-`r` summaries, KFDA is shown across the tested `r` grid as a flat binary reference, while KDMLP and SupCon vary with the chosen target dimension.
