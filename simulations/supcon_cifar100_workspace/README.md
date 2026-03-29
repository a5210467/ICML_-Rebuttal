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

This scans a set of CIFAR-100 binary pairs with a quick pass, then re-checks the strongest pairs with stronger settings.
