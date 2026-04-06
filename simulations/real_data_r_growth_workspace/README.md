# Real-Data r-Growth Examples

Main script:

- `real_data_r_growth_examples.py`

Outputs:

- `outputs/selected_cases_summary.csv`
- `outputs/openml_selected_curves.csv`
- `outputs/openml_all_cases_summary.csv`
- `outputs/openml_all_cases_grid.png`
- `outputs/openml_all_cases/*.png`
- `outputs/cifar100_selected_curves.csv`
- `outputs/openml_r_growth_examples.png`
- `outputs/cifar100_r_growth_examples.png`

This workspace answers a focused reviewer question using saved real-data results from the OpenML and CIFAR-100 comparisons. It shows one case where increasing the KDMLP target dimension `r` helps substantially and one case where it does not, while keeping the same SupCon curve and the flat binary KFDA reference on the same plot.

| Label | Domain | Task | `r=1` KDMLP | Best `r` | Best KDMLP | SupCon best | KFDA |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | OpenML | sonar | 0.798 | 32 | 0.905 | 0.869 | 0.655 |
| B | OpenML | hypothyroid | 0.979 | 1 | 0.979 | 0.969 | 0.948 |
| C | CIFAR-100 | lion vs possum | 0.953 | 4 | 0.992 | 0.945 | 0.984 |
| D | CIFAR-100 | bear vs palm_tree | 0.992 | 1 | 0.992 | 0.992 | 0.984 |

![OpenML r-growth examples](outputs/openml_r_growth_examples.png)

The workspace now also exports all OpenML datasets one by one. Each dataset gets its own curve in `outputs/openml_all_cases/`, and the grid below gives a quick overview of all 13 tasks.

![OpenML all-case grid](outputs/openml_all_cases_grid.png)

![CIFAR-100 r-growth examples](outputs/cifar100_r_growth_examples.png)
