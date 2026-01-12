### How to keep it reproducible now, even before the “one command run”
You can still be very reproducible by adopting this rule:
Rule: every notebook must end by writing outputs to results/ and figures/
Example at the end of a notebook:
baseline_results.to_csv("../results/baseline_metrics.csv", index=False)
plt.savefig("../figures/baseline_performance.png", dpi=300, bbox_inches="tight")
This means:
you can run notebooks in any order
outputs are always in predictable places
later, “make all” just reruns those notebooks/scripts


# Data Description

## Dataset Used

This thesis uses the **Adult Reconstruction** dataset (`adult_reconstruction.csv`),
introduced by Ding et al. (2021) as part of the Folktables benchmark suite.
The dataset is a reconstructed and modernized version of the UCI Adult Census
Income dataset (Becker & Kohavi, 1996), derived from U.S. Census microdata.

The dataset contains demographic, educational, occupational, and employment
variables and is used to predict whether an individual’s income exceeds
$50,000 USD per year.

## Data Source and Provenance

The Adult Reconstruction dataset was obtained via the **Folktables** framework
and is derived from the **American Community Survey (ACS)** data distributed
through **IPUMS CPS**.

Primary references:

- Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021).
  *Retiring Adult: New Datasets for Fair Machine Learning*.
  Advances in Neural Information Processing Systems (NeurIPS).
  https://arxiv.org/abs/2108.04884

- Flood, S., King, M., Rodgers, R., Ruggles, S., & Warren, J. R. (2020).
  *Integrated Public Use Microdata Series, Current Population Survey:
  Version 8.0* [dataset].
  Minneapolis, MN: IPUMS.
  https://doi.org/10.18128/D030.V8.0

Original benchmark dataset:

- Becker, B., & Kohavi, R. (1996).
  *Adult Data Set*.
  UCI Machine Learning Repository.

## Data Availability

The file `adult_reconstruction.csv` is **not included** in this repository.

According to the IPUMS CPS terms of use, the Adult Reconstruction dataset is
intended for **replication purposes only** and may not be redistributed without
explicit permission. Researchers wishing to use the data must request access
directly via IPUMS CPS.

All preprocessing, feature engineering, data splitting, and transformations
applied in this thesis are fully documented and reproducible using the code
and notebooks provided in this repository.

## Ethical Considerations

The dataset contains sensitive demographic attributes such as gender, race,
and age. These variables are used exclusively for research purposes related to
model performance, robustness, interpretability, and fairness evaluation.

The data is known to exhibit demographic imbalances and historical biases
inherent to census-based datasets. These limitations are explicitly discussed
in the thesis and are taken into account when interpreting empirical results.

No new data was collected for this study, and no personally identifiable
information is stored or redistributed in this repository.
