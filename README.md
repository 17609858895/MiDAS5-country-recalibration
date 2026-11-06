# MiDAS 5 country recalibration

Analysis and figure-generation source for **Few local labels correct country offsets in global anaerobic-digester microbiome models**.

The study separates country offsets from within-country prediction error, evaluates empirical-Bayes recalibration from a few measured digesters, and estimates within-country microbial associations with ammonium nitrogen. The repository contains the scientific code and the nonquantitative illustration needed for Figure 6. Manuscript files, raw sequencing tables, derived datasets and fitted model objects are excluded.

## Data

The public MiDAS 5 input release is https://doi.org/10.6084/m9.figshare.24219199.v2 (CC BY 4.0). The source article is Dueholm et al., Nature Communications 15, 5361 (2024), https://doi.org/10.1038/s41467-024-49641-y. Retain these citations when using the data. Extract the release into `data_extracted/` with the input basenames listed in `scripts/01_prepare_data.py`. These directories are relative to the repository root.

The article's supplementary data contain processed matrices and result tables. To redraw figures without fitting models, the processed data, result CSVs and fitted-result objects supplied with the supplementary analysis package must be placed in the corresponding `processed/` and `results/` directories. Otherwise run the analysis pipeline from raw inputs first. No fitted objects are included in this source-only repository.

## Environment and execution

Use Python 3.12 in an isolated environment and install `requirements.txt`. Some analyses are computationally intensive and parallel scripts use multiple CPU cores. Run scripts from the repository root. The numbered scientific pipeline is:

1. `01_prepare_data.py`, `02_screen_targets.py`, `03_benchmark20.py`: prepare the matrices, select labelled subsets and compare regression algorithms.
2. `04_invariance_prototype.py`: feature-based alternatives to recalibration.
3. `05_causal_dml.py`, `05b_causal_dml_within.py`, `05c_dml_refutation_parallel.py`: pooled, within-country and sensitivity analyses. The historical filename uses causal terminology; the reported estimates are adjusted cross-sectional associations.
4. `06_shift_aware_stacking.py`, `07_cross_primer.py`, `08_fewshot_recalibration.py`, `09_taxonomic_resolution.py`: stacking, primer transfer, few-label correction and resolution comparisons.
5. `10b_fewshot_conformal.py`: prediction intervals after recalibration; run after 08. `11_shap_shortcut.py` through `14_recovery_hierarchy_simpson.py`: interpretation, shrinkage, operating-state classification and country-scale diagnostics.
6. `16_run_offsets.py`, `17_label_budget.py`: sequencing-run comparisons and variance-based budgets. Run `15_supplementary_tables.py` after the analyses to assemble the supplementary workbook.

For each entry above, prepend `python scripts/`. Intermediate predictions and fitted-result objects are written to `results/`. Existing files may be replaced when analyses are repeated.

After results are available, redraw the ten main figures and one supplementary figure with:

```bash
python scripts/figures/generate_ten_main_figures.py
```

Use `--only Fig6` to redraw one figure. Original panel modules remain required numerical dependencies. The composition entry point redraws independent panels, exports PNG/PDF files and writes figure source tables; it does not fit prediction models. The included Natural Earth geographic boundaries are public-domain base data; their upstream source is https://www.naturalearthdata.com/.

## Interpretation

Paired A/B samples remain together in outer validation. The label-free abundance filter uses all available samples; internal regularisation may split A/B samples. The survey supplied both model selection and evaluation. The source field is ammonium nitrogen, in mg N/L. Names containing TAN or NH4 are historical identifiers; free-ammonia results use an explicitly assumed equilibrium proxy.

The label budget estimates a recoverable offset component; expected R-squared gains use evaluation-set variation retrospectively. Empirical coverage is reported for the empirical-Bayes interval extension, whose full procedure has no established finite-sample coverage proof. Double-machine-learning coefficients are adjusted associations, not identified causal effects.

This repository was checked for Python syntax and source consistency. Publishing it does not represent a new execution of the full modelling pipeline.
