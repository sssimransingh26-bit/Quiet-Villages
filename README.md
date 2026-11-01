# The Villages That Went Quiet

Can the 2001 census tell us which Uttarakhand villages lost most of their people by 2011?
An experiment for the DEV Hacktoberfest Open-Source AI Challenge, Week 4 (Ghost in the Machine).
The write-up is in `results/post.md`.

**Label (fixed before scoring):** depopulated = 2011 population below 70% of 2001, or no 2011 record.
13,707 villages with at least 50 people in 2001; 1,046 (7.6%) depopulated.
**Features:** only 2001 information. SIZE = 6 features; FULL = 37 (SIZE + 19 amenities + 12 census shares).
**Evaluation:** five folds that hold out whole districts (13 districts), bootstrap 95% intervals
(optimistic: villages within a district are not independent).

## Results (see `results/`)
| Model | Features | AUC | Hit rate, riskiest 10% |
|---|---|---|---|
| Plain rule (smaller = riskier) | population | 0.681 | 17.2% |
| Gradient boosting | FULL | 0.759 | 25.0% |
| TabPFN | SIZE | 0.713 | 19.1% |
| TabPFN | FULL | 0.771 | 26.7% |

TabPFN vs gradient boosting on FULL: 0.012 AUC apart with overlapping intervals; no paired test was run.

## Reproduce
1. Download SHRUG 2.2 modules from https://www.devdatalab.org/shrug: Core keys (Population Census keys),
   Population Census (2001 Population Census Abstract, 2001 Village Directory), in .dta form.
2. `python cut_uk.py` and `python cut_by_shrid.py` in the download folder cut them down to Uttarakhand (state 05).
3. `python build_v2.py <keys_dir_with_pc01r_and_pc11r_dta> <pca01.csv> <vd01.csv> <out_dir>` builds the table and the baselines.
4. `python ablation.py <out_dir>/uk_villages_v2.csv` scores feature groups.
5. `python run_tabpfn_v2.py data/uk_villages_v2_slim.csv` (needs `pip install tabpfn`, the TabPFN license accepted at
   https://platform.priorlabs.ai/account/licenses, and your API key in the `TABPFN_TOKEN` environment variable. Never commit the key.)
6. `python make_chart.py results/results_v2_baselines.json results/tabpfn_results.json results/quiet_villages_results.png`

## Limits
Data ends in 2011 (India's 2021 census was not conducted). Uttarakhand only, rural villages only. Villages that emptied completely
appear to be missing from the 2011 table, so 64 villages with no 2011 row are labelled depopulated and may include merges or renames.
Associations, not causes. This repository deliberately contains no ranked list of villages.

## Licenses and credits
Code: MIT. Derived data (`data/`): CC BY-NC-SA 4.0, as required by SHRUG.
Data: SHRUG 2.2, Development Data Lab; Census of India. Asher, Lunt, Matsuura and Novosad (2021),
*Development research at high geographic resolution*, World Bank Economic Review 35(4).
