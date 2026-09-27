# Next-Day Mean Wind Speed Forecasting at Şarköy, Türkiye — code and derived results

This repository accompanies the article *"Next-Day Mean Wind Speed Forecasting in a High-Variability
Coastal Microclimate: An Explainable Gradient-Boosting Benchmark and Predictability Analysis for Şarköy,
Türkiye"* by M. Tan, Y. Çetinceviz and M. Gürdal, *Sustainable Energy Technologies and Assessments*
(manuscript SETA-D-26-05002).

The target is the mean wind speed of the next calendar day at Şarköy. The forecast origin is 00:00 local
time (UTC+3) at the end of day *t*, so lead times run from 0 to 24 h across day *t*+1. The modelling
matrix covers 2,964 consecutive days, 19 November 2016 to 30 December 2024, with 559 predictors built
from the Şarköy station, the Ganos station and sixteen stations of the Trakya network.

The raw station records are not included (see *Data*).

## Versions

* **v2.0.0** adds the notebooks of the revised article in `revision/`: a leakage-free rebuild of the
  predictor matrix (forward-only gap filling, statistics fitted before the first test year), a nested
  rolling-origin evaluation over 2020–2024, block-bootstrap intervals and Diebold–Mariano tests, the
  predictor ablation, out-of-sample attribution, high-wind scores and the LSTM search. It also corrects
  the rolling windows in `features.ipynb` to 3, 7, 14, 21 and 30 days, the set used by the reported
  models, and adds `requirements.txt`.
* **v1** holds the notebooks and outputs of the original submission (root folder).

## Layout

| Path | Content |
|---|---|
| `features.ipynb` | Builds the 559-predictor matrix (`df_final_features.feather`) from the raw records |
| `modelling.ipynb`, `visualization.ipynb` | Models, Optuna searches and figures of the original submission |
| `best_params.json`, `best_xgboost_model.json`, `best_xgboost_model.pkl` | Hyperparameters and the fitted XGBoost model of the original submission; its per-day outputs are in the v1 data record |
| `revision/revision_utils.py` | Shared helpers: feature groups, model factory, bootstrap, Diebold–Mariano test, month-wise scaling, results store |
| `revision/notebooks/06_audit_calendar_and_leakage.ipynb` | Calendar and one-day-offset audit, coverage by station and variable, gap list, feature availability |
| `revision/notebooks/07_leakage_free_rebuild_and_holdout.ipynb` | Leakage-free matrix, 2024 hold-out table, imputation sensitivity |
| `revision/notebooks/07b_lstm_hpo.ipynb` | LSTM search on the leakage-free matrix (400 TPE trials, seed 42; five-seed final fits) |
| `revision/notebooks/08_nested_rolling_origin.ipynb` | Frozen pipeline over five outer test years; all selection inside each training window |
| `revision/notebooks/09_uncertainty_and_significance.ipynb` | Moving-block bootstrap, Diebold–Mariano tests, residual diagnostics |
| `revision/notebooks/10_ablation_and_feature_count.ipynb` | Ablation ladder, regional-only and physics-guided subsets, error against predictor count |
| `revision/notebooks/11_xai_out_of_sample.ipynb` | Held-out permutation importance, gain, SHAP, grouped importance, stability across years |
| `revision/notebooks/12_high_wind_events.ipynb` | High-wind thresholds and scores (bias, RMSE, attenuation, POD, FAR, CSI, AUC) |
| `revision/notebooks/13_map_diagram_graphical_abstract.ipynb` | Station map, design diagram, graphical abstract |
| `revision/notebooks/14_final_figures.ipynb` | Final versions of the article figures |
| `revision/tables/` | Every table written by the notebooks, including the 559-row feature-availability table and the list of gaps |
| `revision/logs/` | Library versions recorded by each notebook run |
| `revision/station_metadata.csv` | Station identifiers, coordinates, elevations and distances |

## Running

The notebooks were written for Google Colab and expect the Drive folder `MyDrive/windforecast_rev1/`
(set the environment variable `SARKOY_BASE` to use another folder):

```
windforecast_rev1/
├── data/                       # raw MGM exports (not redistributable)
├── features.ipynb
└── revision/
    ├── revision_utils.py
    ├── figures/  tables/  logs/
    └── station_metadata.csv
```

Run order: 06 → 07 → 07b → 08 → 09 → 10 → 11 → 12 → 13 → 14. Each notebook writes the numbers it
produces to `revision/results_NN.json`. Notebook 08 is the long one (about 40–60 min per outer year on a
Colab GPU) and resumes from the last completed year. Outside Colab, install the pinned versions with
`pip install -r requirements.txt`.

The hold-out models use the hyperparameters of the original searches (`best_params.json`); those
searches ran with an unseeded Optuna TPE sampler. The rolling-origin and LSTM searches fix the sampler
seed at 42.

## Data

Raw observations come from the Turkish State Meteorological Service (MGM, https://www.mgm.gov.tr) and
cannot be redistributed; equivalent records can be requested from MGM. The engineered feature matrix is
derived from them and is therefore not included either. Derived outputs (per-day predictions, metrics,
importance tables) are in the data record cited below.

## Citation

Code: https://doi.org/10.5281/zenodo.21178711 · Data: https://doi.org/10.5281/zenodo.21184121
(both DOIs always resolve to the latest version).

## Licence

Code: MIT (see `LICENSE`). Derived result tables: CC BY 4.0.
