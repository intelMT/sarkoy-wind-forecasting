"""revision_utils.py - shared helpers for the SETA-D-26-05002 revision notebooks (06-13).

Copy this file to  MyDrive/windforecast/revision/  (the notebooks add that folder to sys.path).
Nothing here touches the original notebooks or files; every new output is written under
<BASE>/revision/.
"""
import os, sys, json, math, time, warnings, subprocess
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
# Paths and run mode
# ----------------------------------------------------------------------------------------------
def get_paths():
    """BASE = the original project folder on Google Drive; REV = new outputs of this revision."""
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except Exception:
        in_colab = False
    if in_colab:
        base = os.environ.get('SARKOY_BASE', '/content/drive/MyDrive/windforecast_rev1/')
    else:
        base = os.environ.get('SARKOY_BASE', './windforecast/')
    base = base if base.endswith('/') else base + '/'
    rev = os.path.join(base, 'revision') + '/'
    for sub in ['', 'figures', 'tables', 'models', 'variants', 'logs']:
        os.makedirs(os.path.join(rev, sub), exist_ok=True)
    fast = os.environ.get('SARKOY_FAST', '0') == '1'   # smoke-test mode with tiny budgets
    return base, rev, in_colab, fast


def has_gpu():
    try:
        return subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
    except Exception:
        return False


def log_versions(rev):
    """Record the exact library versions of the current run (reported in Table S-versions)."""
    import importlib, platform
    pk = ['numpy', 'pandas', 'sklearn', 'scipy', 'xgboost', 'lightgbm', 'catboost', 'optuna', 'shap',
          'statsmodels', 'matplotlib', 'seaborn', 'pyarrow', 'ephem', 'tensorflow']
    v = {'python': platform.python_version()}
    for p in pk:
        try:
            v[p] = importlib.import_module(p).__version__
        except Exception:
            v[p] = 'not installed'
    v['gpu'] = has_gpu()
    fn = os.path.join(rev, 'logs', 'versions_%s.json' % time.strftime('%Y%m%d_%H%M%S'))
    json.dump(v, open(fn, 'w'), indent=1)
    return v

# ----------------------------------------------------------------------------------------------
# Stations and feature bookkeeping
# ----------------------------------------------------------------------------------------------
STATIONS_ALL = ['cerkezkoy', 'hamzebeyli', 'meric', 'muratli', 'kesan', 'ipsala', 'enez', 'malkara',
                'edirne', 'tekirdag', 'uzunkopru', 'saray', 'havsa', 'hayrabolu', 'lalapasa', 'corlu',
                'marmara', 'kemal', 'havalimani']
STATIONS_DROPPED = ['hamzebeyli', 'kemal', 'havalimani']          # >10 % missing -> removed
STATIONS_USED = [s for s in STATIONS_ALL if s not in STATIONS_DROPPED]
STATION_LABEL = {'cerkezkoy': 'Çerkezköy', 'hamzebeyli': 'Hamzabeyli', 'meric': 'Meriç', 'muratli': 'Muratlı',
                 'kesan': 'Keşan', 'ipsala': 'İpsala', 'enez': 'Enez', 'malkara': 'Malkara',
                 'edirne': 'Edirne', 'tekirdag': 'Tekirdağ', 'uzunkopru': 'Uzunköprü', 'saray': 'Saray',
                 'havsa': 'Havsa', 'hayrabolu': 'Hayrabolu', 'lalapasa': 'Lalapaşa', 'corlu': 'Çorlu',
                 'marmara': 'Marmara Ereğlisi', 'kemal': 'Kemal', 'havalimani': 'Havalimanı',
                 'sarkoy': 'Şarköy', 'ganos': 'Ganos'}
SOURCE, TARGET = 'mean_wind_speed', 'target'
FEATURES_559 = ["max_pressure", "max_humidity", "max_wind_speed", "max_temperature", "min_pressure", "min_humidity", "min_temperature", "mean_pressure", "mean_wind_speed", "mean_temperature", "total_precipitation", "max_humidity_at_ganos", "max_temperature_at_ganos", "min_humidity_at_ganos", "min_temperature_at_ganos", "mean_humidity_at_ganos", "mean_temperature_at_ganos", "total_precipitation_at_ganos", "max_wind_speed_horizontal", "max_wind_speed_vertical", "mean_wind_speed_horizontal", "mean_wind_speed_vertical", "max_speed_cerkezkoy_horizontal", "max_speed_cerkezkoy_vertical", "mean_speed_cerkezkoy_horizontal", "mean_speed_cerkezkoy_vertical", "max_speed_meric_horizontal", "max_speed_meric_vertical", "mean_speed_meric_horizontal", "mean_speed_meric_vertical", "max_speed_muratli_horizontal", "max_speed_muratli_vertical", "mean_speed_muratli_horizontal", "mean_speed_muratli_vertical", "max_speed_kesan_horizontal", "max_speed_kesan_vertical", "mean_speed_kesan_horizontal", "mean_speed_kesan_vertical", "max_speed_ipsala_horizontal", "max_speed_ipsala_vertical", "mean_speed_ipsala_horizontal", "mean_speed_ipsala_vertical", "max_speed_enez_horizontal", "max_speed_enez_vertical", "mean_speed_enez_horizontal", "mean_speed_enez_vertical", "max_speed_malkara_horizontal", "max_speed_malkara_vertical", "mean_speed_malkara_horizontal", "mean_speed_malkara_vertical", "max_speed_edirne_horizontal", "max_speed_edirne_vertical", "mean_speed_edirne_horizontal", "mean_speed_edirne_vertical", "max_speed_tekirdag_horizontal", "max_speed_tekirdag_vertical", "mean_speed_tekirdag_horizontal", "mean_speed_tekirdag_vertical", "max_speed_uzunkopru_horizontal", "max_speed_uzunkopru_vertical", "mean_speed_uzunkopru_horizontal", "mean_speed_uzunkopru_vertical", "max_speed_saray_horizontal", "max_speed_saray_vertical", "mean_speed_saray_horizontal", "mean_speed_saray_vertical", "max_speed_havsa_horizontal", "max_speed_havsa_vertical", "mean_speed_havsa_horizontal", "mean_speed_havsa_vertical", "max_speed_hayrabolu_horizontal", "max_speed_hayrabolu_vertical", "mean_speed_hayrabolu_horizontal", "mean_speed_hayrabolu_vertical", "max_speed_lalapasa_horizontal", "max_speed_lalapasa_vertical", "mean_speed_lalapasa_horizontal", "mean_speed_lalapasa_vertical", "max_speed_corlu_horizontal", "max_speed_corlu_vertical", "mean_speed_corlu_horizontal", "mean_speed_corlu_vertical", "max_speed_marmara_horizontal", "max_speed_marmara_vertical", "mean_speed_marmara_horizontal", "mean_speed_marmara_vertical", "max_wind_angle_sin", "max_wind_angle_cos", "mean_wind_angle_sin", "mean_wind_angle_cos", "mid_wind_speed", "lag_1_mean_temperature", "lag_2_mean_temperature", "lag_1_min_temperature", "lag_2_min_temperature", "lag_1_max_temperature", "lag_2_max_temperature", "lag_1_mean_humidity", "lag_2_mean_humidity", "lag_1_min_humidity", "lag_2_min_humidity", "lag_1_max_humidity", "lag_2_max_humidity", "lag_1_mean_pressure", "lag_2_mean_pressure", "lag_1_min_pressure", "lag_2_min_pressure", "lag_1_max_pressure", "lag_2_max_pressure", "lag_1_total_precipitation", "lag_2_total_precipitation", "lag_1_mean_wind_speed", "lag_2_mean_wind_speed", "lag_1_max_wind_speed", "lag_2_max_wind_speed", "lag_1_mean_wind_speed_horizontal", "lag_2_mean_wind_speed_horizontal", "lag_1_mean_wind_speed_vertical", "lag_2_mean_wind_speed_vertical", "lag_1_max_wind_speed_horizontal", "lag_2_max_wind_speed_horizontal", "lag_1_max_wind_speed_vertical", "lag_2_max_wind_speed_vertical", "lag_1_mid_wind_speed", "lag_2_mid_wind_speed", "lag_1_mean_temperature_at_ganos", "lag_2_mean_temperature_at_ganos", "lag_1_max_temperature_at_ganos", "lag_2_max_temperature_at_ganos", "lag_1_min_temperature_at_ganos", "lag_2_min_temperature_at_ganos", "lag_1_mean_humidity_at_ganos", "lag_2_mean_humidity_at_ganos", "lag_1_max_humidity_at_ganos", "lag_2_max_humidity_at_ganos", "lag_1_min_humidity_at_ganos", "lag_2_min_humidity_at_ganos", "lag_1_total_precipitation_at_ganos", "lag_2_total_precipitation_at_ganos", "day_of_week", "day_of_year_sin", "day_of_year_cos", "season_custom", "season_custom_sin", "season_custom_cos", "month_1", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10", "month_11", "month_12", "nightdark_hours", "sarkoy_temp_mean_to_max_ratio", "sarkoy_temp_mean_relative_pos", "sarkoy_temp_max_to_mean_ratio", "sarkoy_temp_mean_to_min_ratio", "sarkoy_hum_mean_to_max_ratio", "sarkoy_hum_range", "sarkoy_hum_mean_relative_pos", "sarkoy_hum_max_to_mean_ratio", "sarkoy_hum_mean_to_min_ratio", "sarkoy_press_range", "sarkoy_press_mean_relative_pos", "sarkoy_press_max_to_mean_ratio", "sarkoy_press_mean_to_min_ratio", "sarkoy_wind_mean_to_max_ratio", "abs_max_wind_speed_horizontal", "square_max_wind_speed_horizontal", "abs_max_wind_speed_vertical", "square_max_wind_speed_vertical", "abs_mean_wind_speed_horizontal", "square_mean_wind_speed_horizontal", "abs_mean_wind_speed_vertical", "square_mean_wind_speed_vertical", "abs_max_speed_cerkezkoy_horizontal", "square_max_speed_cerkezkoy_horizontal", "abs_max_speed_cerkezkoy_vertical", "square_max_speed_cerkezkoy_vertical", "abs_mean_speed_cerkezkoy_horizontal", "square_mean_speed_cerkezkoy_horizontal", "abs_mean_speed_cerkezkoy_vertical", "square_mean_speed_cerkezkoy_vertical", "abs_max_speed_meric_horizontal", "square_max_speed_meric_horizontal", "abs_max_speed_meric_vertical", "square_max_speed_meric_vertical", "abs_mean_speed_meric_horizontal", "square_mean_speed_meric_horizontal", "abs_mean_speed_meric_vertical", "square_mean_speed_meric_vertical", "abs_max_speed_muratli_horizontal", "square_max_speed_muratli_horizontal", "abs_max_speed_muratli_vertical", "square_max_speed_muratli_vertical", "abs_mean_speed_muratli_horizontal", "square_mean_speed_muratli_horizontal", "abs_mean_speed_muratli_vertical", "square_mean_speed_muratli_vertical", "abs_max_speed_kesan_horizontal", "square_max_speed_kesan_horizontal", "abs_max_speed_kesan_vertical", "square_max_speed_kesan_vertical", "abs_mean_speed_kesan_horizontal", "square_mean_speed_kesan_horizontal", "abs_mean_speed_kesan_vertical", "square_mean_speed_kesan_vertical", "abs_max_speed_ipsala_horizontal", "square_max_speed_ipsala_horizontal", "abs_max_speed_ipsala_vertical", "square_max_speed_ipsala_vertical", "abs_mean_speed_ipsala_horizontal", "square_mean_speed_ipsala_horizontal", "abs_mean_speed_ipsala_vertical", "square_mean_speed_ipsala_vertical", "abs_max_speed_enez_horizontal", "square_max_speed_enez_horizontal", "abs_max_speed_enez_vertical", "square_max_speed_enez_vertical", "abs_mean_speed_enez_horizontal", "square_mean_speed_enez_horizontal", "abs_mean_speed_enez_vertical", "square_mean_speed_enez_vertical", "abs_max_speed_malkara_horizontal", "square_max_speed_malkara_horizontal", "abs_max_speed_malkara_vertical", "square_max_speed_malkara_vertical", "abs_mean_speed_malkara_horizontal", "square_mean_speed_malkara_horizontal", "abs_mean_speed_malkara_vertical", "square_mean_speed_malkara_vertical", "abs_max_speed_edirne_horizontal", "square_max_speed_edirne_horizontal", "abs_max_speed_edirne_vertical", "square_max_speed_edirne_vertical", "abs_mean_speed_edirne_horizontal", "square_mean_speed_edirne_horizontal", "abs_mean_speed_edirne_vertical", "square_mean_speed_edirne_vertical", "abs_max_speed_tekirdag_horizontal", "square_max_speed_tekirdag_horizontal", "abs_max_speed_tekirdag_vertical", "square_max_speed_tekirdag_vertical", "abs_mean_speed_tekirdag_horizontal", "square_mean_speed_tekirdag_horizontal", "abs_mean_speed_tekirdag_vertical", "square_mean_speed_tekirdag_vertical", "abs_max_speed_uzunkopru_horizontal", "square_max_speed_uzunkopru_horizontal", "abs_max_speed_uzunkopru_vertical", "square_max_speed_uzunkopru_vertical", "abs_mean_speed_uzunkopru_horizontal", "square_mean_speed_uzunkopru_horizontal", "abs_mean_speed_uzunkopru_vertical", "square_mean_speed_uzunkopru_vertical", "abs_max_speed_saray_horizontal", "square_max_speed_saray_horizontal", "abs_max_speed_saray_vertical", "square_max_speed_saray_vertical", "abs_mean_speed_saray_horizontal", "square_mean_speed_saray_horizontal", "abs_mean_speed_saray_vertical", "square_mean_speed_saray_vertical", "abs_max_speed_havsa_horizontal", "square_max_speed_havsa_horizontal", "abs_max_speed_havsa_vertical", "square_max_speed_havsa_vertical", "abs_mean_speed_havsa_horizontal", "square_mean_speed_havsa_horizontal", "abs_mean_speed_havsa_vertical", "square_mean_speed_havsa_vertical", "abs_max_speed_hayrabolu_horizontal", "square_max_speed_hayrabolu_horizontal", "abs_max_speed_hayrabolu_vertical", "square_max_speed_hayrabolu_vertical", "abs_mean_speed_hayrabolu_horizontal", "square_mean_speed_hayrabolu_horizontal", "abs_mean_speed_hayrabolu_vertical", "square_mean_speed_hayrabolu_vertical", "abs_max_speed_lalapasa_horizontal", "square_max_speed_lalapasa_horizontal", "abs_max_speed_lalapasa_vertical", "square_max_speed_lalapasa_vertical", "abs_mean_speed_lalapasa_horizontal", "square_mean_speed_lalapasa_horizontal", "abs_mean_speed_lalapasa_vertical", "square_mean_speed_lalapasa_vertical", "abs_max_speed_corlu_horizontal", "square_max_speed_corlu_horizontal", "abs_max_speed_corlu_vertical", "square_max_speed_corlu_vertical", "abs_mean_speed_corlu_horizontal", "square_mean_speed_corlu_horizontal", "abs_mean_speed_corlu_vertical", "square_mean_speed_corlu_vertical", "abs_max_speed_marmara_horizontal", "square_max_speed_marmara_horizontal", "abs_max_speed_marmara_vertical", "square_max_speed_marmara_vertical", "abs_mean_speed_marmara_horizontal", "square_mean_speed_marmara_horizontal", "abs_mean_speed_marmara_vertical", "square_mean_speed_marmara_vertical", "abs_lag_1_mean_wind_speed_horizontal", "square_lag_1_mean_wind_speed_horizontal", "abs_lag_2_mean_wind_speed_horizontal", "square_lag_2_mean_wind_speed_horizontal", "abs_lag_1_mean_wind_speed_vertical", "square_lag_1_mean_wind_speed_vertical", "abs_lag_2_mean_wind_speed_vertical", "square_lag_2_mean_wind_speed_vertical", "abs_lag_1_max_wind_speed_horizontal", "square_lag_1_max_wind_speed_horizontal", "abs_lag_2_max_wind_speed_horizontal", "square_lag_2_max_wind_speed_horizontal", "abs_lag_1_max_wind_speed_vertical", "square_lag_1_max_wind_speed_vertical", "abs_lag_2_max_wind_speed_vertical", "square_lag_2_max_wind_speed_vertical", "air_density", "potential_temp_c", "wind_power_proxy", "diurnal_temp_range", "dew_point_approx", "dryness_index", "wind_power_moisture", "days_since_high_wind", "days_since_low_wind", "lunar_sin", "lunar_cos", "is_full_moon", "is_new_moon", "temp_diff_sarkoy_ganos", "wind_speed_diff", "wind_angle_diff_2", "sarkoy_rain_event", "sarkoy_log_precipitation", "sarkoy_days_since_last_rain", "sarkoy_light_rain", "sarkoy_moderate_rain", "sarkoy_heavy_rain", "sarkoy_annual_cum_precip", "ganos_rain_event", "ganos_log_precipitation", "ganos_days_since_last_rain", "ganos_light_rain", "ganos_moderate_rain", "ganos_heavy_rain", "ganos_annual_cum_precip", "sarkoy_consecutive_rain_event_days", "ganos_consecutive_rain_event_days", "is_high_wind_day", "consecutive_high_wind_days", "sarkoy_wind_dir_streak", "humidity_wind_interaction", "lag_humidity_wind_interaction", "diff_1_mean_pressure", "temp_humidity_ratio", "sqrt_mean_wind_speed", "precip_wind_interaction", "lag_1_mean_speed_havsa_horizontal", "lag_1_max_speed_havsa_horizontal", "lag_1_mean_speed_havsa_vertical", "lag_1_max_speed_havsa_vertical", "lag_1_mean_speed_lalapasa_horizontal", "lag_1_max_speed_lalapasa_horizontal", "lag_1_mean_speed_lalapasa_vertical", "lag_1_max_speed_lalapasa_vertical", "lag_1_mean_speed_tekirdag_horizontal", "lag_1_max_speed_tekirdag_horizontal", "lag_1_mean_speed_tekirdag_vertical", "lag_1_max_speed_tekirdag_vertical", "lag_1_mean_speed_corlu_horizontal", "lag_1_max_speed_corlu_horizontal", "lag_1_mean_speed_corlu_vertical", "lag_1_max_speed_corlu_vertical", "lag_1_mean_speed_meric_horizontal", "lag_1_max_speed_meric_horizontal", "lag_1_mean_speed_meric_vertical", "lag_1_max_speed_meric_vertical", "lag_1_mean_speed_kesan_horizontal", "lag_1_max_speed_kesan_horizontal", "lag_1_mean_speed_kesan_vertical", "lag_1_max_speed_kesan_vertical", "lag_1_mean_speed_cerkezkoy_horizontal", "lag_1_max_speed_cerkezkoy_horizontal", "lag_1_mean_speed_cerkezkoy_vertical", "lag_1_max_speed_cerkezkoy_vertical", "lag_1_mean_speed_muratli_horizontal", "lag_1_max_speed_muratli_horizontal", "lag_1_mean_speed_muratli_vertical", "lag_1_max_speed_muratli_vertical", "lag_1_mean_speed_ipsala_horizontal", "lag_1_max_speed_ipsala_horizontal", "lag_1_mean_speed_ipsala_vertical", "lag_1_max_speed_ipsala_vertical", "lag_1_mean_speed_enez_horizontal", "lag_1_max_speed_enez_horizontal", "lag_1_mean_speed_enez_vertical", "lag_1_max_speed_enez_vertical", "lag_1_mean_speed_malkara_horizontal", "lag_1_max_speed_malkara_horizontal", "lag_1_mean_speed_malkara_vertical", "lag_1_max_speed_malkara_vertical", "lag_1_mean_speed_uzunkopru_horizontal", "lag_1_max_speed_uzunkopru_horizontal", "lag_1_mean_speed_uzunkopru_vertical", "lag_1_max_speed_uzunkopru_vertical", "lag_1_mean_speed_saray_horizontal", "lag_1_max_speed_saray_horizontal", "lag_1_mean_speed_saray_vertical", "lag_1_max_speed_saray_vertical", "lag_1_mean_speed_marmara_horizontal", "lag_1_max_speed_marmara_horizontal", "lag_1_mean_speed_marmara_vertical", "lag_1_max_speed_marmara_vertical", "lag_1_mean_speed_edirne_horizontal", "lag_1_max_speed_edirne_horizontal", "lag_1_mean_speed_edirne_vertical", "lag_1_max_speed_edirne_vertical", "lag_1_mean_speed_hayrabolu_horizontal", "lag_1_max_speed_hayrabolu_horizontal", "lag_1_mean_speed_hayrabolu_vertical", "lag_1_max_speed_hayrabolu_vertical", "hum_diff_sarkoy_ganos", "temp_diff_sarkoy_ganos_lag_1", "temp_diff_sarkoy_ganos_lag_2", "hum_diff_sarkoy_ganos_lag_1", "hum_diff_sarkoy_ganos_lag_2", "mean_wind_speed_roll_3_volatility", "mean_wind_speed_ema_3", "mean_wind_speed_roll_7_volatility", "mean_wind_speed_ema_7", "mean_wind_speed_dev_from_ema_7", "mean_wind_speed_roll_14_volatility", "mean_wind_speed_ema_14", "mean_wind_speed_dev_from_ema_14", "mean_wind_speed_roll_21_volatility", "mean_wind_speed_ema_21", "mean_wind_speed_dev_from_ema_21", "mean_wind_speed_roll_30_volatility", "mean_wind_speed_ema_30", "mean_wind_speed_dev_from_ema_30", "log_mean_wind_speed", "wind_season_sin_interaction", "wind_season_cos_interaction", "corr_sarkoy_havsa_horizontal", "corr_sarkoy_lalapasa_horizontal", "corr_sarkoy_tekirdag_horizontal", "corr_sarkoy_corlu_horizontal", "corr_sarkoy_meric_horizontal", "corr_sarkoy_kesan_horizontal", "corr_sarkoy_cerkezkoy_horizontal", "corr_sarkoy_muratli_horizontal", "corr_sarkoy_ipsala_horizontal", "corr_sarkoy_enez_horizontal", "corr_sarkoy_malkara_horizontal", "corr_sarkoy_uzunkopru_horizontal", "corr_sarkoy_saray_horizontal", "corr_sarkoy_marmara_horizontal", "corr_sarkoy_edirne_horizontal", "corr_sarkoy_hayrabolu_horizontal", "cube_mean_wind_speed", "pressure_humidity_interaction", "vol_season_sin_3", "vol_season_cos_3", "vol_season_sin_30", "vol_season_cos_30", "wind_speed_ratio", "temp_precip_interaction", "sarkoy_mean_wind_dir_cat_E", "sarkoy_mean_wind_dir_cat_N", "sarkoy_mean_wind_dir_cat_NE", "sarkoy_mean_wind_dir_cat_NW", "sarkoy_mean_wind_dir_cat_S", "sarkoy_mean_wind_dir_cat_SE", "sarkoy_mean_wind_dir_cat_SW", "sarkoy_mean_wind_dir_cat_W", "lag1_sarkoy_mean_wind_dir_cat_E", "lag1_sarkoy_mean_wind_dir_cat_N", "lag1_sarkoy_mean_wind_dir_cat_NE", "lag1_sarkoy_mean_wind_dir_cat_NW", "lag1_sarkoy_mean_wind_dir_cat_S", "lag1_sarkoy_mean_wind_dir_cat_SE", "lag1_sarkoy_mean_wind_dir_cat_SW", "lag1_sarkoy_mean_wind_dir_cat_W", "wind_dir_cat_concat_E_E", "wind_dir_cat_concat_E_N", "wind_dir_cat_concat_E_NE", "wind_dir_cat_concat_E_NW", "wind_dir_cat_concat_E_S", "wind_dir_cat_concat_E_SE", "wind_dir_cat_concat_E_SW", "wind_dir_cat_concat_E_W", "wind_dir_cat_concat_NE_E", "wind_dir_cat_concat_NE_N", "wind_dir_cat_concat_NE_NE", "wind_dir_cat_concat_NE_NW", "wind_dir_cat_concat_NE_S", "wind_dir_cat_concat_NE_SE", "wind_dir_cat_concat_NE_SW", "wind_dir_cat_concat_NE_W", "wind_dir_cat_concat_NW_E", "wind_dir_cat_concat_NW_N", "wind_dir_cat_concat_NW_NE", "wind_dir_cat_concat_NW_NW", "wind_dir_cat_concat_NW_S", "wind_dir_cat_concat_NW_SE", "wind_dir_cat_concat_NW_SW", "wind_dir_cat_concat_NW_W", "wind_dir_cat_concat_N_E", "wind_dir_cat_concat_N_N", "wind_dir_cat_concat_N_NE", "wind_dir_cat_concat_N_NW", "wind_dir_cat_concat_N_S", "wind_dir_cat_concat_N_SE", "wind_dir_cat_concat_N_SW", "wind_dir_cat_concat_N_W", "wind_dir_cat_concat_SE_E", "wind_dir_cat_concat_SE_N", "wind_dir_cat_concat_SE_NE", "wind_dir_cat_concat_SE_NW", "wind_dir_cat_concat_SE_S", "wind_dir_cat_concat_SE_SW", "wind_dir_cat_concat_SE_W", "wind_dir_cat_concat_SW_E", "wind_dir_cat_concat_SW_N", "wind_dir_cat_concat_SW_NE", "wind_dir_cat_concat_SW_NW", "wind_dir_cat_concat_SW_S", "wind_dir_cat_concat_SW_SW", "wind_dir_cat_concat_SW_W", "wind_dir_cat_concat_S_E", "wind_dir_cat_concat_S_N", "wind_dir_cat_concat_S_NE", "wind_dir_cat_concat_S_NW", "wind_dir_cat_concat_S_S", "wind_dir_cat_concat_S_SE", "wind_dir_cat_concat_S_SW", "wind_dir_cat_concat_S_W", "wind_dir_cat_concat_W_E", "wind_dir_cat_concat_W_N", "wind_dir_cat_concat_W_NE", "wind_dir_cat_concat_W_NW", "wind_dir_cat_concat_W_S", "wind_dir_cat_concat_W_SE", "wind_dir_cat_concat_W_SW", "wind_dir_cat_concat_W_W", "vol_season_sin_7", "vol_season_cos_7", "vol_season_sin_14", "vol_season_cos_14", "wind_speed_momentum", "total_precipitation_roll_7_volatility"]

LOCAL_RAW_WIND = ['mean_wind_speed', 'max_wind_speed', 'mean_wind_speed_horizontal', 'mean_wind_speed_vertical',
                  'max_wind_speed_horizontal', 'max_wind_speed_vertical']
PHYSICS_KEYS = ['air_density', 'potential_temp', 'dew_point', 'dryness', 'moisture', 'wind_power', 'power_proxy',
                'diurnal_temp_range']
CALENDAR_KEYS = ['day_of_week', 'day_of_year', 'season', 'month_', 'nightdark', 'daylight']
LUNAR_KEYS = ['lunar', 'moon']


def _station_in(name):
    for s in STATIONS_ALL:
        if f'_{s}_' in f'_{name}_':
            return s
    return None


def feature_group(name):
    """Fine-grained group of a predictor (used for grouped importance and the ablation ladder)."""
    n = name.lower()
    st = _station_in(n)
    if st is not None:
        base = f'{"max" if "max_speed" in n else "mean"}_speed_{st}_'
        raw = n in (base + 'horizontal', base + 'vertical')
        return f'neighbour_raw:{st}' if raw else f'neighbour_eng:{st}'
    if 'ganos' in n:
        return 'ganos'
    if any(k in n for k in LUNAR_KEYS):
        return 'lunar'
    if any(k in n for k in CALENDAR_KEYS) and 'vol_season' not in n and 'wind_season' not in n:
        return 'calendar'
    if 'dir_cat' in n or 'dir_streak' in n:
        return 'local_direction'
    base = n
    for p in ('lag_1_', 'lag_2_'):
        if base.startswith(p):
            base = base[len(p):]
    if base in LOCAL_RAW_WIND:
        return 'local_raw_wind'
    if any(k in n for k in PHYSICS_KEYS):
        return 'physics'
    if 'wind' in n or 'speed' in n:
        return 'local_wind_eng'
    if base in ('max_pressure', 'min_pressure', 'mean_pressure', 'max_temperature', 'min_temperature',
                'mean_temperature', 'max_humidity', 'min_humidity', 'mean_humidity', 'total_precipitation'):
        return 'local_met'
    return 'local_met_eng'


def group_table(features):
    return pd.DataFrame({'feature': features, 'group': [feature_group(f) for f in features]})


def coarse_group(g):
    """Coarse groups for grouped permutation importance."""
    if g.startswith('neighbour'):
        return 'station:' + g.split(':')[1]
    return {'local_raw_wind': 'Şarköy same-day/lagged wind', 'local_wind_eng': 'Şarköy wind statistics',
            'physics': 'Şarköy thermodynamics/physics', 'local_met': 'Şarköy meteorology',
            'local_met_eng': 'Şarköy meteorology (engineered)', 'local_direction': 'Şarköy direction classes',
            'calendar': 'Calendar/astronomical', 'lunar': 'Lunar phase (negative control)',
            'ganos': 'Ganos'}.get(g, g)


def ablation_sets(features):
    """Nested feature sets for the ablation ladder requested by Reviewer 4 (comment 8) plus the
    physics-only / local-only / neighbour-only variants requested by Reviewer 3 (comment 3)."""
    g = {f: feature_group(f) for f in features}
    local = [f for f in features if g[f] in ('local_raw_wind', 'local_wind_eng', 'physics', 'local_met',
                                            'local_met_eng', 'local_direction', 'calendar', 'lunar')]
    sets = {
        'b_local_raw_wind': [f for f in features if g[f] == 'local_raw_wind'],
        'c_local_engineered': local,
        'd_local_plus_ganos': local + [f for f in features if g[f] == 'ganos'],
    }
    sets['e_plus_neighbour_raw'] = sets['d_local_plus_ganos'] + [f for f in features if g[f].startswith('neighbour_raw')]
    sets['f_full'] = list(features)
    sets['x_neighbour_only'] = [f for f in features if g[f].startswith('neighbour')]
    sets['x_physics_guided'] = [f for f in features if g[f] in ('local_raw_wind', 'local_met', 'physics')]
    return sets

# ----------------------------------------------------------------------------------------------
# Readable names (eastward/northward terminology requested by Reviewer 4, comment 9)
# ----------------------------------------------------------------------------------------------
def pretty(name):
    n = name
    wrap = None
    if n.startswith('abs_'):
        wrap, n = 'abs', n[4:]
    elif n.startswith('square_'):
        wrap, n = 'sq', n[7:]
    lag = ''
    for p, lab in (('lag_1_', ' (t−1)'), ('lag_2_', ' (t−2)')):
        if n.startswith(p):
            n, lag = n[len(p):], lab
    comp = {'horizontal': 'u', 'vertical': 'v'}
    parts = n.split('_')
    st = _station_in(n)
    if n.startswith('corr_sarkoy_') and st:
        lab = f'7-d corr. Şarköy–{STATION_LABEL[st]} u'
    elif st and ('speed' in n):
        kind = 'max' if n.startswith('max') else 'mean'
        c = comp.get(parts[-1], '')
        lab = f'{STATION_LABEL[st]} {kind} {c}'.strip()
    elif n in ('mean_wind_speed_horizontal', 'mean_wind_speed_vertical', 'max_wind_speed_horizontal',
               'max_wind_speed_vertical'):
        lab = f'Şarköy {parts[0]} {comp[parts[-1]]}'
    elif n == 'mid_wind_speed':
        lab = 'Şarköy mid wind speed'
    elif n in ('mean_wind_speed', 'max_wind_speed'):
        lab = f'Şarköy {parts[0]} wind speed'
    else:
        lab = n.replace('_', ' ')
        lab = lab[:1].upper() + lab[1:]
    if wrap == 'abs':
        lab = f'|{lab}|'
    elif wrap == 'sq':
        lab = f'({lab})²'
    return lab + lag

# ----------------------------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------------------------
def load_matrix(base, rev, which='causal', verbose=True):
    """'causal'   -> revision/df_final_features_causal.feather (leakage-free rebuild, notebook 07)
       'published'-> df_final_features.feather (matrix behind the submitted manuscript)"""
    fn = os.path.join(rev, 'df_final_features_causal.feather') if which == 'causal' else \
        os.path.join(base, 'df_final_features.feather')
    if which == 'causal' and not os.path.exists(fn):
        raise FileNotFoundError('Run notebook 07 first: ' + fn)
    df = pd.read_feather(fn)
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.set_index('date_time')
    df = df.sort_index()
    if 'target_observed' not in df.columns:
        df['target_observed'] = True
    df['target_observed'] = df['target_observed'].astype(bool)
    if verbose:
        print(f'Loaded {which} matrix: {df.shape}, {df.index.min().date()} -> {df.index.max().date()}, '
              f'observed targets: {int(df.target_observed.sum())}')
    return df


def get_X(df, features=None):
    features = features or FEATURES_559
    miss = [f for f in features if f not in df.columns]
    if miss:
        warnings.warn(f'{len(miss)} expected predictors missing (filled with 0): {miss[:5]} ...')
        df = df.copy()
        for f in miss:
            df[f] = 0.0
    X = df[features].astype(float)
    return X


def load_raw_sarkoy(base):
    s = pd.read_json(os.path.join(base, 'data', 'Meteorology_Sarkoy_2014_2024.json'))
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def observed_target_mask(index, raw_sarkoy):
    """True where the NEXT calendar day has an observed (non-missing) Şarköy mean wind speed."""
    obs = raw_sarkoy[SOURCE].notna()
    nxt = pd.DatetimeIndex(index) + pd.Timedelta(days=1)
    return pd.Series(obs.reindex(nxt).fillna(False).values, index=index)

# ----------------------------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------------------------
def baseline_predictions(df, train_mask):
    """Persistence, 7-day moving average and a seasonal-naive climatology that uses TRAINING rows only
    (the submitted version averaged over all years, which let the test year into this baseline)."""
    out = pd.DataFrame(index=df.index)
    out['persistence'] = df[SOURCE]
    out['moving_average_7d'] = df[SOURCE].rolling(7, min_periods=1).mean()
    doy = df.index.dayofyear
    clim = df.loc[train_mask, TARGET].groupby(doy[train_mask]).mean()
    clim = clim.reindex(range(1, 367)).interpolate(limit_direction='both')
    out['seasonal_naive'] = clim.reindex(doy).values
    out['persistence_2d'] = df[SOURCE].shift(1)        # origin one day earlier (market-compatible check)
    return out

# ----------------------------------------------------------------------------------------------
# Metrics, block bootstrap, Diebold-Mariano
# ----------------------------------------------------------------------------------------------
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def maxae(y, p): return float(np.max(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def metrics(y, p):
    return {'RMSE': rmse(y, p), 'MAE': mae(y, p), 'R2': r2(y, p), 'MaxAE': maxae(y, p), 'n': int(len(y))}


def mbb_indices(n, block, rng):
    k = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block, 0) + 1, size=k)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return idx


def block_bootstrap(y, preds, block=7, B=2000, seed=42, groups=None):
    """Moving-block bootstrap (Künsch 1989). preds: dict name -> array. groups: optional array of
    stratum labels (e.g. outer year) - blocks are resampled within each stratum and then pooled.
    Returns dict name -> dict(metric -> array of B values)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    P = {k: np.asarray(v, float) for k, v in preds.items()}
    if groups is None:
        strata = [np.arange(len(y))]
    else:
        g = np.asarray(groups)
        strata = [np.where(g == u)[0] for u in pd.unique(g)]
    out = {k: {'RMSE': np.empty(B), 'MAE': np.empty(B), 'R2': np.empty(B)} for k in P}
    for b in range(B):
        idx = np.concatenate([s[mbb_indices(len(s), min(block, len(s)), rng)] for s in strata])
        yb = y[idx]
        for k, p in P.items():
            pb = p[idx]
            out[k]['RMSE'][b] = rmse(yb, pb)
            out[k]['MAE'][b] = mae(yb, pb)
            out[k]['R2'][b] = r2(yb, pb)
    return out


def ci(a, alpha=0.05):
    a = np.asarray(a)
    return float(np.nanpercentile(a, 100 * alpha / 2)), float(np.nanpercentile(a, 100 * (1 - alpha / 2)))


def dm_test(e_a, e_b, h=1, power=2, lag=None):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction and a
    Newey-West (Bartlett) long-run variance. d = |e_a|^p - |e_b|^p ; negative mean -> model a better.
    Returns (DM statistic, two-sided p-value, mean loss differential)."""
    from scipy import stats
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    d = np.abs(e_a) ** power - np.abs(e_b) ** power
    n = len(d)
    dbar = d.mean()
    if lag is None:
        lag = max(h - 1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    dc = d - dbar
    gamma0 = np.dot(dc, dc) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        lrv += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    lrv = max(lrv, 1e-12)
    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return float(stat), float(p), float(dbar)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj

# ----------------------------------------------------------------------------------------------
# Month-specific scaling for linear models / SVR (as in the original modelling notebook)
# ----------------------------------------------------------------------------------------------
class MonthScaled:
    """Wraps an estimator; fits one StandardScaler per calendar month on the training rows only."""
    def __init__(self, estimator):
        self.estimator = estimator

    def _tx(self, X, fit=False):
        from sklearn.preprocessing import StandardScaler
        Xs = np.empty(X.shape, float)
        months = pd.DatetimeIndex(X.index).month
        if fit:
            self.scalers = {}
            for m in np.unique(months):
                self.scalers[m] = StandardScaler().fit(X.values[months == m])
            self.global_ = StandardScaler().fit(X.values)
        for m in np.unique(months):
            sc = self.scalers.get(m, self.global_)
            Xs[months == m] = sc.transform(X.values[months == m])
        return np.nan_to_num(Xs)

    def fit(self, X, y):
        self.estimator.fit(self._tx(X, fit=True), np.asarray(y))
        return self

    def predict(self, X):
        return self.estimator.predict(self._tx(X))

# ----------------------------------------------------------------------------------------------
# Model factory with the hyperparameters reported in Table S2 (submitted version)
# ----------------------------------------------------------------------------------------------
PUBLISHED_PARAMS = {
    'XGBoost_50': dict(max_depth=6, learning_rate=0.01646338725108453, n_estimators=471, min_child_weight=4,
                       subsample=0.5287942212625109, colsample_bytree=0.557288672298357, gamma=0.1433760901202984,
                       reg_lambda=0.11347174478637617, reg_alpha=0.8791225318788727),
    'XGBoost_400': dict(n_estimators=1834, max_depth=11, learning_rate=0.004854856793781717,
                        subsample=0.40566526847932777, colsample_bytree=0.3968091211824199,
                        gamma=1.4684181309759596, min_child_weight=2, reg_alpha=0.9001406603742224,
                        reg_lambda=0.23250578488239765),
    'LightGBM': dict(num_leaves=138, learning_rate=0.01405669721982889, n_estimators=425, min_child_samples=23,
                     subsample=0.7826656508469495, colsample_bytree=0.6348504347088767,
                     reg_alpha=0.9994237846782444, reg_lambda=0.9940731906006898),
    'CatBoost': dict(iterations=468, learning_rate=0.013714778651531173, depth=6,
                     l2_leaf_reg=1.2648502968627125e-05, border_count=201),
    'RandomForest': dict(n_estimators=942, max_depth=19, min_samples_split=12, min_samples_leaf=5,
                         max_features=1.0, bootstrap=True),
    'SVR': dict(kernel='rbf', C=1.441814588208259, epsilon=0.37064901603771033, gamma='auto'),
}
LINEAR_ALPHAS = [0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10, 15,
                 20, 25, 30, 50, 100]
LINEAR_L1 = [0.1, 0.3, 0.5, 0.7, 0.9]


def make_model(name, params=None, gpu=None, fast=False):
    params = dict(params or {})
    gpu = has_gpu() if gpu is None else gpu
    if fast:  # smoke-test caps
        for k in ('n_estimators', 'iterations'):
            if k in params:
                params[k] = min(params[k], 40)
    if name.startswith('XGBoost'):
        import xgboost as xgb
        return xgb.XGBRegressor(objective='reg:squarederror', random_state=42, tree_method='hist',
                                device='cuda' if gpu else 'cpu', verbosity=0, **params)
    if name == 'LightGBM':
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=42, subsample_freq=1, verbose=-1, **params)
    if name == 'CatBoost':
        import catboost as cb
        return cb.CatBoostRegressor(random_seed=42, verbose=0, thread_count=-1, **params)
    if name == 'RandomForest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if name == 'SVR':
        from sklearn.svm import SVR
        return MonthScaled(SVR(**params))
    if name in ('LassoCV', 'RidgeCV', 'ElasticNetCV'):
        from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=6)
        est = {'LassoCV': LassoCV(cv=cv, alphas=LINEAR_ALPHAS, random_state=42, max_iter=20000 if fast else 200000),
               'RidgeCV': RidgeCV(cv=cv, alphas=LINEAR_ALPHAS),
               'ElasticNetCV': ElasticNetCV(cv=cv, alphas=LINEAR_ALPHAS, l1_ratio=LINEAR_L1, random_state=42,
                                            max_iter=20000 if fast else 200000)}[name]
        return MonthScaled(est)
    raise ValueError(name)


def reduce_features(X, y, importance_threshold=0.005, vif_max=10.0, n_trees=1000, fast=False):
    """Two-stage reduction of Section 2.4 (RF importance threshold, then iterative VIF pruning),
    fitted on the rows passed in (training rows only)."""
    from sklearn.ensemble import RandomForestRegressor
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    rf = RandomForestRegressor(n_estimators=50 if fast else n_trees, random_state=42, n_jobs=-1)
    rf.fit(X.values, np.asarray(y))
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    keep = list(imp[imp > importance_threshold].sort_values(ascending=False).index)
    if len(keep) < 3:
        keep = list(imp.sort_values(ascending=False).index[:10])
    while len(keep) > 2:
        Xs = X[keep].values
        Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
        vifs = [variance_inflation_factor(Xs, i) for i in range(len(keep))]
        j = int(np.nanargmax(vifs))
        if not np.isfinite(vifs[j]) or vifs[j] > vif_max:
            keep.pop(j)
        else:
            break
    return keep

# ----------------------------------------------------------------------------------------------
# Optuna search spaces (fixed a priori for the nested rolling-origin evaluation)
# ----------------------------------------------------------------------------------------------
def suggest(trial, name):
    if name == 'XGBoost':   # the 50-trial search space of the submitted study
        return dict(max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    gamma=trial.suggest_float('gamma', 0.0, 0.5),
                    reg_lambda=trial.suggest_float('reg_lambda', 0.0, 1.0),
                    reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0))
    if name == 'LightGBM':
        return dict(num_leaves=trial.suggest_int('num_leaves', 16, 256),
                    learning_rate=trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                    n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                    min_child_samples=trial.suggest_int('min_child_samples', 5, 60),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                    reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
                    reg_lambda=trial.suggest_float('reg_lambda', 0.0, 1.0))
    if name == 'CatBoost':
        return dict(iterations=trial.suggest_int('iterations', 200, 1000),
                    depth=trial.suggest_int('depth', 4, 8),
                    learning_rate=trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                    l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                    border_count=trial.suggest_int('border_count', 32, 255))
    if name == 'RandomForest':
        return dict(n_estimators=trial.suggest_int('n_estimators', 200, 1000),
                    max_depth=trial.suggest_int('max_depth', 5, 25),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    max_features=trial.suggest_categorical('max_features', [0.3, 0.6, 1.0]),
                    bootstrap=True)
    if name == 'SVR':
        return dict(kernel='rbf', C=trial.suggest_float('C', 0.1, 100.0, log=True),
                    epsilon=trial.suggest_float('epsilon', 0.01, 1.0),
                    gamma=trial.suggest_categorical('gamma', ['scale', 'auto']))
    raise ValueError(name)

# ----------------------------------------------------------------------------------------------
# Plot style (same look as visualization.ipynb) and figure saving
# ----------------------------------------------------------------------------------------------
def set_style():
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'ieee', 'no-latex'])
    except Exception:
        pass
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'lines.linewidth': 1.2, 'axes.linewidth': 0.8, 'legend.fontsize': 6,
                         'figure.dpi': 150, 'savefig.dpi': 600, 'figure.figsize': (3.3, 2.5),
                         'font.family': 'serif', 'mathtext.fontset': 'dejavusans'})


def savefig(fig, rev, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(rev, 'figures', f'{name}.{ext}'), bbox_inches='tight')
    print('saved figure', name)

# ----------------------------------------------------------------------------------------------
# Results store: every number quoted in the revised documents is written here under a {{KEY}}
# ----------------------------------------------------------------------------------------------
class Results:
    def __init__(self, rev):
        self.path = os.path.join(rev, 'revision_results.json')
        self.d = {}

    def put(self, key, value, fmt=None):
        if fmt is not None and isinstance(value, (int, float, np.floating, np.integer)):
            value = format(float(value), fmt) if not fmt.endswith('d') else format(int(round(float(value))), fmt)
        self.d[key] = str(value)
        return self

    def save(self):
        cur = json.load(open(self.path)) if os.path.exists(self.path) else {}
        cur.update(self.d)
        json.dump(cur, open(self.path, 'w'), indent=1, ensure_ascii=False, sort_keys=True)
        print(f'{len(self.d)} values written to {self.path}')


def fmt_ci(lo, hi, d=3):
    return f'{lo:.{d}f}–{hi:.{d}f}'


def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


# ----------------------------------------------------------------------------------------------
# Raw-record helpers (used by the audit and the leakage-free rebuild)
# ----------------------------------------------------------------------------------------------
MODEL_CODES = {'Persistence': 'PERS', 'Seasonal-naive': 'SNAIVE', '7-day moving average': 'MA7',
               'LassoCV': 'LASSO', 'RidgeCV': 'RIDGE', 'ElasticNetCV': 'ENET', 'SVR': 'SVR',
               'Random Forest': 'RF', 'LightGBM': 'LGBM', 'CatBoost': 'CAT',
               'XGBoost (50 trials)': 'XGB50', 'XGBoost (400 trials)': 'XGB400',
               'XGBoost (reduced set)': 'XGBRED', 'LSTM': 'LSTM'}


def raw_merged(base):
    """Rebuild the merged raw table exactly as features.ipynb does (inner join on the date index,
    '_diff_' columns dropped, u/v components added, columns with >10 % gaps removed).
    Returns (numeric frame of retained columns, dropped columns, sarkoy, trakya, ganos)."""
    sar = load_raw_sarkoy(base)
    trk = pd.read_csv(os.path.join(base, 'data', 'wind_speed_trakya.csv'), low_memory=False)
    trk['date_time'] = pd.to_datetime(trk['date_time'])
    trk = trk.set_index('date_time').sort_index()
    gan = pd.read_csv(os.path.join(base, 'data', 'Meteorology_Ganos_2016_Nov_2024.csv'))
    gan['date_time'] = pd.to_datetime(gan['Date'], errors='coerce')
    gan = gan.dropna(subset=['date_time']).drop(columns=['Date']).set_index('date_time').sort_index()
    raw = sar.join(trk, how='inner', rsuffix='_trakya').join(gan, how='inner')
    raw = raw[[c for c in raw.columns if '_diff_' not in c]]
    spd = [c for c in raw.columns if 'speed' in c and 'angle' not in c]
    ang = [c for c in raw.columns if 'angle' in c]
    new = {}
    for s, a in zip(spd, ang):
        sp = pd.to_numeric(raw[s], errors='coerce')
        an = pd.to_numeric(raw[a], errors='coerce')
        new[s + '_horizontal'] = -sp * np.sin(np.radians(an))
        new[s + '_vertical'] = -sp * np.cos(np.radians(an))
    raw = pd.concat([raw, pd.DataFrame(new, index=raw.index)], axis=1)
    num = raw.select_dtypes('number')
    miss = num.isna().mean() * 100
    dropped = list(miss[miss > 10.0].index)
    kept = [c for c in num.columns if c not in dropped]
    return num[kept], dropped, sar, trk, gan


def missing_flags(base):
    """Daily indicator per source: 1 if any retained variable of that source was missing that day."""
    num, dropped, *_ = raw_merged(base)
    M = num.isna()
    flags = {'miss_sarkoy': M[[c for c in M.columns if _station_in(c) is None and 'ganos' not in c]].any(axis=1),
             'miss_ganos': M[[c for c in M.columns if 'ganos' in c]].any(axis=1)}
    for s in STATIONS_USED:
        cols = [c for c in M.columns if _station_in(c) == s]
        if cols:
            flags['miss_' + s] = M[cols].any(axis=1)
    return pd.DataFrame(flags).astype(int)


def max_gap(series_isna):
    """Longest run of consecutive missing values in a boolean Series."""
    g = (~series_isna).cumsum()[series_isna]
    return int(g.value_counts().max()) if len(g) else 0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def event_scores(y, p, thr):
    """Conditional bias/RMSE, peak attenuation and categorical scores for high-wind days."""
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y, float), np.asarray(p, float)
    ev, fc = y >= thr, p >= thr
    hits, miss, fa = int((ev & fc).sum()), int((ev & ~fc).sum()), int((~ev & fc).sum())
    out = dict(n_events=int(ev.sum()), hits=hits, misses=miss, false_alarms=fa,
               POD=hits / (hits + miss) if hits + miss else np.nan,
               FAR=fa / (hits + fa) if hits + fa else np.nan,
               CSI=hits / (hits + miss + fa) if hits + miss + fa else np.nan,
               freq_bias=(hits + fa) / (hits + miss) if hits + miss else np.nan)
    if ev.sum():
        out.update(bias=float(np.mean(p[ev] - y[ev])), rmse=rmse(y[ev], p[ev]),
                   attenuation=float(np.mean(p[ev]) / np.mean(y[ev])))
    out['AUC'] = float(roc_auc_score(ev, p)) if 0 < ev.sum() < len(ev) else np.nan
    return out
