import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, numeric_columns: list, threshold: float = 0.05):
    """
    Detect data drift between two datasets using the Kolmogorov-Smirnov test.
    """
    drift_report = []

    for col in numeric_columns:
        if col in reference_df.columns and col in current_df.columns:
            result: KstestResult = ks_2samp(reference_df[col], current_df[col])
            p_value = result.pvalue
            drift_detected = p_value < threshold

            drift_report.append({
                "feature": col,
                "p_value": round(p_value, 4),
                "drift": drift_detected
            })

    return drift_report
