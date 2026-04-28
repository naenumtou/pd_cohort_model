
import warnings
import pandas as pd
import numpy as np

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper function
def _bound_bucket(
    n: int,
    start: float,
    diff_odr: float,
) -> tuple[np.array, np.array]:
    
    """
    Create buckting for KS-Test.

    Description:
        The bucket(s) have been created from minimum and maximum values from
        actual ODR and fitted ODR (Gamma). To build the buckets, starting with
        the smallest value from actual ODR and fitted ODR min(min, min) as starting point.
        Then, add the multiplication of difference value (largest odr - smallest odr) / (n - 2).
        The result is maximum bucket and then applying shift(1) and fillna first value = 0
        to derive the minimum bucket. 

    Args:
        n (int)             : The number of actual lifetime (NOT from Gamma fitted curve).
        start (float)       : The smallest value computing from min(min(actual ODR), min(fitted ODR)).
        diff_odr (float)    : The difference value computing from (largest odr - smallest odr) / (n - 2).

    Returns:
        np.array: The maximum bucket (upper bounds).
        np.array: The minimum bucket (lower bounds).

    Notes:
        - N/A.
    """

    idx = np.arange(n)
    bucket = start + (diff_odr * idx)
    bucket[0] = start
    bucket[-1] = 1.0
    max_buc = pd.Series(bucket)
    min_buc = max_buc.shift(1).fillna(0)

    return max_buc.values, min_buc.values

# Score bands summary
def segment_summary(
    bin_labels: pd.Series,
    y_true: pd.Series
) -> pd.DataFrame:
    
    """
    Summary table for segmentation.

    Description:
        Summary table for assigned segmentation.

    Args:
        bin_labels (pd.Series)  : Output from segmentation.
        y_true (pd.Series)      : The actual target.

    Returns:
        pd.DataFrame: The summary table for statistical tests.

    Notes:
        - N/A.
    """

    df = pd.DataFrame(
        {
            "segment": bin_labels,
            'default': y_true.values
        }
    )
    agg = {
        "n": ("default", "size"),
        "bad": ("default", "sum"),
        "odr": ("default", 'mean')
    }
    summary = (
        df.groupby("segment", observed = True)
        .agg(**agg)
        .reset_index()
        .sort_values("odr", ascending = False)
    )
    summary['good'] = summary['n'] - summary['bad']
    summary['cum_bad'] = summary['bad'].cumsum() / summary['bad'].sum()
    summary['cum_good'] = summary['good'].cumsum() / summary['good'].sum()
    summary['roc'] = (summary['cum_good'] - summary['cum_good'].shift(1, fill_value = 0)) * \
                    (summary['cum_bad'] + summary['cum_bad'].shift(1, fill_value = 0)) * 0.5
    summary['ks'] = abs(summary['cum_good'] - summary['cum_bad'])

    return summary

# Lifetime KS Test
def lifetime_ks(
    actual: dict,
    fitted: dict
) -> dict:
    
    """
    KS-Test for lifetime ODR.

    Description:
        The one sample K-S test tests the null hypothesis that a sample of 
        data comes from a specified distribution. This test is applied to the 
        actual lifetime ODR tested against the fitted ODR from Gamma CDF function. 
        A D-statistic (KS-Stat) smaller than the D-criteria value indicates that
        it fails to reject the null hypothesises, and conclude that the lifetime ODR
        follows Gamma CDF function.

    Args:
        actual (dict)   : Input dictionary. Keys are segmentation name corresponding to the pool.
                          Values are pd.DataFrame contained imputed with weighted Chain-Ladder triangle table (Not run-off).
                          {keys: values} --> {pool (tuple , str): ODR (pd.DataFrame)}
        fitted (dict)   : Input dictionary. Keys are segmentation name corresponding to the pool.
                          Values are pd.DataFrame contained fitted lifetime ODR.
                          {keys: values} --> {pool (tuple , str): ODR (pd.DataFrame)}

    Returns:
        Dictionary: Keys are segmentation name corresponding to the pool.
                    Values are KS Test results.
                    {keys: values} --> {
                                            pool (tuple , str): KS-Test results (dict) --> {
                                                                                                "n": int,
                                                                                                "KS-Stat": float,
                                                                                                "D-Critica": float,
                                                                                                "Result": str)
                                                                                           }
                                       }

    Notes:
        - N/A.
    """
    
    d_critical = {
        1: 0.975,   2: 0.842, 3: 0.708,  4: 0.624,  5: 0.563,
        6: 0.519,   7: 0.483, 8: 0.454,  9: 0.43,   10: 0.409,
        11: 0.391, 12: 0.375, 13: 0.361, 14: 0.349, 15: 0.338,
        16: 0.327, 17: 0.318, 18: 0.309, 19: 0.301, 20: 0.294,
        21: 0.287, 22: 0.281, 23: 0.275, 24: 0.269, 25: 0.264,
        26: 0.259, 27: 0.254, 28: 0.25,  29: 0.246, 30: 0.242,
        31: 0.238, 32: 0.234, 33: 0.231, 34: 0.227, 35: 0.224,
        36: 0.221, 37: 0.218, 38: 0.215, 39: 0.213, 40: 0.21 
    }

    print("=== Processing ===\n[KS Test for Gamma fitted curves]")

    ks_results = {}

    for i, ((pool, c_odr), (_, g_odr)) in enumerate(zip(actual.items(), fitted.items())):
        
        # Constant parameters
        n = len(c_odr) #Test only actual available
        actual_odr = c_odr[:n]
        fitted_odr = g_odr[:n]
        max_odr = max(max(actual_odr), max(fitted_odr))
        min_odr = min(min(actual_odr), min(fitted_odr))
        diff = (max_odr - min_odr) / (n - 2)

        # KS Test
        max_bucket, min_bucket = _bound_bucket(n, start = min_odr, diff_odr = diff)
        actual_diff = (
            (actual_odr[None, :] >= min_bucket[:, None]) & \
            (actual_odr[None, :] < max_bucket[:, None])
        ).sum(axis = 1).cumsum() / n
        fitted_diff = (
            (fitted_odr[None, :] >= min_bucket[:, None]) & \
            (fitted_odr[None, :] < max_bucket[:, None])
        ).sum(axis = 1).cumsum() / n
        ks_stat = np.max(np.abs(actual_diff - fitted_diff))

        # Get D-Critical value for the test
        if n > 40 and n <= 45:
            d_crit = 0.198
        elif n > 45 and n <= 50:
            d_crit = 0.188
        elif n > 50:
            d_crit = 1.358
        else:
            d_crit = d_critical[n]
        
        # KS Test Result
        if ks_stat <= d_crit:
            ks_result = "Pass"
        else:
            ks_result = "Fail"
        
        ks_results[pool] = {
            "n": n,
            "KS-Stat": ks_stat,
            "D-Critical": d_crit,
            "Result": ks_result
        }
        print(f"    [✓] Pool {i}: Segment - {pool}")
    
    return ks_results
