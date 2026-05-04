
import warnings
import pandas as pd
import numpy as np

from scipy.optimize import curve_fit

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper function
def to_array(
    data_dict: dict,
    data_key: str
) -> np.array:
   
   """
    To array function.

    Description:
        Convert dictionary to array for calculation.

    Args:
        data (dict)         : Input dictionary. Keys are segmentation name.
                            Values are unbias calibration results
                            {keys: values} --> {
                                                segment (str): unbias result (dict) --> {
                                                                                        "n": int,
                                                                                        "Unbias": np.array,
                                                                                        }
                                                }
        data_key (str)      : Key of weights in dictionary.

    Returns:
        np.ndarray: Array contained information for calculation.

    Notes:
        - N/A.
    """
   
   return np.array([v[data_key] for v in data_dict.values()])

def weighted_avg(
    data_dict: dict,
    weight_key: str,
    pd_key: str
) -> np.ndarray:
      
    """
    Weighted average of segmentation.

    Description:
        Compute weighted average of segmentation to portfolio level.

    Args:
        data (dict)         : Input dictionary. Keys are segmentation name.
                            Values are unbias calibration results
                            {keys: values} --> {
                                                segment (str): unbias result (dict) --> {
                                                                                        "n": int,
                                                                                        "Unbias": np.array,
                                                                                        }
                                                }
        weight_key (str)    : Key of weights in dictionary.
        pd_key (str)        : Key of PD in dictionary.

    Returns:
        np.ndarray: Weighted average of portfolio level.

    Notes:
        - N/A.
    """

    data = to_array(data_dict, pd_key)
    weights = to_array(data_dict, weight_key)

    return np.sum(data * weights[:, None], axis = 0) / np.sum(weights)

# Cumulative to marginal
def cum_to_mar(
    cum: np.ndarray
) -> np.ndarray:

    """
    Marginal PD.

    Description:
        The through the cycle curve (TTC) has been built based on observation time
        and cumulative throughout lifetime of portfolio. To derive marginal PD
        from cumulative PD, the difference between n + 1 to n is computed. T

    Args:
        cum (np.array): Input array as cumulative PD.

    Returns:
        np.ndarray: Marginal PD.

    Notes:
        - The first year of marginal PD is equal to cumulative PD.
    """

    return np.diff(cum, prepend = 0)

def mar_to_con(mar: np.ndarray) -> np.ndarray:
   """Marginal PD → Conditional PD
   S(t) = cumprod(1 - mar)
   con(t) = 1 - S(t)/S(t-1)
   """
   s = np.cumprod(1 - mar)
   s_lag = np.concatenate([[1.0], s[:-1]])
   return 1 - s / s_lag

def con_to_mar(con: np.ndarray) -> np.ndarray:
   """Conditional PD → Marginal PD
   S(t) = cumprod(1 - con)
   mar(t) = 1 - S(t)/S(t-1)
   """
   s = np.cumprod(1 - con)
   s_lag = np.concatenate([[1.0], s[:-1]])
   return 1 - s / s_lag
def mar_to_cum(mar: np.ndarray) -> np.ndarray:
   """Marginal PD → Cumulative PD
   S(t) = cumprod(1 - mar)
   CumPD(t) = 1 - S(t)
   """
   return 1 - np.cumprod(1 - mar)