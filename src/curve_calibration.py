
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
) -> np.ndarray:
   
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
                                                                                        "Unbias": np.ndarray,
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
                                                                                        "Unbias": np.ndarray,
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
        from cumulative PD, the difference between n + 1 to n is computed.

    Args:
        cum (np.ndarray): Input array as cumulative PD.

    Returns:
        np.ndarray: Marginal PD.

    Notes:
        - The first year of marginal PD is equal to cumulative PD.
    """

    return np.diff(cum, prepend = 0)

# Marginal to conditional
def mar_to_con(
    mar: np.ndarray
) -> np.ndarray:
    
    """
    Conditional PD.

    Description:
        Given that the model was developed on 12-months ODR, the proposed logit approach
        focuses on overlaying macro effect on corresponding 12-months PD component.
        Macro effect has to be incorporated on Conditional PD to align with this basis.

    Args:
        mar (np.array): Input array as marginal PD.

    Returns:
        np.ndarray: Conditional PD.

    Notes:
        - The first year of conditional PD is equal to cumulative PD and marginal PD.
    """

    # Function to support 1D array or 2D array
    if mar.ndim == 1:
        cum_shift = np.concatenate(([0], np.cumsum(mar)[:-1]))
    elif mar.ndim == 2:
        cum = np.cumsum(mar, axis = 1)
        cum_shift = np.concatenate(
            [np.zeros((mar.shape[0], 1)), cum[:, :-1]],
            axis = 1
        )
    else:
        print("[WARN]: Only 1D or 2D arrays supported")
        
    return mar / (1 - cum_shift)

# To 12-months Basis
def to_twelve_basis(
    con: np.ndarray
) -> np.ndarray:
    
    """
    Convert 1-month conditional PD to 12-months conditional PD.

    Description:
        Given the fact that some models was developed based on 1-month ODR, the propose
        is to convert from 1-month basis to the same as the macro effect on corresponding
        12-months before incorporating into the PD Curves.

    Args:
        mar (np.array): Input array as marginal PD.

    Returns:
        np.ndarray: 12-months conditional PD.

    Notes:
        - N/A.
    """

    return 1 - ((1 - con) ** 12)
