
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

# Weighted average
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
        mar (np.ndarray): Input array as marginal PD.

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

# To 12-months basis
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
        con (np.ndarray): Input array as conditional PD.

    Returns:
        np.ndarray: 12-months conditional PD.

    Notes:
        - N/A.
    """

    return 1 - ((1 - con) ** 12)

# To 1-month basis
def _to_one_basis(
    con: np.ndarray
) -> np.ndarray:
       
    """
    Convert 12-months conditional PD to 1-month conditional PD.

    Description:
        Converting 12-months conditional PD to 1-month conditional PD. In case,
        the cohort model was built in monthly basis.

    Args:
        con (np.ndarray): Input array as conditional PD.

    Returns:
        np.ndarray: 1-month conditional PD.

    Notes:
        - N/A.
    """

    return 1 - ((1 - (1 - (1 - con) ** (1 / 12))))

# Conditional to marginal
def _con_to_mar(
    con: np.ndarray
) -> np.ndarray:
    
    """
    Convert conditional PiT PD to marginal PiT PD.

    Description:
        Post macro effects adjustment, the conditional PiT PD is converted to marginal PiT PD.

    Args:
        con (np.ndarray): PiT Conditional PD.

    Returns:
        np.ndarray: PiT Marginal PD.

    Notes:
        - The first year of marginal PiT PD is equal to conditional PiT PD.
    """

    cp = np.cumprod(1 - con, axis = -1)
    shift = np.ones_like(con)
    shift[..., 1:] = cp[..., :-1]

    return con * shift

# Marginal to cumulative
def _mar_to_cum(
    mar: np.ndarray
) -> np.ndarray:

    """
    Convert marginal PiT PD to cumulative PiT PD.

    Description:
        Post macro effects adjustment, the marginal PiT PD is converted to cumulative PiT PD.

    Args:
        mar (np.ndarray): PiT Marginal PD.

    Returns:
        np.ndarray: PiT Cumulative PD.

    Notes:
        - The first year of cumulative PiT PD is equal to marginal PiT PD and conditional PiT PD.
    """

    if mar.ndim == 1:
        return np.cumsum(mar)
    elif mar.ndim == 2:
        return np.cumsum(mar, axis = 1)
    else:
        print("[WARN]: Only 1D or 2D arrays supported")
    
# Making array for forward-looking prediction
def _ffill_to_n(
    fwl: np.ndarray,
    n: int,
    odr_level: str = "Yearly"
) -> np.ndarray:
    
    """
    Create forcasting PD array.

    Description:
        Generally, the forecasting from FWL Model of macroeconomics have limit range of data.
        The function is created reasonable forecasting array by taking the last position as
        the latest forecasting information available to calibrate with cohort for lifetime PD.
       
        For the monthly level of cohort, since the FWL Model modeled on 12-months ODR, the
        first 12 positions will assume to be equal to the first forecasting of 12-month PD.

    Args:
        fwl (np.ndarray)    : n-periods of FWL Prediction.
        n (int)             : n-periods of cohort curve.
        odr_level (str)     : The level of calculated lifetime ODR. Default = "Yearly".

    Returns:
        np.ndarray: Logit function to FWL Prediction.

    Notes:
        - The ODR Level MUST consist with the inital level of development.
        - If odr_level = "Yearly", this means target for calibartion is the first year PD.
        - If odr_level = "Monthly", this means target for calibartion is the 12-months PD.
    """

    fwl = np.asarray(fwl, dtype = float)
    length = len(fwl)

    # For very long forecasting window
    if length >= n:
        return fwl[:n]

    # Yearly level cohort
    if odr_level == "Yearly":
        # Forward fill with latest information
        fill = np.repeat(fwl[-1], n - length)
        return logit(np.concatenate([fwl, fill]))
    
    # Monthly level cohort
    else:
        # First 12-months are the first fwl prediction
        out = np.empty(n, dtype = float) #Create empty array
        first_len = min(12, n) 
        out[:first_len] = fwl[0] #Fill first 12-months with first prediction

        if n <= 12:
            return logit(out) #Return if less than 12 months

        # For more than 12 months
        tail = fwl[1:]
        remaining = n - 12
        if len(tail) >= remaining:
            out[12:] = tail[:remaining]
        else:
            out[12:12 + len(tail)] = tail
            out[12 + len(tail):] = tail[-1]

        return logit(out)
