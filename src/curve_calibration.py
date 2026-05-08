
import warnings
import pandas as pd
import numpy as np

from scipy.special import logit, expit
from scipy.optimize import minimize

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper function
# To array from dict
def _to_array(
    data_dict: dict,
    data_key: str
) -> np.ndarray:
   
   """
    To array function.

    Description:
        Convert dictionary to array for calculation.

    Args:
        data (dict)         : Input dictionary. Keys are segmentation name.
                            Values are unbias calibration results.
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
def _weighted_avg(
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
                            Values are unbias calibration results.
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
def _cum_to_mar(
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
def _mar_to_con(
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
def _one_to_twelve(
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
def _twelve_to_one(
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
    
# Calibration process portfolio level
def port_calibrate_pd(
    base_curve: dict,
    weight_key: str,
    pd_key: str,
    fwl_predict: np.ndarray,
    odr_level: str = "Yearly"
) -> np.ndarray:
    
    """
    Liftime carlibration process on portfolio level.

    Description:
        Weighted average of cumulative lifetime PD has been used for calibration.
        The cumulative is converting marginal and conditional basis and transforming
        to logit scale for calibration. Post calibration is then transformed back to
        expit (%) following by conditional and marginal. The final curve will be
        cumulative lifetime PiT PD.
       
        For the monthly level of cohort, there are addtional steps converting from
        1-month to 12-months and convert back to 1-month again.

    Args:
        base_curve (dict)           : Input dictionary. Keys are segmentation name.
                                    Values are unbias calibration results.
                                    {keys: values} --> {
                                                        segment (str): unbias result (dict) --> {
                                                                                                "n": int,
                                                                                                "Unbias": np.ndarray,
                                                                                                }
                                                        }
        weight_key (str)            : Key of weights in dictionary.
        pd_key (str)                : Key of PD in dictionary.
        fwl_predict (np.ndarray)    : n-periods of FWL Prediction.
        odr_level (str)             : The level of calculated lifetime ODR. Default = "Yearly".

    Returns:
        np.ndarray: Weighted average calibrated cumulative lifetime PiT PD.

    Notes:
        - The ODR Level MUST consist with the inital level of development.
        - If odr_level = "Yearly", this means target for calibartion is the first year PD.
        - If odr_level = "Monthly", this means target for calibartion is the 12-months PD.
    """
    
    port_cum = _weighted_avg(base_curve, weight_key, pd_key)
    port_mar = _cum_to_mar(port_cum)
    port_con = _mar_to_con(port_mar)
    target = logit(port_cum[0])
    t = len(port_cum)

    # For monthly level
    if odr_level == "Monthly":
        port_con = _one_to_twelve(port_con)
        target = logit(port_cum[11]) #At month 12

    # FWL Prediction
    mev_effect = _ffill_to_n(fwl_predict, t, odr_level)
    
    # MEV Effect
    port_logit = logit(port_con)
    port_calibrate = port_logit + mev_effect - target
    port_expit = expit(port_calibrate)

    # For monthly level
    if odr_level == "Monthly":
        port_expit = _twelve_to_one(port_expit) #Back to 12-months
    
    port_mar_post = _con_to_mar(port_expit)
    port_cum_post = _mar_to_cum(port_mar_post)
    
    return port_cum_post

# Calibration process segment level
def seg_calibrate_pd(
    base_curve: dict,
    weight_key: str,
    pd_key: str,
    fwl_predict: np.ndarray,
    delta: np.ndarray,
    odr_level: str = "Yearly"
) -> np.ndarray:
    
    """
    Liftime carlibration process on portfolio level.

    Description:
        The cumulative lifetime PD on each segment has been used for calibration.
        The cumulative is converting marginal and conditional basis and transforming
        to logit scale for calibration.
        
        It was observed that if the PiT PD is calculated for weighted average curve,
        it is slightly different from weighted average PiT PD post-calibration from segment.
        The delta need to be incorporated into the calibrartion.
        
        Post calibration is then transformed back to expit (%) following by conditional and marginal.
        The final curve will be cumulative lifetime PiT PD.
       
        For the monthly level of cohort, there are addtional steps converting from
        1-month to 12-months and convert back to 1-month again.

    Args:
        base_curve (dict)           : Input dictionary. Keys are segmentation name.
                                    Values are unbias calibration results.
                                    {keys: values} --> {
                                                        segment (str): unbias result (dict) --> {
                                                                                                "n": int,
                                                                                                "Unbias": np.ndarray,
                                                                                                }
                                                        }
        weight_key (str)            : Key of weights in dictionary.
        pd_key (str)                : Key of PD in dictionary.
        fwl_predict (np.ndarray)    : n-periods of FWL Prediction.
        delta (np.ndarray)          : n-periods of optimized delta for minize the noise.
        odr_level (str)             : The level of calculated lifetime ODR. Default = "Yearly".

    Returns:
        np.ndarray: Calibrated cumulative lifetime PiT PD.

    Notes:
        - The ODR Level MUST consist with the inital level of development.
        - If odr_level = "Yearly", this means target for calibartion is the first year PD.
        - If odr_level = "Monthly", this means target for calibartion is the 12-months PD.
    """
    
    port_cum = _weighted_avg(base_curve, weight_key, pd_key)
    target = logit(port_cum[0]) #Same target as portfolio level
    t = len(port_cum)

    # Segment level
    seg_cum = _to_array(base_curve, pd_key)
    seg_mar = _cum_to_mar(seg_cum)
    seg_con = _mar_to_con(seg_mar)

    # For 1-month basis
    if odr_level == "Monthly":
        seg_con = _one_to_twelve(seg_con)
        target = logit(port_cum[11]) #At month 12, same target as portfolio level
    
    # FWL Prediction
    mev_effect = _ffill_to_n(fwl_predict, t, odr_level)
    
    # MEV Effect (+Delta)
    seg_logit = logit(seg_con)
    seg_calibrate = seg_logit + mev_effect - target + delta
    seg_expit = expit(seg_calibrate)

    # For 1-month basis
    if odr_level == "Monthly":
        seg_expit = _twelve_to_one(seg_expit)

    seg_mar_post = _con_to_mar(seg_expit)
    seg_cum_post = _mar_to_cum(seg_mar_post)

    return seg_cum_post

# Objective function
def objective(
    delta: np.ndarray,
    base_curve: dict,
    weight_key: str,
    pd_key: str,
    fwl_predict: np.ndarray
) -> float:

    """
    Cost objective function.

    Description:
        The sum of absolute difference in post-calibration of portfolio level and segment level
        over the lifetime is minimum.
       
    Args:
        delta (np.ndarray)          : n-periods of optimized delta for minize the noise.
        base_curve (dict)           : Input dictionary. Keys are segmentation name.
                                    Values are unbias calibration results.
                                    {keys: values} --> {
                                                        segment (str): unbias result (dict) --> {
                                                                                                "n": int,
                                                                                                "Unbias": np.ndarray,
                                                                                                }
                                                        }
        weight_key (str)            : Key of weights in dictionary.
        pd_key (str)                : Key of PD in dictionary.
        fwl_predict (np.ndarray)    : n-periods of FWL Prediction.

    Returns:
        float: The sum of absolute difference in post-calibration of portfolio level and segment level.

    Notes:
        - N/A.
    """

    port_cum_post = port_calibrate_pd(base_curve, weight_key, pd_key, fwl_predict)
    seg_cum_post = seg_calibrate_pd(base_curve, weight_key, pd_key, fwl_predict, delta)
    weights = _to_array(base_curve, weight_key)
    w_seg_cum_post = np.sum(seg_cum_post * weights[:, None], axis = 0) / np.sum(weights)

    # Cost function
    diff = np.abs(port_cum_post - w_seg_cum_post)

    return np.sum(diff)

# Optimization delta
def find_delta(
    base_curve: dict,
    weight_key: str,
    pd_key: str,
    fwl_predict: np.ndarray,
    odr_level: str = "Yearly",
    method: str = 'L-BFGS-B'
) -> dict:
   
    """
    Find optimized delta function.

    Description:
        The function for delta optimization to minimize the noise effect. 
       
    Args:
        base_curve (dict)           : Input dictionary. Keys are segmentation name.
                                    Values are unbias calibration results.
                                    {keys: values} --> {
                                                        segment (str): unbias result (dict) --> {
                                                                                                "n": int,
                                                                                                "Unbias": np.ndarray,
                                                                                                }
                                                        }
        weight_key (str)            : Key of weights in dictionary.
        pd_key (str)                : Key of PD in dictionary.
        fwl_predict (np.ndarray)    : n-periods of FWL Prediction.
        odr_level (str)             : The level of calculated lifetime ODR. Default = "Yearly".
        method (str)                : Method for optimization.

    Returns:
        dict: Output dictionary. Keys are segmentation name.
            Values are PiT Calibration results.
            {keys: values} --> {
                                segment (str): PiT result (np.ndarray) --> np.ndarray
                                }

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[Delta optimization for lifetime PD]")

    port_cum = _weighted_avg(base_curve, weight_key, pd_key)
    t = len(port_cum)

    # Initial delta = 0
    delta0 = np.zeros(t)
    result = minimize(
        fun = objective,
        x0 = delta0,
        args = (base_curve, weight_key, pd_key, fwl_predict),
        method  = method,
        options = {'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 100_000}
    )

    # Delta result
    delta_opt = result.x

    # Re-fitting with segmentation curves
    segment_keys = list(base_curve.keys())
    seg_cum_post = seg_calibrate_pd(base_curve, weight_key, pd_key, fwl_predict, delta_opt, odr_level)
    seg_cum_post = dict(zip(segment_keys, seg_cum_post))

    print("=== Result ===")
    print(f"    Loss: {round(result.fun, 4)}")
    print(f"    Delta: {delta_opt}")

    return seg_cum_post
