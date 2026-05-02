
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

from scipy.special import logit, expit
from scipy.stats import norm
from joblib import Parallel, delayed
from statsmodels.stats.diagnostic import spec_white
from statsmodels.stats.stattools import durbin_watson

from src.plot_function import plot_dep_var
from src.stats_testing import vif_test, and_dar_test, adf_test

# Helper function
def _expand_sign(
    cols: list,
    base_func: callable,
    sign_group: pd.DataFrame,
    mev_col: str,
) -> pd.DataFrame:
    
    """
    Retaining sign after transformation.

    Description:
        To retain the original sign after applying the transformation.
        The MEV(s) have been appiled several transformation logics but
        the expected sign should remain the same. For example, GDP has
        negative relationship with default rate, meaning when the GDP
        increases, the default rate will decrease. This relationship is
        also apply to "GDP_MA3M" (3-month moving average) even it is
        in the different forms. 

    Args:
        cols (list)                 : List of MEV(s) to expand.
        base_func (callable)        : The function for expanding sign.
        sign_group (pd.DataFrame)   : The expected sign input data.
        mev_col (str)               : Name of MEV Column. 

    Returns:
        pd.DataFrame: The output of expected sign with expanding equal to transformed MEV(s).

    Notes:
        - The function will be called 2 times. After first transformed and second transformed.
    """

    cols = pd.Index(cols)
    base = cols.map(base_func)
    return (
        sign_group
        .merge(pd.DataFrame({mev_col: base, "NEW": cols}), on = mev_col)
        .assign(MEV = lambda x: x["NEW"])
        .drop(columns = "NEW")
    )

# MEV(s) Transformation
def mev_transformation(
    raw_data: pd.DataFrame,
    sign_group: pd.DataFrame,
    mev_col: str,
    tpye_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    MEV(s) Transformation

    Description:
        As part of IFRS 9 Requirement, the PD models incorporate the forward-looking information
        to derive the forward-looking provision, which includes the expected impact of
        macroeconomic conditions. THe Macroeconomics Variables MEV(s) are the time series data
        and might not have direct (linear) relationship with dependence variable. It need to 
        consider serval formation to obtain any and addtional relationship as per:
            1. First transformation
                - Year-on-Year Changed --> Rate variables use simple difference, Non-Rate variables use Percent changed.
                - Natural logarithm --> For any np.inf or -np.inf will be forced to zero.
                - Moving average --> Using windows 3, 6, 9 and 12 months for the windows.

            2. Second transformation
                - The monthly MEV(s) lag data points are computed by lagging the current month data behind
                that show a leading indication trend over time series. An indicator is anything that can be used to
                predict future financial economic trends. The leading indicators are indicators that usually, but not always,
                change before the economy as overall changes. Therefore, it is useful as short-term predictors of the economy.

    Args:
        raw_data (pd.DataFrame)     : The raw un-transformed MEV(s) time series data.
        sign_group (pd.DataFrame)   : The original expected sign of MEV(s) with default rate.
        mev_col (str)               : Name of MEV Column.
        tpye_col (str)              : Name of MEV's types Column. 

    Returns:
        pd.DataFrame: The output of transformed MEV(s).
        pd.DataFrame: The output of expected sign with expanding equal to transformed MEV(s).

    Notes:
        - N/A.
    """
    
    print("=== Processing ===\n[MEV(s) Transformation]")

    # Interpolation to monthly basis
    main_cols = raw_data.columns
    data = raw_data.interpolate(
    method = 'linear',
    axis = 0
    )

    # Define rate variables
    s_g = sign_group.copy()
    rate_var = s_g[s_g[tpye_col] == 'Rate'][mev_col].tolist()

    # First transform
    # YoY Changed - Rate variables --> Simple difference 
    diff = data[
        [c for c in main_cols if c in rate_var]
    ].diff(12).add_suffix("_C")

    # YoY Changed - Non-Rate variables --> Percent changed 
    pct = data[
        [c for c in main_cols if c not in rate_var]
    ].pct_change(12).add_suffix("_C")

    # Natural logarithm
    log = np.log(
        data
    ).replace([np.inf, -np.inf], 0).add_suffix("_LN")

    # Moving average - Windows 3, 6, 9 and 12 months
    ma = pd.concat(
        [
            data[main_cols].rolling(w).mean().add_suffix(f"_MA{w}M")
            for w in (3, 6, 9, 12)
        ],
        axis = 1,
    )

    # Concat all first transformations
    data = pd.concat([data, diff, pct, log, ma], axis = 1)

    # Retain expected sign after first transformed
    sign_C = _expand_sign(
        cols = diff.columns.append(pct.columns),
        base_func = lambda c: c.replace("_C", ""),
        sign_group = s_g,
        mev_col = mev_col
    )
    sing_LN = _expand_sign(
        cols = log.columns,
        base_func = lambda c: c.replace("_LN", ""),
        sign_group = s_g,
        mev_col = mev_col
    )
    sign_MA = _expand_sign(
        cols = ma.columns,
        base_func = lambda c: c.split("_MA")[0],
        sign_group = s_g,
        mev_col = mev_col
    )
    # Concat first sign and group information
    first_sign_transformed = pd.concat(
        [sign_group, sign_C, sing_LN, sign_MA],
        ignore_index = True,
    )

    # Second transform (Lag indicators)
    lag_df = pd.concat(
        [
            data.shift(l).add_suffix(f"_LAG{l}M")
            for l in (3, 6, 9, 12)
        ],
        axis = 1,
    )

    data = pd.concat([data, lag_df], axis = 1)

    # Retain expected sign after second transformed
    sign_lag = _expand_sign(
        cols = lag_df.columns,
        base_func = lambda c: c.rsplit("_LAG", 1)[0],
        sign_group = first_sign_transformed,
        mev_col = mev_col
    )

    # Concat sign and group information at the end of transformation
    final_sign_transformed = pd.concat(
        [first_sign_transformed, sign_lag],
        ignore_index = True,
    )
    print(f"=== Result ===\nTotal MEV(s): {data.shape[1]}")

    return data, final_sign_transformed

# Data preparation
def prepare_training_set(
    X: pd.DataFrame,
    y: pd.DataFrame,
    dep_col: str,
    model_method: str,
    outplot: bool = True
) -> tuple[pd.DataFrame, pd.Series, None]:

    """
    Independence variables and dependence variable data preparation.

    Description:
        The monthly ODR(s) are converting to several forms of dependence variables.
        For example, logit function for logit model, CF' (Standardized) for Vasicek model,
        and CCI (Credit Cycle Index). The function is managed any missing values in
        time series by;
            1) Removed all consecutive periods of zero ODR that cannot model.
            2) Filled with mean for any missing values in between timer series.
        The function is also making a equal range of index for MEV(s) Data. Given the fact that
        the dependence variable range is the key consideration for regression model, the MEV(s)
        should be followed this range,

    Args:
        X (pd.DataFrame)    : The transformed MEV(s) Data.
        y (pd.DataFrame)    : The dependence variable target data (ODR or CCI).
        dep_col (str)       : Name of dependence variable column.
        model_method (str)  : Name of the regression method. The function is computed;
                            1) model_method = "Logit" --> logit = ln(ODR / (1 - ODR)).
                            2) model_method = "CF" --> Inverse transformation of normal ODR.
                               CF = np.ppf(ODR), then apply standardization by CF' = (CF - mean) / std
                            3) model_method = "CCI" --> Dependence variable calcualted from CCI Method.
                               Do not need to perform any computation here.
        outplot (bool)      : Option for output plotting.

    Returns:
        pd.DataFrame    : The output of transformed MEV(s) equal range with dependence variable.
        pd.Series       : The output of dependence variable for regression model.
        Figure          : Showing figure from matplotlib.

    Notes:
        - The output parameters need to define either 3 or 2 depending on outplot option.
        - If outplot = Ture --> output parameters will be 3.
        - If outplot = False --> output parameters will be 2.
    """    

    print(f"=== Processing ===\n[Data preparation for {model_method} model]")

    if model_method == "CCI":
        y_data = y[dep_col] #CCI is estimated from another process
        X_data = X.reindex(y_data.index) #Equal range to dependence variable
        if outplot:
            fig = plot_dep_var(y, y_data, model_method)
            return X_data, y_data, fig
        else:
            return X_data, y_data
    
    else:

        # Remove leading zeros dependence variable
        # If any leading zeros will be removed for the series
        mask = y[dep_col].ne(0)
        y = y.loc[mask.cumsum().ne(0)][dep_col].copy()

        # For any missing values in between series --> Fill with mean
        mean_y = y[y != 0].mean()
        y = y.replace(0, mean_y)
        X_data = X.reindex(y.index) #Equal range to dependence variable

        if model_method == "Logit":
            y_data = logit(y) #Logit will preserve expit output range (0-1) as %PD
            if outplot:
                fig = plot_dep_var(y, y_data, model_method)
                return X_data, y_data, fig
            else:
                return X_data, y_data

        elif model_method == "CF":
            cf = pd.Series(norm.ppf(y), index = y.index, name = y.name) #Convert to Cycle Factor (CF)

            # For the Vasicek model, it needs standardization data to perform the regression model
            # It will force the intercept (constant term) to be zero.
            mean_cf = cf.mean()
            std_cf = cf.std()
            mean_X = X_data.mean()
            std_X = X_data.std()

            # Standardized data
            y_data = (cf - mean_cf) / std_cf
            X_data = (X_data - mean_X) / std_X

            # Save standardized parameters     
            stadardized_params = pd.DataFrame(
                {
                    "mean": mean_X,
                    "std": std_X,
                }
            )
            stadardized_params.loc["Dependence_Variable", ["mean", "std"]] = [mean_cf, std_cf] #Add CF parameters
            stadardized_params.index.name = "Variables"
            filename = "standardized_params"
            stadardized_params.to_parquet(
                f"../model/{filename}.parquet",
                engine = 'pyarrow'
                )

            if outplot:
                fig = plot_dep_var(y, y_data, model_method)
                return X_data, y_data, fig
            else:
                return X_data, y_data

        else:
            return print("[WARN]: model_method must be 'Logit', 'CF' or 'CCI'")
