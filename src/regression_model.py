
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm


from joblib import Parallel, delayed

from src.stats_testing import vif_test, and_dar_test, adf_test
from statsmodels.stats.diagnostic import spec_white
from statsmodels.stats.stattools import durbin_watson

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

# Univariate analysis




"""









"""

# Multivariate analysis

"""
# Multiple linear regression
yTrain = train['CF']

def run_model(i, combination, train, yTrain, ActualODR, CF_SD, CF_MEAN):
    XTrain = sm.add_constant(train[combination], has_constant='add')
    model = sm.OLS(yTrain, XTrain).fit()

    numsVariable = len(model.params) - 1
    modelName = np.repeat(f'MODEL_{i + 1}', numsVariable)
    modelMember = np.arange(1, numsVariable + 1)
    modelVariable = np.array(model.params.index)[1:]
    modelCoefficient = np.array(model.params)[1:]
    modelpValue = np.array(model.pvalues)[1:]
    modelVIF = np.repeat(1, numsVariable) if len(combination) == 1 else np.array(VIF(model.model.exog))[1:]
    modelR2 = np.repeat(model.rsquared, numsVariable)
    modelaR2 = np.repeat(model.rsquared_adj, numsVariable)
    modelNormal = np.repeat(NormalityTest(model.resid), numsVariable)
    modelHetero = np.repeat(spec_white(model.resid, model.model.exog)[1], numsVariable)
    modelAutoCorr = np.repeat(durbin_watson(model.resid), numsVariable)
    modelStationary = np.repeat(ADFTest(model.resid), numsVariable)
    modelBacktest = np.repeat(Backtesting(model, XTrain, ActualODR, CF_SD, CF_MEAN), numsVariable)
    modelOutsampletest = np.repeat(Outsample(XTrain, yTrain, CF_SD, CF_MEAN), numsVariable)

    lags = int(4 * (XTrain.shape[0] / 100) ** (2 / 9))
    HACModel = model.get_robustcov_results(cov_type='HAC', maxlags=lags)
    HACPValue = np.array(HACModel.pvalues)[1:]

    return np.column_stack((
    modelName, modelMember, modelVariable,
    modelCoefficient, modelpValue, HACPValue,
    modelVIF, modelR2, modelaR2,
    modelNormal, modelHetero, modelAutoCorr,
    modelStationary, modelBacktest, modelOutsampletest
    ))


results = Parallel(n_jobs=-1)(
delayed(run_model)(i, combination, train, yTrain, ActualODR, CF_SD, CF_MEAN)
for i, combination in enumerate(allPossibleCombinations)
)

summary = np.vstack(results)

results = Parallel(n_jobs=-1, prefer='threads')



"""
# Run parallel
