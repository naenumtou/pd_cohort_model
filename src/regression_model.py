
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

from scipy.special import logit, expit
from scipy.stats import norm, t
from varclushi_opt import VarClusHi_Opt
from joblib import Parallel, delayed
from statsmodels.stats.diagnostic import spec_white
from statsmodels.stats.stattools import durbin_watson

from src.plot_function import plot_dep_var, plot_univariate, plot_cluster_timeseries
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

# Categories passed results
def _categorise(
    row: pd.DataFrame
) -> str:
    
    """
    Categories passed results.

    Description:
        Categories passed results for data plotting.

    Args:
        row (pd.DataFrame): The data with univariate analysis result of R-Sqaure, p-value and intuitive sign.

    Returns:
        str: The flagged string for categorise passed variables.

    Notes:
        - N/A.
    """

    if not row["sig_ok"]:
        return "Not significant"
    if not row["r2_ok"]:
        return "R² ≤ 50%"
    if not row["sign_ok"]:
        return "Wrong sign"
    
    return "Pass all 3"

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
                            3) model_method = "CCI" --> Dependence variable calculated from CCI Method.
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

# Univariate analysis
def single_regression(
    X: pd.DataFrame,
    y: pd.Series,
    sign: pd.DataFrame,
    mev_col: str,
    sign_col: str,
    p_threshold: float = 0.05,
    r2_threshold: float = 0.5,
    outplot: bool = True
) -> tuple[list, None]:
    
    """
    Univariate analysis by single linear regression.

    Description:
        Upon completion of transforming MEVs, preliminary assessments are performed
        to further shortlist the MEVs prior to the multivariate analysis.Any MEVs will be
        selected if it passes the following criteria;
            1) p-value significant.
            2) Predictive power of R-Square more than 50%.
            3) An intuitive relationship between the MEVs and dependence variable where the expected sign is predefined.
               Either sign (0) with p-value significant and more than 50% R-Sqaure are allowed.

    Args:
        X (pd.DataFrame)        : The transformed MEV(s) Data.
        y (pd.Series)           : The dependence variable target data (Logit, CF or CCI).
        sign (pd.DataFrame)     : The data of MEV(s) sign and group contained.
        mev_col (str)           : Name of MEV(s) column.
        sign_col (str)          : Name of MEV(s) sign column.
        p_threshold (float)     : p-value threshold. Default is 0.05 (5%).
        r2_threshold (float)    : R-Square threshold. Default is 0.5 (50%).
        outplot (bool)          : Option for output plotting.

    Returns:
        List    : The selected MEV(s).
        Figure  : Showing figure from matplotlib.

    Notes:
        - The output parameters need to define either 2 or 1 depending on outplot option.
        - If outplot = Ture --> output parameters will be 2.
        - If outplot = False --> output parameters will be 1.
    """    

    print(f"=== Processing ===\n[Univariate analysis]")

    # For vector calculation
    X_arr = X.values
    y_arr = y.values
    n = len(y_arr)

    # Vectorized correlation
    corr = np.corrcoef(X_arr.T, y_arr)[-1, :-1]
    r2 = corr ** 2

    # p-value
    t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2 + 1e-12))
    p_values = 2 * t.sf(np.abs(t_stat), df = n - 2)

    # Coefficient = corr * (std_y / std_x)
    std_x = X_arr.std(axis = 0)
    std_y = y_arr.std()
    coff = corr * (std_y / (std_x + 1e-12))

    # Sign of MEV(s)
    sign_map = sign.set_index(mev_col)[sign_col].to_dict()
    sign_mev = sign[sign_col].values
    
    # Selection
    passed = (p_values < p_threshold) & (r2 >= r2_threshold) & ((sign_mev == 0) | (coff / sign_mev > 0))

    # To DataFrame
    results = pd.DataFrame(
        {
            mev_col: X.columns,
            "coefficient": coff,
            "p_value": p_values,
            "r2": r2
        }
    )
    
    results[sign_col] = results[mev_col].map(sign_map)
    results["pass"] = passed
    passed_vars = results.loc[passed, mev_col].tolist()

    print(f"=== Result ===\nNumber of passed variables: {len(passed_vars)}")

    if outplot is False:
        return results
    else:

        # Data for the plot
        df = results.copy()
        df["neg_log_p"] = -np.log10(df["p_value"].clip(lower = 1e-300))
        df["sign_ok"] = (np.sign(df["coefficient"]) == df[sign_col]) | (df[sign_col] == 0)
        df["sig_ok"] = df["p_value"] < p_threshold
        df["r2_ok"] = df["r2"] >= r2_threshold
        df["category"] = df.apply(_categorise, axis = 1)
        fig = plot_univariate(df, p_threshold, r2_threshold)
        
        return results, fig

# Multivariate analysis
def multivariate_selection(
    X: pd.DataFrame,
    group: pd.DataFrame,
    mev_col: str,
    group_col: str,
    univariate_result: pd.DataFrame,
    n_select: int = 1,
    outplot: bool = True
) -> tuple[list, None]:
    
    """
    Multivariate analysis by cluster analysis.

    Description:
        One of the commonly used methods in the industry to assess the multicollinearity is
        Variable Clustering, which attempts to divide a set of variables into non-overlapping clusters.
        It finds clusters or groups of the variables that are as correlated as possible with
        the variables within the cluster and as uncorrelated as possible with the variables from other. 
        If it is observed that the eigenvalue of the cluster is greater than a certain threshold,
        then the cluster will be further split into separate cluster.

        The "VarClusHi_Opt" library is custom build. (pip install varclushi_opt).
        Visit link: https://github.com/naenumtou/varclushi_opt
        
        Analysis can then be done on the output of VarClusHi_Opt in order to select the variables from each cluster.
        The clusters from the output will be used as a starting point in order to group the variables into groups
        where the variables within the group are highly correlated with each other but have low correlation for
        other groups. The R-Squared with Own Cluster and Next Closest columns can be used as a general guide.
        Below are the criteria for the selection per cluster;
            1) n variable(s) with the lowest R-square ratio.
            2) n variable(s) with the highest R-square.

    Args:
        X (pd.DataFrame)                    : The transformed MEV(s) Data.
        group (pd.DataFrame)                : The data of MEV(s) sign and group contained.
        mev_col (str)                       : Name of MEV(s) column.
        group_col (str)                     : Name of MEV(s) group column.
        univariate_result (pd.DataFrame)    : The table result from univariate analysis.
        n_select (int)                      : Number of selected variables per cluster.
        outplot (bool)                      : Option for output plotting.

    Returns:
        List    : The selected MEV(s) with cluster number and group of MEV(s).
        Figure  : Showing figure from matplotlib.

    Notes:
        - The output parameters need to define either 2 or 1 depending on outplot option.
        - If outplot = Ture --> output parameters will be 2.
        - If outplot = False --> output parameters will be 1.
    """

    print(f"=== Processing ===\n[Multivariate analysis]")

    # Passed variables from univariate to list
    passed_vars = univariate_result[univariate_result["pass"] == True][mev_col].tolist()

    # Cluster analysis
    data = X[passed_vars]
    cluster_opt = VarClusHi_Opt(data, maxeigval2 = 1, maxclus = None)
    cluster_opt.varclus()
    clsuter_df = cluster_opt.rsquare
    clsuter_df["Cluster"] += 1 #Make cluster index start from 1

    # Mapping R-Sqaure for selection
    r2_map = univariate_result.set_index(mev_col)["r2"].to_dict()
    clsuter_df["r2"] = clsuter_df["Variable"].map(r2_map)

    # Mapping group for combination
    group_map = group.set_index(mev_col)[group_col].to_dict()
    clsuter_df[group_col] = clsuter_df["Variable"].map(group_map)

    # Sorting
    g = clsuter_df.groupby("Cluster", sort = False)

    # R-Sqaure ratio (Low --> high)
    clsuter_df["r2_ratio_rank"] = g["RS_Ratio"].rank(
        method = "first",
        ascending = True   
    )

    # R-Sqaure (High --> low)
    clsuter_df["r2_rank"] = g["r2"].rank(
        method = "first",
        ascending = False
    )

    # Selection
    clsuter_df["pass"] = (
        (clsuter_df["r2_ratio_rank"] <= n_select) |
        (clsuter_df["r2_rank"] <= n_select)
    )

    passed_vars = clsuter_df[clsuter_df["pass"] == True][["Variable", group_col, "Cluster"]].values.tolist()

    print(f"=== Result ===\nNumber of passed variables: {len(passed_vars)}")

    if outplot is False:
        return passed_vars
    
    else:
        fig = plot_cluster_timeseries(X, clsuter_df)
        return passed_vars, fig

# All possible combinations
def get_combinations(
    data: list,
    N: int
) -> list:
    
    """
    Variables combinations.

    Description:
        For the purpose of model development all possible combinations of variables in each
        cluster will need to be assessed for model development. This will ensure an exhaustive
        list of all possible models would be considered in order to generate the best possible model.

        One combination must not contain variables from same cluster and must be difference
        from variables group. One combination is capped maximum number of variables up to 3 variables.
        This is to avoid multicollinearity issue in the multiple linear regression model.

    Args:
        data (list) : List of result from cluster analysis, containing variable, group, cluster number.
        N (int)     : Number of MEV(s) per combination needed.

    Returns:
        List: The possible combinations up to N variables with conditional.

    Notes:
        - N/A.
    """
    
    print(f"=== Processing ===\n[Possible combinations of {N} variable(s)]")
    
    # Map cluster and subgroup to compact integers
    cluster_id = {}
    subgroup_id = {}
    cid = sid = 0

    encoded = []
    for name, cluster, subgroup in data:
        if cluster not in cluster_id:
            cluster_id[cluster] = cid; cid += 1
        if subgroup not in subgroup_id:
            subgroup_id[subgroup] = sid; sid += 1

        encoded.append((
            name,
            1 << cluster_id[cluster],   #Cluster bit
            1 << subgroup_id[subgroup]  #Subgroup bit
        ))

    results = []
    n_data = len(encoded)

    def dfs(idx, k, comb, used_clusters, used_subgroups):
        # Done
        if k == 0:
            results.append(comb.copy())
            return

        # Prune: not enough elements left
        if idx == n_data or n_data - idx < k:
            return

        name, cbit, sbit = encoded[idx]

        # Skip
        dfs(idx + 1, k, comb, used_clusters, used_subgroups)

        # Take (only if cluster & subgroup unused)
        if not (used_clusters & cbit or used_subgroups & sbit):
            comb.append(name)
            dfs(idx + 1, k - 1, comb,
                used_clusters | cbit,
                used_subgroups | sbit)
            comb.pop()

    dfs(0, N, [], 0, 0)

    print(f"    Number of combinations: {len(results)}")

    return results

