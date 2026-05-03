
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import anderson, norm
from statsmodels.tsa.stattools import adfuller
from scipy.special import expit

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
                          {keys: values} --> {pool (tuple , str): ODR (np.array)}
        fitted (dict)   : Input dictionary. Keys are segmentation name corresponding to the pool.
                          Values are pd.DataFrame contained fitted lifetime ODR.
                          {keys: values} --> {pool (tuple , str): ODR (np.array)}

    Returns:
        Dictionary: Keys are segmentation name corresponding to the pool.
                    Values are KS Test results.
                    {keys: values} --> {
                                            pool (tuple , str): KS-Test results (dict) --> {
                                                                                                "n": int,
                                                                                                "KS-Stat": float,
                                                                                                "D-Critica": float,
                                                                                                "Result": str
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

# VIF Test for multicollinearity of independence variables
def vif_test(
    x: np.array
) -> list:

    """
    Variance Inflation Factor (VIF) for multicollinearity.

    Description:
        In modelling development, multicollinearity is a phenomenon in which two or more predictor variables
        in a multiple regression model are highly correlated, meaning that one can be linearly predicted
        from the others with a substantial degree of accuracy.
        
        The Variance Inflation Factor (VIF) computation is used to depict the existence of multicollinearity
        (correlation between independence variables) in a regression analysis. The VIF quantifies how much
        the variance of the estimated regression coefficients are inflated as compared to when the predictor
        variables are not linearly related.

    Args:
        x (np.arrat): Input data of independence variables MEV(s) in the model.

    Returns:
        List: List of VIF on each independence variables MEV(s) in the model.

    Notes:
        - If the VIF is less than (<) 5, the model is passed multicollinearity assumption.
        - If the VIF is greater than or equal to (>=) 5, the model is not passed multicollinearity assumption.
    """

    vif = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
    vif[0] = 0 #Intercept do not need to calculate VIF

    return vif

# Anderson-Darling Test for residual normality
def and_dar_test(
    residual: pd.Series
) -> float:
    
    """
    Anderson-Darling Test for residual normality.

    Description:
        The Anderson-Darling test tests the null hypothesis that a sample is 
        drawn from a population that follows a particular distribution, which is
        a normal distribution in this case. The output from Scipy library returns
        the Anderson-Darling test statistic not p-value when method is None.
        
        To derive the p-value follows the SAS Calculation logit, the paper of
        R.B. D'Augostino and M.A. Stephens, Eds., 1986, Goodness-of-Fit Techniques, Marcel Dekker is leveraged.

    Args:
        residual (pd.Series): Model residual from statsmodels.

    Returns:
        Float: p-value follows SAS Calculation logic.

    Notes:
        - If the p-value is greater than (>) 0.05, the residual is followed normal distribution.
        - If the p-value is less than or equal to (<=) 0.05, the model is not passed normality assumption.
    """
    
    ad, _, _ = anderson(residual, dist = "norm",  method = None)
    ad_adj = ad * (1 + (0.75 / residual.shape[0]) + 2.25 / (residual.shape[0] ** 2))

    if ad_adj >= 0.6:
        p_value = np.exp(1.2937 - 5.709 * ad_adj + 0.0186 * np.power(ad_adj, 2))
    elif ad_adj >= 0.34:
        p_value = np.exp(0.9177 - 4.279 * ad_adj - 1.38 * np.power(ad_adj, 2))
    elif ad_adj > 0.2:
        p_value = 1 - np.exp(-8.318 + 42.796 * ad_adj - 59.938 * np.power(ad_adj, 2))
    else:
        p_value = 1 - np.exp(-13.436 + 101.14 * ad_adj - 223.73 * np.power(ad_adj, 2))

    return p_value

# ADF Test for residual stationary (Co-integrated)
def adf_test(
    residual: pd.Series
) -> float:
    
    """
    Augmented Dickey-Fuller Test for residual stationary.

    Description:
        The time series model is generally required staionary variables in the model.
        However, it might limit the comprehensive combination if selecting only stationary variables.

        The Engle-Granger Cointegration methodology is leveraged two-stages regression as following.
        First, assuming all independence variables are unit-root (non-stationary).
        Second, using the linear combination of independence variables run regression.
        Third, ADF testing on model residuals instead of each independence variables.
        If the residuals are stationary, the independence variables are cointegrated.

    Args:
        residual (pd.Series): Model residual from statsmodels.

    Returns:
        Float: p-value.

    Notes:
        - If the p-value is less than or equal to (<=) 0.1, the residual is stationary.
        - If the p-value is greater than (>) 0.1, the model is not passed normality assumption.
        - The limitation of Engle-Granger Cointegration methodology:
            - The cointegration result is appropriate for two variables. For multiple variables, the Johansen cointegration test is better.
            - The method assumes a linear cointegrating relationship, which may not hold in all cases.
            - The results can be sensitive to the choice of the dependent variable in the cointegrating regression.
    """

    _, p_value, _, _, _, _ = adfuller(
        residual,
        maxlag = None,
        regression = "n", #No constant and No trend.
        autolag = "AIC"
    )

    return p_value

# Back-testing
def back_testing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: callable,
    model_method: str,
    mean_cf: float = None,
    std_cf: float = None
) -> float:
    
    """
    In-sample back-testing.

    Description:
        The In-sample back-testing or exceed rate evaluates model performance on observations
        that are used to develop the model by comparing values predicted by the model to
        historical values. By calculating prediction intervals at 95% confidence level,
        analysing discrepancies between the actual and prediction. Higher incidence of
        discrepancies does not necessarily mean the model should be rejected.

    Args:
        X_train (pd.DataFrame)  : The transformed MEV(s) Data.
        y_train (pd.Series)     : The dependence variable target data (Logit, CF or CCI).
        model (callable)        : The tranined regression model.
        model_method (str)      : Name of the regression method. The function is computed;
                                1) model_method = "Logit" --> %ODR vs %predicted ODR.
                                2) model_method = "CF" --> Inverse CF and compute %ODR vs %predicted ODR.
                                3) model_method = "CCI" --> CCI vs predicted CCI.
        mean_cf (float, None)   : Mean of CF to inverse calculation for CF Model.
                                If None, for CCI and Logit models.
        std_cf (float, None)    : Standard deviation of CF to inverse calculation for CF Model.
                                If None, for CCI and Logit models.

    Returns:
        Float: Exceed rate.

    Notes:
        - N/A.
    """

    if model_method == "CCI":
        y_pred = model.predict(X_train)
        y_true = y_train.copy()

    if model_method == "Logit":
        y_pred = expit(model.predict(X_train))
        y_true = expit(y_train)

    if model_method == "CF":
        y_pred = pd.Series(norm.cdf((model.predict(X_train)) * std_cf + mean_cf))
        y_true = pd.Series(norm.cdf(y_train * std_cf + mean_cf))

    sd = y_pred.std()
    upper = y_pred + 2 * sd
    lower = y_pred - 2 * sd
    exceed_rate = (y_true.lt(lower).sum() + y_true.gt(upper).sum()) / len(y_true)
    
    return exceed_rate

# Out-sample testing
def out_sample_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_method: str,
    mean_cf: float = None,
    std_cf: float = None
) -> float:

    """
    Out-sample back-testing.

    Description:
        For the out-of-sample testing or breach rate, a rolling windows approach was leveraged.
        The rolling window segregates the data into training sample and testing sample, with the observations
        are in the rolling window used as the training sample. The model with the selected variables is then
        re-fitting on the training sample and its performance is evaluated on the testing sample by calculating
        prediction intervals at 95% and analysing discrepancies.

    Args:
        X_train (pd.DataFrame)  : The transformed MEV(s) Data.
        y_train (pd.Series)     : The dependence variable target data (Logit, CF or CCI).
        model (callable)        : The tranined regression model.
        model_method (str)      : Name of the regression method. The function is computed;
                                1) model_method = "Logit" --> %ODR vs %predicted ODR.
                                2) model_method = "CF" --> Inverse CF and compute %ODR vs %predicted ODR.
                                3) model_method = "CCI" --> CCI vs predicted CCI.
        mean_cf (float, None)   : Mean of CF to inverse calculation for CF Model.
                                If None, for CCI and Logit models.
        std_cf (float, None)    : Standard deviation of CF to inverse calculation for CF Model.
                                If None, for CCI and Logit models.

    Returns:
        Float: Exceed rate.

    Notes:
        - N/A.
    """

    size = len(X_train)
    sample_size = int(size / 2)
    train_size = int(sample_size / 2)
    test_size = int(sample_size / 2)
    steps = train_size + test_size

    breach = []
    for i in range(train_size, size, steps):
        X_train_sample, X_test_sample = X_train.iloc[i - train_size:i], X_train.iloc[i:i + test_size]
        y_train_sample, y_test_sample = y_train.iloc[i - train_size:i], y_train.iloc[i:i + test_size]

        #Re-fitting model with the same combination but different periods
        model = sm.OLS(y_train_sample, X_train_sample).fit() #Do not need to constant --> already had in data

        if model_method == "CCI":
            y_pred = model.predict(X_test_sample)
            y_true = y_test_sample.copy()

        if model_method == "Logit":
            y_pred = expit(model.predict(X_test_sample))
            y_true = expit(y_test_sample)

        if model_method == "CF":
            y_pred = pd.Series(norm.cdf((model.predict(X_test_sample)) * std_cf + mean_cf))
            y_true = pd.Series(norm.cdf(y_test_sample * std_cf + mean_cf))

        sd = y_pred.std()
        upper = y_pred + 2 * sd
        lower = y_pred - 2 * sd
        breach.append(y_true.lt(lower).sum() + y_true.gt(upper).sum())
    breach_total = sum(breach)

    return breach_total / (test_size * sample_size)
