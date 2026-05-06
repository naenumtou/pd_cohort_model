
import warnings
import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta
from scipy.stats import gamma
from scipy.optimize import curve_fit

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper function
def _stable_period(
    df: pd.DataFrame,
    date_col: str,
    end_date: str,
    segment_col: str,
    pool: list,
    performance_window: int = 12
) -> pd.DataFrame:

    """
    Find stable period to build cohort.

    Description:
        Stable period of cohort curve need to consist with the performance window.
        For example, IFRS 9 uses "12-months" as performance window of ODR.
        This means the last snapshort to be an observation point is the last period - 12.
        The last -1 month snapshort can have more stable period as 12 + 1 = 13.
        The last -2 month snapshort can have more stable period as 12 + 2 = 14 and so on unitl reach the oldest date.
        Therefore, the cohort curve counting periods need to complie with this concept.

    Args:
        df (pd.DataFrame)           : Input dataframe.
        date_col (str)              : Datetime column.
        end_date (str)              : The lasest date from development data.
        segment_col (str)           : Segmentation column used to define the minimum date.
        pool (list)                 : Group of segmentations that will be built as one cohort.
        performance_window (int)    : Performance window used to flag ever default. Default = 12.
                                    If there is no particular reason, the 12-months do not need to change.

    Returns:
        pd.DataFrame: DataFrame with possible valid stable period for the segmentation.

    Notes:
        - N/A.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]) #Parse dates --> ensure datetime format

    end_dt = pd.to_datetime(end_date) if end_date else df[date_col].max() #End date
    seg_start = df[df[segment_col].isin(pool)][date_col].min() #Using .isin() to allow a single segment in pool

    diff = relativedelta(end_dt, seg_start)
    diff_months = (diff.months + (diff.years * 12)) + performance_window #Plus performance window
    dates_range = pd.date_range(
        seg_start,
        end_dt,
        freq = "ME"
    )

    stable_period = []
    for p in pool:
        # Loop date from end to start
        records = []
        performance_window = 12
        for date in reversed(dates_range):
            if performance_window > diff_months:
                break
            times_range = pd.DataFrame(
                {
                    "times": range(1, performance_window + 1)
                }
            )   
            times_range.insert(0, "date", date)
            records.append(times_range)
            performance_window += 1
        records = pd.concat(records)
        records.insert(0, "segment", p)
        stable_period.append(records)

    return pd.concat(stable_period, ignore_index = True)

# Development factor
def _dev_factor(
    data: dict
) -> dict:
    
    """
    Development factor for Chain-Ladder analysis.

    Description:
        The run-off triangle table needed to be filled to avoid negativly increased in cumulative ODR,
        when weighted average which incorrect by its definition. The method computes the slope of all
        available information and will impute with run-off triangle tabla later.

    Args:
        data (dictionary): Input dictionary. Keys are segmentation name corresponding to the pool.
                           Values are pd.DataFrame contained run-off triangle table.
                           {keys: values} --> {pool (tuple , str): ODR (pd.DataFrame)}

    Returns:
        Dictionary: Keys are segmentation name corresponding to the pool.
                    Values are list of development factors for corresponding to the pool.
                    {keys: values} --> {pool (tuple , str): development factor (list)}

    Notes:
        - N/A.
    """

    dev_factors = {}
    for pool, c_odr in data.items():
        arr = c_odr.values
        weights = np.asarray(c_odr.index.get_level_values("n"), dtype = float)
        n_cols = arr.shape[1]
        factors = [1] #First column do not need development factor
        
        for i in range(0, n_cols - 1):
            current = arr[:, i]
            forward = arr[:, i + 1]
            # Mask --> only valid forward (Run-off triangle table)
            valid_mask = ~np.isnan(forward)

            if np.all(valid_mask):
                # If all valid --> do not need development factor --> slope = 1
                factors.append(1)
            else:
                # If there is missing in forward window --> compute development factor
                # Get only valid array
                w = weights[valid_mask] #Number of observation for weights
                c = current[valid_mask] #Current values as denominator
                f = forward[valid_mask] #Forward values as numerator
            
                # Development factor
                num = np.dot(w, f)
                den = np.dot(w, c)
                factor = float(num / den) if den != 0 else np.nan
                factors.append(factor)

        dev_factors[pool] = factors

    return dev_factors

# Gamma CDF Function
def _gamma_cdf(
    x: np.ndarray,
    alpha: float,
    beta: float,
    constant: float
) -> np.ndarray:
    
    """
    Gamma cumulative distribution function.

    Description:
        To define the Chain-Ladder cumulative lifetime ODR is following the Gamma distribution.
        The Gamma CDF is using 3 parameters, which are;
            1. alpha is shape of the distribution.
            2. loc is the location of the distribution. In this case, the location is fixed --> loc = 0.
            3. beta (scale) is the size of the distribution.
            4. constant is a control parameter. Given the cumulative lifetime ODR might have a small number.
               Thus, by adding the control parameter will proivde more stable fitting results.

    Args:
        x (np.ndarray)      : The input for fitting (Chain-Ladder cumulative lifetime ODR).
        alpha (float)       : Shape parameter.
        beta (float)        : Scale parameter.
        constant (float)    : Control parameter.

    Returns:
        np.ndarray: Fitted output from Gamma CDF Function.

    Notes:
        - N/A.
    """

    return gamma.cdf(x, alpha, loc = 0, scale = beta) * constant

# Unbias calibration with odds function
def _odds_calibration(
    ttc: np.ndarray,
    odr_12_unbias: float,
    odr_level: str = "Yearly"
) -> np.ndarray:
    
    """
    Odds calibration function for unbias.

    Description:
        ODR Calibration is based on the concept that ratio of odds ratio
        for month m or year y and 12 months or 1-year ODR. By assumption that
        it will remain the same shape for segmentation level and the lifetime pool level.

    Args:
        ttc (np.ndarray)        : The input for calibration (Gamma cumulative lifetime ODR).
        odr_12_unbias (float)   : The 12-months unbias ODR.
        odr_level (str)         : The level of calculated lifetime ODR. Default = "Yearly".

    Returns:
        np.ndarray: Fitted output from odds calibration formula.

    Notes:
        - The ODR Level MUST consist with the inital level of development.
        - If odr_level = "Yearly", this means target for calibartion is the first year ODR.
        - If odr_level = "Monthly", this means target for calibartion is the 12-months ODR.
    """
    
    if odr_level == "Yearly":
        target = ttc[0] #At first position is 1-year ODR
    elif odr_level == "Monthly":
        target = ttc[11] #At 12 position is 12-months (1-year) ODR
    else:
        return print("[WARN]: ODR Level must be 'Yearly' or 'Monthly'")
    
    # Odds calibration
    unbias_curve = (
        odr_12_unbias * (ttc / target)
    ) / (
        (odr_12_unbias * (ttc / target)) + 
        (((1 - ttc) / (1 - target)) * (1 - odr_12_unbias))
    )

    return unbias_curve

# Cohort builder
def cohort_builder(
    df: pd.DataFrame,
    date_col: str,
    end_date: str,
    segment_col: str,
    pool: list,
    cohort_level: str = "Yearly",
    odr_level: str = "Yearly"
) -> dict:
    
    """
    Cohort builder for each segment.

    Description:
        Each segment is computed stable period for save mapping with cohort raw count.
        The granularity of cohort table can be mixed as level;
            1. Cohort level: Yearly, ODR Level: Monthly    --> Default is tracked by month-by-month in observing year
            2. Cohort level: Yearly, ODR Level: Yearly     --> Default is tracked by summation of year in observing year
            3. Cohort level: Monthly, ODR Level: Monthly   --> Default is tracked by month-by-month in observing month
        The marginal actual ODR is computed based on the granularity.
        The cumulative sum of marginal ODR based on the granularity

    Args:
        df (pd.DataFrame)   : Input dataframe.
        date_col (str)      : Datetime column.
        end_date (str)      : The lasest date from development data.
        segment_col (str)   : Segmentation column used to define the minimum date.
        pool (list)         : Group of segmentations that will be built as one cohort.
        cohort_level (str)  : The level to be set as the cohort. Default = "Yearly".
                            If there is suffcient of historical data, yearly basis is appropriate.
                            If there is insuffcient of historical data, monthly basis is more appropriate.
        odr_level (str)     : The level to be calculate ODR as the times tracking. Default = "Yearly".
                            If there is suffcient of historical data, yearly or monthly basis are appropriate.
                            If there is insuffcient of historical data, only monthly basis is appropriate.

    Returns:
        Dictionary: Keys are segmentation name corresponding to the pool.
                    Values are pd.DataFrame contained run-off triangle table.
                    {keys: values} --> {pool (tuple , str): ODR (pd.DataFrame)}

    Notes:
        - If odr_level = "Yearly", this means observed ODR will be already 12-months ODR but stil needed unbias calibration.
        - If odr_level = "Monthly", this means observed ODR will be 1-month ODR NOT 12-months.
    """

    print("=== Processing ===\n[Cohort building]")

    cumulative = {}
    for i, po in enumerate(pool):        
        stable_period = _stable_period(df, date_col, end_date, segment_col, po)
        cohort_table = pd.merge(
            stable_period,
            df,
            how = "left",
            left_on = [segment_col, date_col, "times"],
            right_on = [segment_col, date_col, "times"]
        )

        if cohort_level == "Yearly":
            cohort_table[date_col] = cohort_table[date_col].dt.year
        if odr_level == "Yearly":
            cohort_table["times"] = (cohort_table['times'] - 1) // 12 + 1
        if cohort_level == "Monthly" and odr_level == "Yearly":
            return print("[WARN]: Cohort granular is incorrect") 

        # Cohort with stable period
        n_segment = cohort_table.groupby(date_col)["n"].sum() #Sum total n per level selected 
        cohort_table = pd.pivot_table(
            cohort_table,
            values = "bad",
            index = [date_col],
            columns = ["times"],
            aggfunc = "sum",
            fill_value = np.nan
        )
        cohort_table["n"] = n_segment
        cohort_table = cohort_table.set_index("n", append = True)

        # Marginal ODR
        cohort_table = cohort_table.div(
            cohort_table.index.get_level_values("n"),
            axis = 0
        )

        # Cumulative ODR
        cohort_table = cohort_table.cumsum(axis = 1)

        # Keep results in dictionary
        pool_t = tuple(po) #Unchange allowed
        cumulative[pool_t] = cohort_table

        print(f"    [✓] Pool {i}: Segment - {po}")

    return cumulative

# Chain-Ladder
def chain_ladder(
    data: dict,
) -> dict:
    
    """
    Chain-Ladder analysis.

    Description:
        Chain-Ladder with devlopment factors imputation.
        The run-off triangle table is filled by development factors.
        The Chain-Ladder method is used latest available ODR and multiplied by development factors.

    Args:
        data (dictionary): Input dictionary. Keys are segmentation name corresponding to the pool.
                           Values are pd.DataFrame contained run-off triangle table.
                           {keys: values} --> {pool (tuple , str): ODR (pd.DataFrame)}

    Returns:
        Dictionary: Keys are segmentation name corresponding to the pool.
                    Values are pd.DataFrame contained imputed with Chain-Ladder triangle table (Not run-off).
                    {keys: values} --> {pool (tuple , str): ODR (pd.DataFrame)}

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[Chain-Ladder by development factor]")

    dev_factors = _dev_factor(data)
    
    # Imputation by development factors
    cumulative = {}

    for i, ((pool, c_odr), (_, dev_f)) in enumerate(zip(data.items(), dev_factors.items())):

        arr = c_odr.copy() #Copy for editing

        # Operate as dataframe
        for col in range(arr.shape[1]):
            for row in range(arr.shape[0]):
                if pd.isnull(arr.iloc[row, col]):

                    # If located index is missing --> imputing by lag-1 ODR with latest available development factor
                    arr.iloc[row, col] = arr.iloc[row, col - 1] * dev_f[col]
        
        # Keep imputed results
        cumulative[pool] = arr
        print(f"    [✓] Pool {i}: Segment - {pool}")

    return cumulative

# Weighted average
def segment_weighted_avg(
    data: dict,
) -> dict:
    
    """
    Weighted average.

    Description:
        Weighted average of cohort post imputed by Chain-Ladder.

    Args:
        data (dictionary): Input dictionary. Keys are segmentation name corresponding to the pool.
                           Values are pd.DataFrame contained imputed with Chain-Ladder triangle table (Not run-off).
                           {keys: values} --> {pool (tuple , str): ODR (pd.DataFrame)}

    Returns:
        Dictionary: Keys are segmentation name corresponding to the pool.
                    Values are np.ndarray contained weighted average lifetime ODR of cohort for a corresponding to the pool.
                    {keys: values} --> {pool (tuple , str): ODR (np.ndarray)}

    Notes:
        - N/A.
    """

    weighted_avg = {}

    for pool, c_odr in data.items():
        n = c_odr.index.get_level_values("n")
        avg = np.average(c_odr, axis = 0, weights = n)
        weighted_avg[pool] = avg

    return weighted_avg

# Gamma fitting
def gamma_fitting(
    data: dict,
    n: int
    odr_level: str = "Yearly"
) -> tuple[dict, dict]:
    
    """
    Gamma cumulative distribution fitting.

    Description:
        Curve fitting with Gamma cumulative distribution function.

    Args:
        data (dictionary)    : Input dictionary. Keys are segmentation name corresponding to the pool.
                               Values are np.ndarray contained weighted average lifetime ODR.
                               {keys: values} --> {pool (tuple , str): ODR (np.ndarray)}
        n (int)              : Times (Number of months or years) for extrapolation with Gamma parameters.
        odr_level (str)      : The level to be calculate ODR as the times tracking. Default = "Yearly".
                             If there is suffcient of historical data, yearly or monthly basis are appropriate.
                             If there is insuffcient of historical data, only monthly basis is appropriate.

    Returns:
        Dictionary: Keys are segmentation name corresponding to the pool.
                    Values are np.ndarray contained fitted lifetime ODR for a corresponding to the pool.
                    {keys: values} --> {pool (tuple , str): ODR (np.ndarray)}
        Dictionary: Keys are segmentation name corresponding to the pool.
                    Values are fitted parameters from Gamma cumulative distribution function.
                    {keys: values} --> {pool (tuple , str): parameters (np.ndarray: --> [Alpha, Beta, Constant])}

    Notes:
        - The ODR Level MUST consist with the inital level of development.
    """

    print("=== Processing ===\n[Gamma distribution fitting]")
    
    gamma_odr = {}
    gamma_params = {}

    for i, (pool, c_odr) in enumerate(data.items()):
        n_odr = [j for j in range(1, len(c_odr) + 1)] #Actual range of times
        
        # Curve fitting
        popt, _ = curve_fit(
            f = _gamma_cdf,
            xdata = n_odr,
            ydata = c_odr,
            p0 = [0.1, 0.1, 0.1],
            bounds = ([1e-8, 1e-8, 1e-8], [np.inf, np.inf, np.inf]),
            method = "trf"
        )
        
        # Estimation curve
        if odr_level == "Yearly":
            n_est = [j for j in range(1, n + 1)]
        elif odr_level == "Monthly":
            n_est = [j for j in range(1, n * 12 + 1)]
        
        gamma_fitted = np.clip(
            np.array([_gamma_cdf(x, *popt) for x in n_est]),
            0.0,
            1.0
        ) #Cap min = 0, max = 1
        gamma_odr[pool] = gamma_fitted
        gamma_params[pool] = popt #Parameters: Alpha, Beta, Constant
        
        print(f"    [✓] Pool {i}: Segment - {pool}")

    return gamma_odr, gamma_params

# Unbias ODR Calibration process
def unbias_calibration(
    lifetime_data: dict,
    unbias_data: pd.DataFrame,
    unbias_segment_col: str,
    unbias_odr_col: str,
    n_col: str
) -> dict:

    """
    The unbias ODR Calibration proceses with fitted pool lifetime level.

    Description:
        ODR Calibration is based on the concept that ratio of odds ratio
        for month m or year y and 12 months or 1-year ODR. By assumption that
        it will remain the same shape for segmentation level and the lifetime pool level.

    Args:
        lifetime_data (dictionary)  : The input for calibration (Gamma cumulative lifetime ODR).
        unbias_data (pd.DataFrame)  : The input of 12-months unbias ODR.
        unbias_segment_col (str)    : Segmentation columns.
        unbias_odr_col (str)        : ODR Columns.
        n_col (str)                 : Number of observation for a corresponding to the segment.

    Returns:
        Dictionary: Keys are segmentation name.
                    Values are unbias calibration results.
                    {keys: values} --> {
                                        segment (str): Unbias calibration results (dict) --> {
                                                                                              "n": int,
                                                                                              "Unbias": np.ndarray,
                                                                                             }
                                        }

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[12-months unbias ODR Calibration]")

    calibrated_unbias = {}
    
    unbias = unbias_data[[unbias_segment_col, unbias_odr_col, n_col]]
    unbias["segment_id"] = unbias[unbias_segment_col].str.extract(r"(\d+)").astype(int)

    #Natual sort by --> segment_0, segment_1, segment_2, ... NOT segment_0, segment_1, segment_10, ...
    unbias = unbias.sort_values(by = ["segment_id"])

    for seg_unbias in unbias[unbias_segment_col]:
        odr = unbias[unbias[unbias_segment_col] == seg_unbias][unbias_odr_col].iloc[0] #Extract only value
        n = unbias[unbias[unbias_segment_col] == seg_unbias][n_col].iloc[0] #Keep n for later calculation
        for pool, curve in lifetime_data.items():

            # Calibration on pool level by seperated unbias 12-months ODR
            if seg_unbias in pool:
                calibrated = _odds_calibration(ttc = curve, odr_12_unbias = odr)
                calibrated_unbias[seg_unbias] = {
                    "n": n,
                    "Unbias": calibrated
                }
                print(f"    [✓] Segment - {seg_unbias}")
    
    return calibrated_unbias
