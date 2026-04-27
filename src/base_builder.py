
import warnings
import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta

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
            2. Cohort level: Yearly, ODR Level: Yearly     --> Default is tracked by summation of year in in observing year
            3. Cohort level: Monthly, ODR Level: Monthly   --> Default is tracked by month-by-month in in observing month
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
        - If odr_level = "Yearly", this means observed ODR will be already 12-months ODR but stil need unbias calibration.
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
                    Values are np.array contained weighted average of cohort for a corresponding to the pool.
                    {keys: values} --> {pool (tuple , str): ODR (np.array)}

    Notes:
        - N/A.
    """

    weighted_avg = {}

    for pool, c_odr in data.items():
        n = c_odr.index.get_level_values("n")
        avg = np.average(c_odr, axis = 0, weights = n)
        weighted_avg[pool] = avg

    return weighted_avg
