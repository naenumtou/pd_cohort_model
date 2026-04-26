
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Helper functions
def _lag_cols(
    base: str,
    n: int
) -> list[str]:

    """
    Lagging columns functions.

    Description:
        Lagging columns only used for the calculation.

    Args:
        base (str)  : Columns name for lagging.
        n (int)     : Window to lag the columns.

    Returns:
        List: List of lag column names.

    Notes:
        - N/A.
    """

    return [f"{base}{i}" for i in range(1, n + 1)]

# Sort and flag
def prepare_dataframe(
    df: pd.DataFrame,
    id_col: str,
    period_col: str,
    default_col: str,
    default_flag: int
) -> pd.DataFrame:
    
    """
    Sort by primary key and period and flag the target for modeling.

    Description:
        Raw transaction must be sorted from historical to current.
        To create the flag of target in the model.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        id_col (str)        : Primary key.
        period_col (str)    : Period key for sorting.
        default_col (str)   : Default column as the target for modeling.
        default_flag (int)  : Default value that greater than the value will be considered as default.

    Returns:
        pd.DataFrame: DataFrame with sorted and flagged the target.

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[Sort and create default flag]")

    df = df.sort_values(by = [id_col, period_col]).copy()

    df[f"def"] = np.where(
        df[default_col].ge(default_flag),
        1, 0
    )

    return df

# Forward performance windows until lifetime
def ever_default_lifetime(
    df: pd.DataFrame,
    id_col: str,
    default_col: str,
    n_lags: int,
    lifetime_lags: int
) -> pd.DataFrame:
    
    """
    Create forward-1 until forward-n columns for target.
    Uses a single groupby().transform(lambda).

    Description:
        The n-lags of column features are created by primary key. 
        bal{w}          Account balance lagged w months.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        id_col (str)        : Primary key.
        default_col (str)   : Default column that already flag (0, 1) for modeling.
        n_lags (int)        : Defined n-lags for ever default creation.
        lifetime_lags (int) : Defined n-lags for forward performance lifetime windows creation.

    Returns:
        pd.DataFrame: DataFrame with forward performance columns and ever default flag appended.

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[Forward performance windows and ever default]")

    # Forward performance windows until lifetime
    grouped = df.groupby(id_col)[default_col]
    shifted = {
        f"{default_col}{i}": grouped.shift(-i).astype(np.float16)
        for i in range(1, lifetime_lags) #(Exclusive) DO NOT NEED +1 because need at least 1 month to observe
    }
    df = df.assign(**shifted)

    # Ever default flag
    cols = _lag_cols(default_col, n_lags)
    window = df[cols]
    df[f"ever_default_{n_lags}"] = window.eq(1).any(axis = 1).astype(np.uint8)

    return df

# Lifetime flag
def lifetime_flag(
    df: pd.DataFrame,
    default_col: str,
    lifetime_lags: int
) -> pd.DataFrame:
    
    """
    Lifetime and times of event flag.

    Description:
        Lifetime flag that identified by any default occured.
        The first default will be assigned to time in the event of default.
        For non-default, lifetime flag will be 0 and time is longest information available.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        default_col (str)   : Default column that already flag (0, 1) for modeling.
        lifetime_lags(int)  : Defined n-lags for forward performance lifetime windows creation.

    Returns:
        pd.DataFrame: DataFrame with lifetime default flag and times of event appended.

    Notes:
        - N/A
    """

    cols = _lag_cols(default_col, lifetime_lags - 1) #-1 because lifetime forward until n-1.
    window = df[cols]
    any_default = window.eq(1).any(axis = 1)
    df["lifetime_flag"] = any_default.astype(int)
    df["times"] = np.where(
        any_default,
        window.eq(1).values.argmax(axis = 1) + 1,
        window.notna().sum(axis = 1)
    )

    return df

def drop_cols(
    df: pd.DataFrame,
    default_col: str,
    lifetime_lags: int
) -> pd.DataFrame:
    
    """
    Drop unused columns.

    Description:
        Preserving the memory by dropping unused/finished columns.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        default_col (str)   : Default column that already flag (0, 1) for modeling.
        lifetime_lags (int) : Defined n-lags for forward performance windows dropping.

    Returns:
        pd.DataFrame: DataFrame with forward performance columns dropped.

    Notes:
        - N/A.
    """
    
    flag_cols_to_drop = [f'{default_col}{i}' for i in range(1, lifetime_lags)]
    df = df.drop(columns = flag_cols_to_drop)

    return df

# ODR Calculation
def odr_series(
    df: pd.DataFrame,
    period_col: str,
    default_col: str
) -> None:
    
    """
    Monthly ODR.

    Description:
        Compute monthly default rates for the forward-looking model.

    Args:
        df (pd.DataFrame)   : Input dataframe.
        period_col (str)    : Period key for summary ('AS_OF_DATE') --> For datatime lable.
        default_col (str)   : Ever default column that already flag (0, 1) for modeling.

    Returns:
        Parquet file: Storaged file as .parquet format in '../data/processed'.

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[ODR Calculation]")

    filename = "monthly_odr"
    agg = {
        "n": (default_col, "size"),
        "bad": (default_col, "sum"),
        "odr": (default_col, "mean")
    }
    odr = df.groupby(period_col).agg(**agg)
    odr.to_parquet(
    f"../data/processed/{filename}.parquet",
    engine = 'pyarrow'
    )
    
    return print(f"=== Result ===\n[Export location: '..data/processed/{filename}.parquet']")