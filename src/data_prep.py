
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Plot waterfall
def plot_exclusion(
    log: list
) -> None:
    
    """
    Plot waterfall exclusion.

    Description:
        Plot count of waterfall exclusion on each criteria.

    Args:
        log (list): List of excluded counts.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A
    """

    df_plot = (
        pd.DataFrame(log, columns = ['Criteria', 'Before', 'After'])
        .set_index('Criteria')
    )

    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray    
    colors = ['red'] * len(df_plot)
    colors[0] = colorG
    colors[-1] = colorY

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.bar(df_plot.index, df_plot["Before"], color = colors)
    ax.bar(df_plot.index, df_plot["After"], color = 'white')
    ax.set_yticklabels([f"{int(x):,}" for x in ax.get_yticks()])
    ax.set_title("Waterfall exclusion")
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Number of observation")
    ax.tick_params(axis = "x", rotation = 90)
    plt.tight_layout()

    return plt.show()

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
        period_col (str)    : Period key for plotting ('AS_OF_DATE') --> For datatime lable.
        default_col (str)   : Ever default column that already flag (0, 1) for modeling.

    Returns:
        Parquet file: Storaged file as .parquet format in '../data/processed'.

    Notes:
        - N/A.
    """

    print("=== Processing ===\n[ODR Calculation]")
    agg = {
        "n": (default_col, "size"),
        "bad": (default_col, "sum"),
        "odr": (default_col, "mean")
    }

    odr = df.groupby(period_col).agg(**agg)
    odr.to_parquet(
    '../data/processed/odr.parquet',
    engine = 'pyarrow'
    )
    
    return print("=== Result ===\n[Export location: '..data/processed/odr.parquet']")
