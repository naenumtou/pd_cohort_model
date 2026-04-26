
import warnings
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

# Cohort count
def cohort_count(
    segment_col: str,
    period_col: str,
    lifetime_col: str,
    default_col: str
) -> None:
    
    """
    Cohort count.

    Description:
        Compute raw summary of cohort in the monthly basis and times basis.

    Args:
        segment_col (str)   : Final segmentation key for modeling.
        period_col (str)    : Period key for summary.
        lifetime_col (str)  : Lifetime observed key for summary.
        default_col (str)   : Lifetime default column that already flag (0, 1) for modeling.

    Returns:
        Parquet file: Storaged file as .parquet format in '../data/processed'.

    Notes:
        - This is NOT a final cohort result as it will be used for further modeling steps.
    """

    print("=== Processing ===\n[Cohort count]")

    filename = "cohort_count"

    df = pd.DataFrame(
        {
            "segment": segment_col,
            "date": period_col,
            "times": lifetime_col,
            'default': default_col.values
        }
    )
    agg = {
        "n": ("default", "size"),
        "bad": ("default", "sum")
    }
    cohort = df.groupby(["segment", "date", "times"], as_index = False).agg(**agg)
    cohort.to_parquet(
    f"../data/processed/{filename}.parquet",
    engine = 'pyarrow'
    )
    
    return print(f"=== Result ===\n[Export location: '..data/processed/{filename}.parquet']")