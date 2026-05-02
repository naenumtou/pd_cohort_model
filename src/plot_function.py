
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Helper function
def _extract_for_plot(
    key_tuple: tuple
) -> str:
    
    """
    Extract string.
    
    Description:
        Extracting string from dictionary keys for setting plot title.

    Args:
        key_tuple (tuple): Keys from dictionary.

    Returns:
        String: String text for plotting.
    
    Notes:
        - N/A.
    """

    nums = tuple(int(re.search(r"\d+", s).group()) for s in key_tuple)

    return f"({nums[0]})" if len(nums) == 1 else f"{nums}"

# Plot waterfall exclusion
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

# Plot ROC Curve
def plot_roc(
    cum_good: pd.Series,
    cum_bad: pd.Series,
) -> None:
    
    """
    Plot ROC Curve.

    Description:
        Plot Receiver Operating Characteristic.
        ROC is a graphical plot illustrating the performance of a binary classification model at threshold.

    Args:
        cum_good (pd.Series)    : Cumulative of good distribution.
        cum_bad (pd.Series)     : Cumulative of bad distribution.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    roc = (cum_good - cum_good.shift(1, fill_value = 0)) * (cum_bad + cum_bad.shift(1, fill_value = 0)) * 0.5
    auc = roc.sum()

    plt.figure(figsize = (10, 6))
    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray    
    plt.plot(
        np.hstack((0, cum_good)),
        np.hstack((0, cum_bad)),
        color = colorY,
        linewidth = 2
    )
    plt.plot([0, 1], [0, 1], c = colorG, linestyle = '--', linewidth = 2)
    plt.plot([], [], ' ', label = f"AUC: {auc * 100:.2f}%")
    plt.plot([], [], ' ', label = f"GINI: {(2 * auc - 1) * 100:.2f}%")
    plt.gca().set_yticklabels([f'{i * 100:.2f}%' for i in plt.gca().get_yticks()])
    plt.gca().set_xticklabels([f'{i * 100:.2f}%' for i in plt.gca().get_xticks()])
    plt.title('ROC Curve')
    plt.xlabel('Percentage of non-defaults')
    plt.ylabel('Percentage of defaults')
    plt.legend(frameon = True, facecolor = 'white')
    plt.tight_layout()

    return plt.show()

# Plot KS
def plot_ks(
    bin_labels: pd.Series,
    cum_good: pd.Series,
    cum_bad: pd.Series,
) -> None:
    
    """
    Plot KS Curve.

    Description:
        Plot Kolmogorov-Smirnov.
        KS measures the maximum separation between the cumulative distribution functions (CDFs) of two samples.

    Args:
        bin_labels (pd.Series)  : Output from segmentation. (Easy for plotting).
        cum_good (pd.Series)    : Cumulative of good distribution.
        cum_bad (pd.Series)     : Cumulative of bad distribution.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    diff = (cum_good - cum_bad).abs().reset_index(drop = True)
    ks = diff.max()
    ks_idx = diff.idxmax()
    cg = cum_good.reset_index(drop = True).loc[ks_idx]
    cb = cum_bad.reset_index(drop = True).loc[ks_idx]

    plt.figure(figsize = (10, 6))
    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray
    plt.plot(
        range(0, len(cum_good)),
        cum_good,
        label = 'Cumulative good',
        color = colorY,
        linewidth = 2
    )
    plt.plot(
        range(0, len(cum_bad)),
        cum_bad,
        label = 'Cumulative bad',
        color = colorG,
        linewidth = 2
    )
    
    plt.vlines(
        ks_idx,
        ymin = min(cg, cb),
        ymax = max(cg, cb),
        colors = "red",
        linestyles = "--",
        linewidth = 2,
    )
    
    plt.plot([], [], ' ', label = f"KS: {ks * 100:.2f}%")
    plt.gca().set_yticklabels([f'{i * 100:.2f}%' for i in plt.gca().get_yticks()])
    plt.xticks(ticks = [i for i in range(0, len(bin_labels))], labels = bin_labels, rotation = 90)
    plt.title('KS Curve')
    plt.xlabel('Segment')
    plt.ylabel('Cumulative distribution')
    plt.legend(frameon = True, facecolor = 'white')
    plt.tight_layout()

    return plt.show()

# Plot monthly classification back-testing
def plot_classification_monthly(
    month: pd.Series,
    bin_labels: pd.Series,
    y_true: pd.Series,
) -> None:
    
    """
    Plot monthly classification back-testing.

    Description:
        Testing classification ability on historical monthly basis.

    Args:
        month (pd.Series)       : The input of month date as a key for calculation.
        bin_labels (pd.Series)  : Output from segmentation.
        y_true (pd.Series)      : The actual target.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    df = pd.DataFrame(
        {
            "month": month,
            "bin": bin_labels,
            'default': y_true.values
        }
    )
    agg = {
        "n": ("default", "size"),
        "bad": ("default", "sum"),
        "odr": ("default", 'mean')
    }
    summary = (
        df.groupby(["month", "bin"], observed = True)
        .agg(**agg)
        .reset_index()
        .sort_values(["month", "odr"], ascending = [True , False])
    )
    summary['good'] = summary['n'] - summary['bad']
    summary["cum_bad"] = (
        summary.groupby("month")["bad"].cumsum() / summary.groupby("month")["bad"].transform("sum")
    )
    summary["cum_good"] = (
        summary.groupby("month")["good"].cumsum() / summary.groupby("month")["good"].transform("sum")
    )
    summary["roc"] = (
        (
            summary["cum_good"] - summary.groupby("month")["cum_good"].shift(1, fill_value = 0)

        ) * \
        (
            summary["cum_bad"] + summary.groupby("month")["cum_bad"].shift(1, fill_value = 0)
        ) * 0.5
    )
    summary["ks"] = abs(summary["cum_good"] - summary["cum_bad"])

    # Back-testing
    auc = summary.groupby("month")["roc"].sum()
    gini = 2 * auc - 1
    ks = summary.groupby("month")["ks"].max()

    # Plot
    fig, axs = plt.subplots(1, 3, figsize = (21, 4), sharex = True)
    fig.subplots_adjust(wspace = 0.2)
    plt.suptitle("Back-testing: Monthly classification", y = 1)
    axs = axs.ravel()

    axs[0].set_title('AUC')
    axs[0].plot(auc, color = 'royalblue', linewidth = 2)
    axs[0].margins(0) #Remove default margins
    axs[0].axhspan(0, 0.6, facecolor = '#C00000', alpha = 0.5)
    axs[0].axhspan(0.6, 0.7, facecolor = '#FFC000', alpha = 0.5)
    axs[0].axhspan(0.7, 1.0, facecolor = '#00B050', alpha = 0.5)
    axs[0].set_yticklabels([f"{y * 100:.2f}%" for y in axs[0].get_yticks()])
    axs[0].xaxis.set_major_locator(mdates.YearLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axs[1].set_title('GINI')
    axs[1].plot(gini, color = 'royalblue', linewidth = 2)
    axs[1].margins(0) #Remove default margins
    axs[1].axhspan(0, 0.2, facecolor = '#C00000', alpha = 0.5)
    axs[1].axhspan(0.2, 0.4, facecolor = '#FFC000', alpha = 0.5)
    axs[1].axhspan(0.4, 1.0, facecolor = '#00B050', alpha = 0.5)
    axs[1].set_yticklabels([f"{y * 100:.2f}%" for y in axs[1].get_yticks()])
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axs[2].set_title('KS')
    axs[2].plot(ks, color = 'royalblue', linewidth = 2)
    axs[2].margins(0) #Remove default margins
    axs[2].axhspan(0, 0.2, facecolor = '#C00000', alpha = 0.5)
    axs[2].axhspan(0.2, 0.4, facecolor = '#FFC000', alpha = 0.5)
    axs[2].axhspan(0.4, 1.0, facecolor = '#00B050', alpha = 0.5)
    axs[2].set_yticklabels([f"{y * 100:.2f}%" for y in axs[2].get_yticks()])
    axs[2].xaxis.set_major_locator(mdates.YearLocator())
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    return plt.show()

# Plot monthly stability back-testing
def plot_stability_monthly(
    month: pd.Series,
    bin_labels: pd.Series,
    y_true: pd.Series,
) -> None:

    """
    Plot monthly stability back-testing.

    Description:
        Testing model stability on historical monthly basis.

    Args:
        month (pd.Series)       : The input of month date as a key for calculation.
        bin_labels (pd.Series)  : Output from segmentation.
        y_true (pd.Series)      : The actual target.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """   

    df = pd.DataFrame(
        {
            "month": month,
            "bin": bin_labels,
            'default': y_true.values
        }
    )
    agg = {
        "n": ("default", "size")
    }
    dist = (
        df.groupby(["month", "bin"], observed = True)
        .agg(**agg)
        .reset_index()
        .assign(total = lambda x: x.groupby("month")["n"].transform("sum"))
        .assign(p = lambda x: x["n"] / x["total"])
        .assign(month_next = lambda x: x["month"] + pd.DateOffset(months=1))
        .sort_values(["month", "bin"], ascending = [True , False])
    )
    dist = pd.merge(
        dist,
        dist[["month", "bin", "p"]],
        how = "left",
        left_on = ["month_next", "bin"],
        right_on = ["month", "bin"],
        suffixes=("_t0", "_t1")
    )
    p0 = np.clip(dist["p_t0"], 1e-6, dist["p_t0"])
    p1 = np.clip(dist["p_t1"], 1e-6, dist["p_t1"])
    dist["psi"] = (p0 - p1) * np.log(p0 / p1)

    # Back-testing
    psi = dist.groupby("month_t0")["psi"].sum()
    portion = pd.pivot_table(
        data = dist,
        index = "month_t0",
        columns = "bin",
        values = "n",
        fill_value = 0
    )
    portion = portion[bin_labels.unique()].div(portion[bin_labels.unique()].sum(axis = 1), axis = 0)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize = (14, 4), sharex = True)
    fig.subplots_adjust(wspace = 0.2)
    plt.suptitle("Back-testing: Monthly stability", y = 1)
    axs = axs.ravel()

    axs[0].set_title('PSI')
    axs[0].plot(psi, color = 'royalblue', linewidth = 2)
    axs[0].margins(0) #Remove default margins
    axs[0].axhspan(0, 0.1, facecolor = '#00B050', alpha = 0.5)
    axs[0].axhspan(0.1, 0.25, facecolor = '#FFC000', alpha = 0.5)
    axs[0].axhspan(0.25, 1.0, facecolor = '#C00000', alpha = 0.5)
    axs[0].set_yticklabels([f"{y * 100:.2f}%" for y in axs[0].get_yticks()])
    axs[0].xaxis.set_major_locator(mdates.YearLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axs[1].set_title('Proportion')
    axs[1].stackplot(portion.index, portion.T, alpha = 0.7, labels = bin_labels.unique())
    axs[1].margins(0) #Remove default margins
    axs[1].set_yticklabels([f"{y * 100:.2f}%" for y in axs[1].get_yticks()])
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axs[1].legend(frameon = True, facecolor = 'white')

    return plt.show()

def plot_segment_odr(
    month: pd.Series,
    bin_labels: pd.Series,
    y_true: pd.Series,
    pool: list
) -> None:
    
    """
    Plot monthly ODR Back-testing.

    Description:
        Showing the ODR on historical monthly basis.

    Args:
        month (pd.Series)       : The input of month date as a key for calculation.
        bin_labels (pd.Series)  : Output from segmentation.
        y_true (pd.Series)      : The actual target.
        pool (list)             : Pool is the group of segmentation.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    df = pd.DataFrame(
        {
            "month": month,
            "segment": bin_labels,
            'default': y_true.values
        }
    )
    agg = {
        "odr": ("default", 'mean')
    }
    odr = (
        df.groupby(["month", "segment"], observed = True)
        .agg(**agg)
        .reset_index()
    )
    # Plot
    fig, axs = plt.subplots(1, len(pool), figsize = (24, 4), sharex = True)
    for ax, lst in zip(axs.flat, pool):
        fig.suptitle("ODR Analysis")
        odr_pool = odr[odr["segment"].isin(lst)]
        sns.lineplot(
            x = "month",
            y = "odr",
            data = odr_pool,
            hue = "segment",
            legend = "full",
            ax = ax
        )
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_yticklabels([f"{y * 100:.2f}%" for y in ax.get_yticks()])
        ax.legend(frameon = True, facecolor = 'white', loc = 'upper right')
        
    return plt.show()

# Plot cumulative ODR by pool
def plot_lifetime(
    data: dict,
    plot_title: str
) -> None:
    
    """
    Plot cumulative lifetime ODR by cohort.

    Description:
        Showing the cumulative lifetime ODR by cohort.

    Args:
        data (dict)         : The dictionary contains cumulative lifetime ODR by cohort.
                            {keys: values} --> {pool (tuple , str): ODR (pd.DataFrame)}
        plot_title (str)    :Name of the plot.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    # Plot
    fig, axs = plt.subplots(2, int(len(data) / 2), figsize = (20, 8), sharex = True)
    for ax, (key, value) in zip(axs.flat, data.items()):
        val_long = (
            value
            .reset_index(level = "date")
            .melt(
                id_vars = "date",
                var_name = "times",
                value_name = "odr"
            )
            .dropna()
        )

        label = _extract_for_plot(key)
        fig.suptitle(plot_title)
        sns.lineplot(
            x = "times",
            y = "odr",
            data = val_long,
            hue = "date",
            legend = "full",
            palette = "viridis",
            errorbar = None,
            marker = None,
            ax = ax
        )
        ax.set_title(f"Segment - {label}")
        ax.set_ylabel('')
        ax.set_xlabel('Lifetime')
        ax.set_yticklabels([f"{y * 100:.2f}%" for y in ax.get_yticks()])
        ax.legend(frameon = True, facecolor = 'white', loc = 'upper right')
        
    return plt.show()

# Plot weighted average
def plot_lifetime_avg(
    data: dict,
    plot_title: str
) -> None:
    
    """
    Plot weighted average cumulative lifetime ODR.

    Description:
        Showing the weighted average cumulative lifetime ODR.

    Args:
        data (dict)         : The dictionary contains cumulative lifetime ODR.
                            {keys: values} --> {pool (tuple , str): ODR (np.array)}
        plot_title (str)    : Name of the plot.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    # Plot
    plt.figure(figsize = (10, 6))
    for key, values in data.items():
        plt.plot(values, label = str(key))
    plt.gca().set_yticklabels([f'{i * 100:.2f}%' for i in plt.gca().get_yticks()])
    plt.gca().set_xticklabels([f'{int(i + 1)}' for i in plt.gca().get_xticks()])
    plt.xlabel("Lifetime")
    plt.ylabel("ODR")
    plt.title(plot_title)
    plt.legend(frameon = True, facecolor = 'white', loc = 'upper left')
    plt.tight_layout()

    return plt.show()

# Plot curves comparision
def plot_lifetime_comp(
    actual: dict,
    fitted: dict,
    plot_title: str
) -> None:
    
    """
    Plot comparision between weighted average lifetime ODR and fitted lifetime ODR.

    Description:
        Showing the comparision between weighted average lifetime ODR and fitted lifetime ODR.

    Args:
        actual (dict)       : The dictionary contains weighted average cumulative lifetime ODR.
                            {keys: values} --> {pool (tuple , str): ODR (np.array)}
        fitted (dict)       : The dictionary contains fitted cumulative lifetime ODR.
                            {keys: values} --> {pool (tuple , str): ODR (np.array)}
        plot_title (str)    : Name of the plot.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    fig, axs = plt.subplots(2, int(len(actual) / 2), figsize = (20, 8), sharex = False)
    fig.suptitle(plot_title)
    for ax, (key, actual_odr), (_, fitted_odr)in zip(axs.flat, actual.items(), fitted.items()):
        label = _extract_for_plot(key)
        ax.set_title(f"Segment - {label}")
        ax.plot(
            actual_odr,
            label = "Actual",
            color = "#0070C0",
            linewidth = 2
        )
        ax.plot(
            fitted_odr,
            label = "Fitted",
            color = "#61CBF4",
            linestyle = "--",
            linewidth = 2
        )
        ax.set_yticklabels([f"{y * 100:.2f}%" for y in ax.get_yticks()])
        ax.set_xticklabels([f"{int(x + 1)}" for x in ax.get_xticks()])
        ax.set_xlabel('Lifetime')
        
        # Marginal ODR
        ax_r = ax.twinx()
        ax_r.plot(
            np.concatenate([[fitted_odr[0]], np.diff(fitted_odr)]), #Keep first position
            label = "Marginal",
            color = "#47D45A",
            linestyle = "--",
            linewidth = 2
        )
        ax_r.set_yticklabels([f"{y * 100:.2f}%" for y in ax_r.get_yticks()])

        # Keep labels for legend
        lines, labels = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        ax.legend(lines + lines_r, labels + labels_r, loc = "center right")
     
    plt.tight_layout()
        
    return plt.show()

# Plot unbias lifetime curves
def plot_unbias_lifetime(
    fitted: dict,
    unbias: dict,
    plot_title: str
) -> None:
        
    """
    Plot unbias lifetime ODR.

    Description:
        Showing the unbias lifetime ODR.

    Args:
        fitted (dict)       : The dictionary contains fitted cumulative lifetime ODR.
                            {keys: values} --> {pool (tuple , str): ODR (np.array)}
        unbias (dict)       : Keys are segmentation name corresponding to the pool.
                            Values are unbias calibration results
                            {keys: values} --> {
                                                segment (str): Unbias calibration results (dict) --> {
                                                                                                      "n": int,
                                                                                                      "Unbias": np.array,
                                                                                                      }
                                                }
        plot_title (str)    : Name of the plot.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    fig, axs = plt.subplots(2, int(len(fitted) / 2), figsize = (20, 8), sharex = False)
    fig.suptitle(plot_title)
    
    for ax, (lifetime_key, lifetime_value) in zip(axs.flat, fitted.items()):
            
            # Plot pool lifetime
            label = _extract_for_plot(lifetime_key)
            ax.set_title(f"Segment - {label}")
            ax.plot(
                lifetime_value,
                label = "Pool level",
                color = "#61CBF4",
                linestyle = "--",
                linewidth = 2
            )

            for unbias_key, unbias_value in unbias.items():
                if unbias_key in lifetime_key:

                    # Plot segment lifetime
                    ax.plot(
                        unbias_value["Unbias"],
                        label = unbias_key,
                        linewidth = 2
                    )
                    ax.set_yticklabels([f"{y * 100:.2f}%" for y in ax.get_yticks()])
                    ax.set_xticklabels([f"{int(x + 1)}" for x in ax.get_xticks()])
                    ax.set_xlabel('Lifetime')
                    ax.legend(frameon = True, facecolor = 'white', loc = "center right")

    plt.tight_layout()
    
    return plt.show()

# Plot series dependence variable
def plot_dep_var(
    y: pd.Series,
    y_target: pd.Series,
    target_label: str
) -> None:
    
    """
    Plot dependence variables for regression model.

    Description:
        Showing the historical dependence variables (logit, CF, CCI) for regression model.

    Args:
        y (pd.Series)           : The actual monthly ODR.
        y_target (pd.Series)    : The transformed dependence variable.
        target_label (str)      : The method name. E.g., "Logit", "CF", "CCI".

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - This function will be called in the src.regression_model in prepare_training_set().
    """
    
    fig, ax = plt.subplots(figsize = (10, 6))
    fig.suptitle("Historical dependence variable")
    colorY = '#ffd500' #Set color theme --> Yellow
    colorG = '#808080' #Set color theme --> Gray

    ax.plot(y_target, color = colorY, linewidth = 2, label = target_label)
    ax.set_yticklabels([f"{y:.4f}" for y in ax.get_yticks()])

    if target_label != "CCI":
        ax_r = ax.twinx()
        ax_r.plot(y, color = colorG, linestyle = "--", linewidth = 2, label = "ODR")
        ax_r.set_yticklabels([f"{y * 100:.2f}%" for y in ax_r.get_yticks()])
        lines, labels = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        ax.legend(lines + lines_r, labels + labels_r, loc = "upper right")
    else:
        ax.legend(loc = "upper right")
    plt.tight_layout()
    
    return fig
