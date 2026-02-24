

from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import matplotlib.cm as cmapper
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
import polars as pl
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import pdist, squareform
from random import randint, uniform
from matplotlib.patches import Patch
from itertools import chain
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter
from typing import Dict, Union, Optional, Tuple, List

# load bacpy modules
from .taxonomy import taxonomy_df


def _generate_random_rgba_colors(n, transparent=False):
    """
    function to generate random rgb or rbga colors
    ----------
    n : int
        number of colors to generate
    transparent : bool
        default=False 
        True generates transparent rgba colors
    Returns
    ----------
        list of n random colors
    """
    colors = []
    for _ in range(n):
        r = randint(0, 255)/255
        g = randint(0, 255)/255
        b = randint(0, 255)/255
        a = uniform(0, 1)
        if transparent:
            colors.append((r, g, b, a))
        else:
            colors.append((r, g, b))
    return colors


def _get_color(color_map, n_taxonomic_classes, transparent):
    """
    functin to get colors
    ----------
    color_map : str
        matplotlib colormap
    n_taxonomic_classes : int
        number of colors we need
    transparent : bool
        whether or not the colors should be transparent (rgba instead of rgb)
    Returns
    ----------
    colors : list
        list of n_taxonomic_classes colors
    """
    cmap = plt.colormaps[color_map]
    colors = cmap(np.linspace(0, 1, n_taxonomic_classes))
    colors = colors if np.unique(colors, axis=0).shape[0] >= n_taxonomic_classes else np.array(_generate_random_rgba_colors(n_taxonomic_classes, transparent=transparent))
    return colors


def _feature_metadata_split(rf_dat):
    """
    function to split rf_dat into features and metadata
    ----------
    rf_dat : pl.DataFrame
    Returns
    ----------
    feature_df, metadata
        2 pl.DataFrames
    """
    # subset columns
    feature_set = [feature for feature in rf_dat.columns if (feature.startswith("wv")) or (feature in ["od"])]
    feature_df = rf_dat.select(feature_set)
    metadata   = rf_dat.select(pl.exclude(feature_set))
    return feature_df, metadata


# ~~~~~~~~~~~~~~~~~~~~~~~~~ CONFUSSION MATRIX ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def _plot_single_cm(confusion_matrix, 
                    taxonomic_level, 
                    annotation, 
                    color_map, 
                    reds,
                    figsize=(16,10),
                    ):
    """
    function to plot a single confusion-matrix
    ----------
    confusion_matrix : pl.DataFrame
        confusion matrix formatted as pl.DataFrame
    taxonomic_level : str
        which taxonomic level is the confusion matrix
    annotation : str
        annotation must a column in confusion matrix, used to plot an extra column and row annotating the data with colors
    color_map : str
        which color map to use for annotations
    reds : matplotlib.colormap
        color map to use for heat
    Returns
    ----------
    g : seaborn.clustermap
        a single confusion matrix
    """

    # conditional branch of annotation is to be added
    if annotation:

        # map the taxonomic level back to taxonomy_df
        annotation_df = (confusion_matrix
                                .select(taxonomic_level)
                                .join(taxonomy_df.select(annotation, taxonomic_level).unique(subset=taxonomic_level, keep="first"), how="left", on = taxonomic_level)
                                .sort(annotation))

        # adjust order of heatmap according to taxonomy
        order_ls = list(annotation_df[taxonomic_level])
        order_ls.append(taxonomic_level)
        confusion_matrix = (confusion_matrix
                                .sort(pl.col(taxonomic_level).cast(pl.Enum(annotation_df[taxonomic_level].unique(maintain_order=True))))
                                .select(order_ls)
                                )

        # get colors and check if enough
        taxonmic_classes = annotation_df[annotation].unique()
        n_taxonomic_classes = len(taxonmic_classes)
        colors = _get_color(color_map, n_taxonomic_classes, transparent=False)

        # now, map the rows to colors
        annotation_df = (annotation_df
                                .join(pl.DataFrame({annotation: taxonmic_classes, "colors": colors}), how="left", on=annotation)
                                .sort(pl.col(taxonomic_level).cast(pl.Enum(confusion_matrix[taxonomic_level].unique(maintain_order=True))))
                                )
        annotation_colors = (annotation_df.select(taxonomic_level, "colors").rename({"colors": "Taxonomy"}).to_pandas().set_index(taxonomic_level))


    else:
        annotation_colors = None


    # convert to pandas and plot
    confusion_matrix_pd = confusion_matrix.select(pl.exclude(taxonomic_level)).to_pandas()
    confusion_matrix_pd.index = confusion_matrix[taxonomic_level].to_list()
    g = sns.clustermap(
                    confusion_matrix_pd, 
                    row_cluster=False, 
                    col_cluster=False, 
                    row_colors = annotation_colors,
                    col_colors = annotation_colors,
                    annot=True, 
                    fmt="d", 
                    linecolor="grey", 
                    cmap=reds,
                    linewidths=.05, 
                    mask=confusion_matrix_pd < 1,
                    annot_kws={'color':'black'},
                    figsize=figsize,
                    dendrogram_ratio=(0.1, 0.05),
                    )
    
    
    # adjust figure for legend
    g.figure.subplots_adjust(right=0.75, left=0)
    
    if annotation:
        # add colors
        annotation_df = annotation_df.select(annotation, "colors").unique(annotation, maintain_order=True)
        lut = {key: color for key, color in zip(annotation_df[annotation], annotation_df["colors"])}
        handles = [Patch(facecolor=lut[name]) for name in lut]
        g.figure.legend(handles, lut, title=annotation, bbox_to_anchor=(0.95, 0.5), bbox_transform=plt.gcf().transFigure, loc='center right')

    # modify color bar
    g.ax_cbar.set_position((0.025, 0.3, 0.01, 0.4)) # tuple of (left, bottom, width, height), optional

    # mark diagonal
    for idx, vals in enumerate(zip(confusion_matrix_pd.columns, confusion_matrix_pd.index)):
        g.ax_heatmap.add_patch(Rectangle((idx, idx), 1, 1, edgecolor='black', fill=False, lw=1))

    # title
    g.ax_heatmap.set_title(f"Confusion matrix for {taxonomic_level} predictions", y=1.05 if annotation else 1)
    g.ax_heatmap.set_ylabel(f"Truth")
    g.ax_heatmap.set_ylabel(f"Prediction")
    return g


def plot_confusion_matrix(
                            confusion_matrices: Dict[str, pl.DataFrame], 
                            taxonomic_level: Union[str, bool] = "strainID", 
                            annotation: Union[str, bool] = "family", 
                            color_map: str = "tab20", 
                            figure_name: str = "confusion_matrix",
                            figsize: Tuple[int, int] = (16, 10),
                        ) -> None:
    """
    Plots and saves confusion matrices for taxonomic classification results.

    This function visualizes the relationship between predicted and true labels. 
    It features a custom color-mapped scale to highlight classification density 
    and supports hierarchical annotations, allowing rows and columns to be 
    grouped and colored by a secondary taxonomic rank (e.g., coloring 
    individual strains by their family).

    Args:
        confusion_matrices: A dictionary where keys are taxonomic levels 
            (e.g., 'genus') and values are Polars DataFrames representing 
            the confusion matrix.
        taxonomic_level: The specific rank to plot. If set to False, 
            the function iterates through and plots all matrices in the dictionary.
        annotation: A secondary taxonomic column present in the matrix used 
            to create color-coded bars for row/column grouping. Set to False 
            to disable.
        color_map: The name of the Matplotlib colormap used for the 
            categorical annotations (e.g., 'tab20', 'Set3').
        figure_name: Base filename for the output files. Saves in .png, 
            .pdf, and .svg formats.
        figsize: The width and height of the resulting figure in inches.

    Returns:
        None. Saves images directly to the current working directory.
    """

    # color settings
    reds = cmapper.get_cmap('Reds', 256)
    newcolors = reds(np.linspace(0, 1, 256))
    newcolors = newcolors[0:150,:]
    newcolors[0,:] = 0
    reds = ListedColormap(newcolors)


    # draw an individual confusion matrix
    if taxonomic_level:

        # generate the cm
        confusion_matrix = confusion_matrices[taxonomic_level]
        fig = _plot_single_cm(confusion_matrix, taxonomic_level, annotation, color_map, reds, figsize)

        # write to disk
        fig.savefig(f"{figure_name}.png", bbox_inches='tight', dpi=600)
        fig.savefig(f"{figure_name}.pdf", bbox_inches='tight')
        fig.savefig(f"{figure_name}.svg", bbox_inches='tight')


    # branch for all cm
    else:
        taxonomic_levels = np.array(list(confusion_matrices.keys()))
        for key in taxonomic_levels:

            # generate the cm
            fig = _plot_single_cm(confusion_matrices[key], key, False, color_map, reds, figsize)

            # write to disk
            fig.savefig(f"{figure_name}_{key}.png", dpi=600)
            fig.savefig(f"{figure_name}_{key}.pdf")
            fig.savefig(f"{figure_name}_{key}.svg")
    plt.clf()
# ~~~~~~~~~~~~~~~~~~~~~~~~~ CONFUSSION MATRIX ~~~~~~~~~~~~~~~~~~~~~~~~~ #


# ~~~~~~~~~~~~~~~~~~~~~~~~~ TAX-METRICS ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def plot_stats(
                    stats: pl.DataFrame,
                    figure_name: str = "stats_plot",
                    kind: str = "bar",
                    figsize: Tuple[int, int] = (8, 5),
              ) -> None:
    """
    Plots classification performance metrics across various taxonomic levels.

    This function takes the statistics generated by `model.evaluate`, reshapes 
    them into a long format, and visualizes accuracy, F1-score, and Matthews 
    Correlation Coefficient (MCC). It supports different 
    distributional visualizations and exports high-resolution files.

    Args:
        stats: Polars DataFrame containing metrics ('accuracy', 'f1', 'mcc') 
            and a 'taxonomic_level' column.
        figure_name: Base path and filename for the exported plots. 
            Saves as .png, .pdf, and .svg.
        kind: The type of visualization to generate. 
            Options: 'bar' (mean with CI), 'box' (quartiles), or 'violin' (density).
        figsize: A tuple defining the width and height of the figure in inches.

    Returns:
        None. Saves images directly to disk.
    """

    # convert the final stats-df to pandas for seaborn
    stats_pd = (stats
                    .select(["taxonomic_level", "accuracy", "f1", "mcc"])
                    .unpivot(index="taxonomic_level", variable_name = "Metric", value_name="Score")
                    .rename({"taxonomic_level": "Taxonomic Level"}).to_pandas())

    # generate the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if kind == "violin":
        ax = sns.violinplot(data = stats_pd,
                            x = "Taxonomic Level",
                            y = "Score",
                            hue = "Metric",
                            inner=None,
                            alpha=0.5,
                            legend=True,
                            )

    # plotting
    if kind == "box" or kind == "violin":
        ax = sns.boxplot(data = stats_pd,
                         x = "Taxonomic Level",
                         hue="Metric",
                         y = "Score",
                         dodge=True,
                         legend=False,
                         fill=False,
                         linecolor="black",
                         gap=0.5,
                         ax=ax,
                         showfliers=False,
                         )
        
        if kind == "violin":
            plt.setp(ax.lines, color='gray')
        
        sns.stripplot(data = stats_pd,
                        x = "Taxonomic Level",
                        y = "Score",
                        hue = "Metric",
                        dodge=True,
                        legend=True if kind == "box" else False,
                        alpha=1,
                        ax=ax)

    if kind == "bar":
        ax = sns.barplot(data = stats_pd,
                         x = "Taxonomic Level",
                         y = "Score",
                         hue = "Metric",
                         alpha=0.5,
                         legend=True,
                         errorbar="ci"
                        )
    
    # general fix
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # general adjustments
    fig.suptitle(f"Metrics across different taxonomic levels")
    fig.tight_layout()

    # write the plot to disc
    fig.savefig(f"{figure_name}.png", dpi=600)
    fig.savefig(f"{figure_name}.pdf")
    fig.savefig(f"{figure_name}.svg")
    plt.clf()
# ~~~~~~~~~~~~~~~~~~~~~~~~~ TAX-METRICS ~~~~~~~~~~~~~~~~~~~~~~~~~ #



# ~~~~~~~~~~~~~~~~~~~~~~~~~ CLUSTER MAP ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def plot_correlogram(
                        rf_dat: pl.DataFrame, 
                        variable: str = "features",
                        aggregate: bool = True,
                        figure_name: str = "correlogram",
                        figsize: Tuple[int, int] = (20, 20),
                    ) -> None:
    """
    Generates a hierarchical cluster map (clustermap) to visualize correlations.

    This function calculates a correlation matrix for either the 
    spectral features (excitation-emission pairs) or the biological samples 
    (strains/taxonomic levels). It then performs hierarchical clustering to 
    group similar variables together, revealing patterns in the spectral 
    fingerprints or biological relationships.

    Args:
        rf_dat: The processed wide-format Polars DataFrame.
        variable: If "features", calculates correlation between spectral channels. 
            If set to a column name (e.g., "strainID"), calculates correlation 
            between biological samples.
        aggregate: Only applies if variable is not "features". If True, calculates 
            the mean spectral profile per group before correlating. Recommended 
            for large datasets to improve readability.
        figure_name: Base filename for saving results (.png, .pdf, .svg).
        figsize: A tuple defining width and height in inches.

    Returns:
        None. Saves the high-resolution clustermap to disk.
    """

    # get the feature columns
    feature_set = [feature for feature in rf_dat.columns if (feature.startswith("wv")) or (feature in ["od"])]

    # make the figure
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # plot feature correlation
        if variable == "features":

            # prepare the data
            cluster_data = (rf_dat
                                .select(feature_set)
                                .corr()
                                .with_columns(pl.Series(feature_set).alias("index"))
                                .fill_nan(0)
                                .to_pandas()
                                .set_index("index"))
            cluster_map = sns.clustermap(cluster_data, figsize=figsize)


        # plot strain correlation
        else:

            # aggregate if large dataframe
            if aggregate:
                rf_dat = rf_dat.select(list(chain(*[[variable], feature_set]))).group_by(variable).mean()


            # get the names and add index to ensure uniqueness
            variable_classes = rf_dat[variable].to_numpy()
            variable_classes = variable_classes + "-" + np.arange(variable_classes.shape[0]).astype(str)

            # transpose and change column names
            cluster_data = rf_dat.select(feature_set).transpose()
            cluster_data.columns = variable_classes
            
            # perform the clustering
            cluster_data = (cluster_data
                                .corr()
                                .with_columns(pl.Series(variable_classes).alias("index"))
                                .fill_nan(0)
                                .to_pandas()
                                .set_index("index"))
            cluster_map = sns.clustermap(cluster_data, figsize=[10 if aggregate else 20]*2)

            cluster_map.ax_heatmap.set_xticklabels([text._text.split("-")[0] for text in cluster_map.ax_heatmap.get_xticklabels()])
            cluster_map.ax_heatmap.set_yticklabels([text._text.split("-")[0] for text in cluster_map.ax_heatmap.get_yticklabels()])
            cluster_map.ax_cbar.set_position((0.05, 0.2, 0.01, 0.4)) # tuple of (left, bottom, width, height), optional

    # draw the plot
    cluster_map.figure.savefig(f"{figure_name}.png", dpi=600)
    cluster_map.figure.savefig(f"{figure_name}.pdf")
    cluster_map.figure.savefig(f"{figure_name}.svg")
    plt.clf()
# ~~~~~~~~~~~~~~~~~~~~~~~~~ CLUSTER MAP ~~~~~~~~~~~~~~~~~~~~~~~~~ #



# ~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE IMPORTANCE ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def _transform_coods(heatmap_data, 
                     highlight):

    # get columns and rows
    rowsFull = heatmap_data.index.to_numpy().astype(str)
    colsFull = heatmap_data.columns.to_numpy().astype(str)

    # now get the coordinates
    rows = highlight["ex"].to_numpy().astype(str)
    rows = (pl.DataFrame({"rows": rows})
                    .join(pl.DataFrame({"rows": rowsFull})
                          .with_columns(pl.Series(np.arange(rowsFull.shape[0])).alias("index")), how="left", on="rows")["index"]
                          .to_numpy())

    cols = highlight["em"].to_numpy().astype(str)
    cols = (pl.DataFrame({"cols": cols})
                    .join(pl.DataFrame({"cols": colsFull}).with_columns(pl.Series(np.arange(colsFull.shape[0])).alias("index")), how="left", on="cols")["index"].to_numpy())

    return rows, cols

def plot_feature_importances(
                                feature_importances: pl.DataFrame, 
                                figure_name: str = "feature_heatmap", 
                                highlight: Union[pl.DataFrame, bool] = False,
                                figsize: Tuple[int, int] = (12, 10)
                            ) -> None:
    """
    Visualizes spectral feature importances as an Excitation-Emission heatmap.

    This function transforms a long-format importance table into a 2D matrix, 
    plotting it as a heatmap where color intensity represents a feature's 
    contribution to the model's decision-making. It allows for the highlighting 
    of specific 'useful' features (e.g., those selected for a smaller assay).

    Args:
        feature_importances: Polars DataFrame containing 'ex', 'em', and 
            'importance' columns. Usually obtained from 
            `model.get_features_importances()`.
        figure_name: Base filename for saving the resulting heatmap. 
            Saves as .png, .pdf, and .svg.
        highlight: Optional Polars DataFrame containing an 'ex_em' column. 
            Features matching these coordinates will be outlined with black boxes.
        figsize: The dimensions of the figure in inches.

    Returns:
        None. Saves high-resolution plots to the local directory.
    """

    fontsize = 15

    features_wide = (feature_importances
                        .select("ex", "em", "importance")
                        .group_by("ex", "em")
                        .agg(pl.col("importance").sum())
                        .sort("ex", descending=True)
                        .pivot(on="em", index="ex", values="importance", sort_columns=True)
                        )

    # create the figure canvas
    fig, ax = plt.subplots(figsize=figsize)

    # plotting
    heatmap_data = (features_wide
                            .to_pandas()
                            .set_index("ex"))
    ax = sns.heatmap(heatmap_data, 
                     cmap = "Reds", 
                     linewidths=.05, 
                     linecolor="grey", 
                     cbar = True, 
                     ax=ax)
    
    # adjust ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
    
    # adjust names
    ax.set_ylabel("Excitation wavelength [nm]", fontsize = fontsize+8)
    ax.set_xlabel("Emission wavelength [nm]", fontsize = fontsize+8)
    
    # general aesthetics
    ax.set_facecolor('lightgrey')
    ax.set_title(f'Feature importances', y=1.02, fontsize = fontsize+15)
    fig.tight_layout()

    # mark top n features
    if highlight is not False:
        rows, cols = _transform_coods(heatmap_data, highlight)
        for idx, _ in enumerate(cols):
            ax.add_patch(Rectangle((cols[idx], rows[idx]), 1, 1, edgecolor='black', fill=False, lw=2))

    fig.savefig(f"{figure_name}.png", dpi=600)
    fig.savefig(f"{figure_name}.pdf")
    fig.savefig(f"{figure_name}.svg")
    plt.clf()
# ~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE IMPORTANCE ~~~~~~~~~~~~~~~~~~~~~~~~~ #


# ~~~~~~~~~~~~~~~~~~~~~~~~~ CUMULATIVE FEATURE IMPORTANCE ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def plot_cumulative_importance(
                                feature_importances: pl.DataFrame,
                                aggregate: bool = True,
                                cutoff: Union[bool, float, int] = False,
                                figure_name: str = "cumulative_importance",
                                figsize: Tuple[int, int] = (8, 5),
                              ) -> None:
    """
    Plots cumulative feature importances.

    This function ranks features by their importance and plots a dual-axis chart: 
    a bar chart for individual importance and a line plot for cumulative 
    importance. It can aggregate by excitation wavelength to show which spectral 
    regions contribute most to the total signal.

    Args:
        feature_importances: Polars DataFrame containing 'ex' and 'importance' 
            columns, or 'feature' and 'importance'.
        aggregate: If True, sums importances by excitation wavelength ('ex'). 
            If False, plots individual features.
        cutoff: Optional threshold to visualize a selection limit.
            - float: The cumulative importance percentage (e.g., 0.8 for 80%).
            - int: The top N number of features to keep.
            - False: Disables cutoff visualization.
        figure_name: Base filename for saving (.png, .pdf, .svg).
        figsize: A tuple defining width and height in inches.

    Returns:
        None. Saves the high-resolution dual-axis plot to disk.
    """
    
    # plot cummulative feature importances (ex)
    if aggregate:
        x = "ex"
        x_axis = "Excitation wavelength"
        agg_importance = (feature_importances
                            .group_by("ex")
                            .agg(pl.col("importance").sum())
                            .sort("importance", descending=True)
                            .with_columns(pl.col("importance").cum_sum().alias("cum_importance"))
                            )            

    else:
        x = "feature"
        x_axis = "Feature"
        agg_importance = (feature_importances
                            .sort("importance", descending=True)
                            .with_columns(pl.col("importance").cum_sum().alias("cum_importance")))
    
    # normalize, so that importance is always as percent
    agg_importance = (agg_importance
                        .with_columns((pl.col("cum_importance")/pl.col("cum_importance").max()).alias("cum_importance")))
    
    if cutoff:
        if type(cutoff) == float:
            agg_importance = (agg_importance
                                .with_columns((pl.col("cum_importance").max()*cutoff).alias("cutoff"))
                                .with_columns((pl.col("cutoff")>pl.col("cum_importance")).alias("keep")))
        elif type(cutoff) == int:
            agg_importance = (agg_importance
                                .with_columns((pl.arange(0, agg_importance.height) < cutoff).alias("keep")))
        else:
            raise ValueError(f"only float and int allowed for cutoff, got {cutoff} with type {type(cutoff)}")

    
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(x=agg_importance[x].cast(str), 
            height=agg_importance["importance"],
            color="grey"
            )

    ax2 = ax1.twinx()
    ax2.plot(agg_importance[x].cast(str), 
             agg_importance["cum_importance"],
             color="black")


    # manage axis labels
    ax1.set_xlim((-0.75, len(agg_importance[x])-0.25))
    ax1.set_ylim((0, np.max(ax1.set_ylim())*1.1))
    ax1.set_ylabel("Importance")
    ax1.set_xlabel(x_axis)
    ax2.set_ylabel("Cumulative importance")
    if len(ax1.get_xticklabels()) > 50:
        ax1.set_xticks([])
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax1.set_xticklabels(labels = ax1.get_xticklabels(), rotation=90)
    
    # get y-scale and adjust
    max_abs = ax1.get_ylim()[1]
    if max_abs == 0:
        base = 0
    else:
        base = int(np.floor(np.log10(max_abs)))
    if base > 3:
        scale = 10**base
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val/scale:.2f}× $10^{{{base}}}$"))


    if cutoff:
        highlight = agg_importance.filter(pl.col("keep"))[-1]
        n_features = agg_importance.filter(pl.col("keep")).shape[0]
        importance = highlight["cum_importance"][0]
        ax2.axhline(y=importance, color="k", linestyle="--", alpha=0.25)
        ax2.axvline(x=highlight["feature"][0], color="k", linestyle="--", alpha=0.25)
        title_str = f"\nCutoff = {cutoff}; Features = {n_features}; Importance = {round(importance, 2)}"
    else:
        title_str = ""

    # add manuel legend
    grey = mpatches.Patch(color='grey', label='Excitation wv')
    black = Line2D([], [], color='black', label='Cumulative')

    ax1.legend(handles=[grey, black],
               loc="upper left",
               frameon=False)

    # save figure
    fig.suptitle("Cumulative feature importance of " + x_axis + "s" + title_str)
    fig.tight_layout()
    fig.savefig(f"{figure_name}.png", dpi=600)
    fig.savefig(f"{figure_name}.pdf")
    fig.savefig(f"{figure_name}.svg")
    plt.clf()
# ~~~~~~~~~~~~~~~~~~~~~~~~~ CUMULATIVE FEATURE IMPORTANCE ~~~~~~~~~~~~~~~~~~~~~~~~~ #




# ~~~~~~~~~~~~~~~~~~~~~~~~~ Wavelengths ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def plot_fluorescent_response(
                                rf_dat: pl.DataFrame,
                                excitation_wavelength: int = 345, 
                                color_by: str = "strainID",
                                palette: Optional[str] = None,
                                figure_name: str = "fluorescent_response",
                                figsize: Tuple[int, int] = (8, 5),
                             ) -> None:
    """
    Plots the emission spectra for a specific excitation wavelength.

    This function filters the wide-format processed data for all features 
    associated with a single excitation wavelength, unpivots them, and 
    visualizes the fluorescent response as line plots. This is useful for 
    comparing spectral fingerprints across different strains or conditions.

    Args:
        rf_dat: The processed wide-format Polars DataFrame.
        excitation_wavelength: The specific excitation wavelength (nm) to 
            extract from the features (e.g., 345).
        color_by: The column name used to group and color the lines 
            (e.g., "strainID", "medium").
        palette: Optional color palette for the plot (Seaborn palette name 
            or list of colors).
        figure_name: Base filename for saving results (.png, .pdf, .svg).
        figsize: A tuple defining width and height in inches.

    Returns:
        None. Saves the high-resolution spectral plot to disk.
    """
                        
    # show the responses
    feature_set = [feature for feature in rf_dat.columns if (feature.startswith("wv")) or (feature in ["od"])]
    metadata    = [col for col in rf_dat.columns if col not in feature_set]

    # melt the dataframe and subset the relevant features
    ex_plot_data = (rf_dat
                        .unpivot(on=feature_set, index=metadata, variable_name="ex_em", value_name="Fluorescent response")

                        # filter excitations
                        .filter(pl.col("ex_em").str.contains(f"wv{excitation_wavelength}"))

                        #split ex_em fields
                        .with_columns(pl.col("ex_em").str.split_exact(".", n=1).struct.rename_fields(["ex", "em"]).alias("fields"))
                        .unnest("fields")

                        # adjust ex and em and types
                        .with_columns(pl.col("ex").str.replace("wv", "").cast(int))
                        .with_columns(pl.col("em").str.replace(f"_fft.....", "").cast(int))

                        #add additional column for fft-transformed data
                        .with_columns(pl.col("ex_em").str.splitn("_", n=2).struct.rename_fields(["ex_em_", "transformation"]).alias("fft"))
                        .unnest("fft")
                        .drop("ex_em_")
                        .with_columns(pl.col("transformation").fill_null("un-transformed"))

                        .sort("em")
                        .to_pandas()
                        )

    # plotting
    wv_plot = sns.relplot(
                            data=ex_plot_data, 
                            x="em", 
                            y="Fluorescent response",
                            hue=color_by,
                            kind="line",
                            facet_kws=dict(sharey=False),
                            err_kws=dict(linewidth=0),
                            palette=palette,
                            height=figsize[1],
                            aspect=figsize[0]/figsize[1],
                            legend=False if len(ex_plot_data[color_by].unique().tolist()) > 20 else True,
                            )

    # adjust axis ranges and labels
    wv_plot.set(xlim=(ex_plot_data["em"].min(), ex_plot_data["em"].max()))
    wv_plot.set(xlabel='Emission wavelength', ylabel='Fluorescent response')

    wv_plot.figure.subplots_adjust(top=.9)
    wv_plot.figure.suptitle(f'Excitation with {excitation_wavelength} nm')

    wv_plot.savefig(f"{figure_name}_ex{excitation_wavelength}.png", dpi=600)
    wv_plot.savefig(f"{figure_name}_ex{excitation_wavelength}.pdf")
    wv_plot.savefig(f"{figure_name}_ex{excitation_wavelength}.svg")
    plt.clf()
# ~~~~~~~~~~~~~~~~~~~~~~~~~ WAVELENGTHS ~~~~~~~~~~~~~~~~~~~~~~~~~ #



# ~~~~~~~~~~~~~~~~~~~~~~~~~ PCA ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def plot_dimensional_reduction(
                                rf_dat: pl.DataFrame,
                                color_by: Union[bool, str] = False,
                                centroid_lines: bool = False,
                                method: str = "pca",
                                distance: str = "braycurtis",
                                kernel: str = "linear",
                                perplexity: Union[bool, int] = False,
                                scale: bool = True,
                                components: List[str] = ["PC1", "PC2"],
                                alpha: float = 0.5,
                                color_map: Union[str, dict] = "tab20",
                                n_jobs: int = -1,
                                figure_name: str = "dimensional_reduction",
                                figsize: Tuple[int, int] = (10, 10),
                              ) -> None:
    """
    Performs dimensionality reduction and visualizes clusters in 2D space.

    Supports PCA (Kernel), MDS, and t-SNE. This utility helps identify 
    natural groupings in the spectral data and assess how well different 
    strains or conditions are separated before training a model.

    Args:
        rf_dat: The processed wide-format Polars DataFrame.
        color_by: Column name to use for coloring data points (e.g., "strainID").
        centroid_lines: If True, draws lines connecting each point to its 
            group centroid, emphasizing cluster spread.
        method: Dimensionality reduction algorithm: 'pca', 'mds', or 'tsne'.
        distance: Metric for distance calculation (e.g., 'braycurtis', 'euclidean').
        kernel: Kernel for PCA (e.g., 'linear', 'rbf').
        perplexity: The perplexity parameter for t-SNE. If False, defaults to 
            sqrt(n_samples).
        scale: If True, applies StandardScaler to features before reduction.
        components: List of two component names to plot (e.g., ["PC1", "PC2"]).
        alpha: Transparency of the data points.
        color_map: Name of Matplotlib palette or a dictionary mapping labels to colors.
        n_jobs: Number of parallel jobs for distance and reduction calculations.
        figure_name: Base filename for saving results (.png, .pdf, .svg).
        figsize: A tuple defining width and height in inches.

    Returns:
        None. Saves the high-resolution reduction plot to disk.
    """

    # perform the split
    feature_df, metadata = _feature_metadata_split(rf_dat)

    # calucate distance
    X_scaled = feature_df.to_numpy().astype(np.float64)
    
    # perform scaling
    if scale:
        X_scaled = StandardScaler().fit_transform(X_scaled)

    if method == "pca":
        # perform dimensional reduction
        print(f"performing {method}...")
        dist = squareform(pdist(X_scaled, metric = distance))
        transformer = KernelPCA(kernel=kernel, 
                                n_jobs=n_jobs,
                                random_state=42)
        xTransformed = transformer.fit_transform(dist)

    if method == "mds":
        print(f"performing {method}...")
        dist = squareform(pdist(X_scaled, metric = distance))
        transformer = MDS(metric=True, 
                          dissimilarity="precomputed", 
                          n_jobs=n_jobs,
                          random_state=42)    
        xTransformed = transformer.fit_transform(dist)


    if method == "tsne":
        print(f"performing {method}...")
        transformer = TSNE(metric=distance, 
                           perplexity=int(np.sqrt(feature_df.shape[0])) if perplexity is False else perplexity,
                           n_jobs=n_jobs,
                           method="barnes_hut",
                           init="random",
                           random_state=42)
        xTransformed = transformer.fit_transform(X_scaled)
    
    # post process data and plot
    explained_variance = np.var(xTransformed, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    reduced_df = (pl.DataFrame(xTransformed)
                        .rename({f"column_{idx}": f"PC{idx+1}" for idx, _ in enumerate(explained_variance)})
                        .select(components))
    reduced_df = pl.concat([metadata, reduced_df], how="horizontal")

    # plot explained variance of the components
    variance_df = pl.DataFrame({'var':explained_variance_ratio, 'PC':[f"PC{pc+1}" for pc, _ in enumerate(explained_variance_ratio)]})


    # get variance
    pc1_str = variance_df.filter(pl.col("PC") == components[0])["PC"][0]
    pc2_str = variance_df.filter(pl.col("PC") == components[1])["PC"][0]
    var1 = round(variance_df.filter(pl.col("PC") == components[0])["var"][0]*100, 1)
    var2 = round(variance_df.filter(pl.col("PC") == components[1])["var"][0]*100, 1)



    # determine the color scheme
    if not color_by:
        kwargs= {
                "hue": None,
                "legend": False,
                "palette": None,
                }
    
    else:
        if type(color_map) == str:
            # get the colors
            grouped_df = (reduced_df
                            .group_by(color_by)
                            .agg(pl.col(components[0]).mean(), pl.col(components[1]).mean()))
            strains = reduced_df[color_by].unique()

            # create the dict
            colors = _get_color(color_map, len(strains), transparent=True)
            lut = {val: tuple(colors[idx]) for idx, val in enumerate(strains)}
        elif type(color_map) == dict:
            lut = color_map
        else:
            ValueError(f"Type of variable should be either str or dict; got {type(color_map)}")

        # get additional kwargs
        kwargs= {
                "hue": color_by,
                "legend": True,
                "palette": lut,
                }

    # generate plot here
    print(f"performing plotting...")
    fig, ax = plt.subplots(figsize=figsize)

    if (color_by is not None) and (centroid_lines):

        # add lines
        for row in reduced_df.iter_rows(named=True):
                color = lut[row[color_by]]
                grouped_subset = (grouped_df
                                  .filter(pl.col(color_by) == row[color_by])
                                  .select(color_by, components[0], components[1]))
                line_df = pl.concat([pl.DataFrame(row).select(grouped_subset.columns), grouped_subset], how="vertical_relaxed")
                g = sns.lineplot(data = line_df, 
                                 x=components[0], 
                                 y=components[1], 
                                 markersize=5,
                                 color=color, 
                                 zorder=2, 
                                 ax = ax, 
                                 alpha=alpha*0.5)
    
    # plot the points
    g = sns.scatterplot(data=reduced_df.to_pandas(),
                        x=components[0], 
                        y=components[1],
                        alpha = alpha,
                        s = 80,
                        ax = ax,
                        edgecolor='none',
                        linewidth=0,
                        zorder=2,
                        **kwargs)
    
    if (color_by is not None):
        # add a legend
        sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))

    ax.spines[["top", "right"]].set_visible(False)
    g.set_xlabel(f'{pc1_str}' + (f' {var1}%' if method != "tsne" else ""))
    g.set_ylabel(f'{pc2_str}' + (f' {var2}%' if method != "tsne" else ""))
    fig.suptitle(f'Dimensional reduction using: {method} & {distance}', fontsize=14)
    fig.tight_layout()

    # save to drive
    print(f"writing plots...")
    fig.savefig(f"{figure_name}.pdf")
    fig.savefig(f"{figure_name}.png", dpi=600)
    fig.savefig(f"{figure_name}.svg")
    plt.clf()
# ~~~~~~~~~~~~~~~~~~~~~~~~~ PCA ~~~~~~~~~~~~~~~~~~~~~~~~~ #


# ~~~~~~~~~~~~~~~~~~~~~~~~~ CLUSTERMAP ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def plot_heatmap(
                    rf_dat: pl.DataFrame,
                    annotation: Union[bool, str] = False,
                    color_map: str = "tab20",
                    figure_name: str = "heatmap",
                    figsize: Tuple[int, int] = (16, 10),
                ) -> None:
    """
    Generates a hierarchical clustered heatmap of features across all samples.

    The rows (samples) and columns (wavelengths) are reordered using hierarchical 
    clustering to bring similar profiles together. If an annotation is provided, 
    a color-coded sidebar identifies the taxonomic groups for each row.

    Args:
        rf_dat: The processed wide-format Polars DataFrame.
        annotation: Column name to use for row-wise color labeling (e.g., "family").
            If False, no color bar is added.
        color_map: Name of the Matplotlib palette for the annotation sidebar.
        figure_name: Base filename for saving results (.png, .pdf, .svg).
        figsize: A tuple defining width and height in inches.

    Returns:
        None. Saves the high-resolution clustermap to disk.
    """

    # perform the split
    feature_df, metadata = _feature_metadata_split(rf_dat)
    feature_df = feature_df.to_pandas()

    # handle color by
    if annotation:
        taxonmic_classes = metadata[annotation].unique()
        colors = _get_color(color_map=color_map, n_taxonomic_classes=len(taxonmic_classes), transparent=False)
        color_df = (pl.DataFrame({"cls": taxonmic_classes, annotation: [color for color in colors]})
                            .to_pandas()
                            .set_index("cls"))[annotation]
        feature_df.index = metadata[annotation].to_list()
    else:
        color_df = None

    # genrate the plot
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clstmp = sns.clustermap(feature_df, 
                                cmap="vlag", 
                                center = 0, 
                                row_colors=color_df,
                                yticklabels=False,
                                figsize=figsize)

    # additional aesthetics
    #clstmp.figure.set_size_inches(16, 10)

    if annotation:
        # plot legend
        lut = {idx: color for idx, color in zip(color_df.index, color_df)}
        handles = [Patch(facecolor=lut[name]) for name in lut]
        clstmp.figure.subplots_adjust(right=0.85, left=0)
        clstmp.figure.legend(handles, lut, title=annotation, bbox_to_anchor=(1, 0.5), bbox_transform=plt.gcf().transFigure, loc='center right')
    

    # adjust colormap position
    clstmp.ax_cbar.set_position((0.025, 0.3, 0.01, 0.4)) # tuple of (left, bottom, width, height), optional

    # write the figures
    plt.savefig(f"{figure_name}.pdf")
    plt.savefig(f"{figure_name}.png", dpi=600)
    plt.savefig(f"{figure_name}.svg")
    plt.clf()
# ~~~~~~~~~~~~~~~~~~~~~~~~~ CLUSTERMAP ~~~~~~~~~~~~~~~~~~~~~~~~~ #

def plot_reisolation(reiso, 
                     orderDf=None, 
                     figName=None):
    pass
