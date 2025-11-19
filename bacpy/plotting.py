

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
import warnings
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler


# load bacpy modules
from bacpy.taxonomy import taxonomy_df


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


def plot_confusion_matrix(confusion_matrices, 
                          taxonomic_level  = "strainID", 
                          annotation       = "family", 
                          color_map        = "tab20", 
                          figure_name      = "confusion_matrix",
                          figsize=(16,10),
                          ):
    """
    function to plot confusion matrices
    ----------
    confusion_matrices : dict
        dictionary containing confusion matrices for all taxonomic level the model is trained on
    taxonomic_level : str
        default = "strainID"
        which taxonomic level is the confusion matrix, False plots all
    annotation : str
        default = "family"
        annotation must a column in confusion matrix, used to plot an extra column and row annotating the data with colors
    color_map : str
        default = "tab20"
        which color map to use for annotations
    figure_name : str
        name of the files to be written
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
def plot_stats(stats,
               figure_name="stats_plot",
               kind = "bar",
               figsize=(8, 5),
               ):
    """
    function to plots stats of the model (accuracy, f1-score and mcc)
    stats : pl.DataFrame
        dataframe containing statistics obtained from model.evaluate(metric="stats" | metric="both")
    figure_name : str
        default = "stats_plot"
        name of the file to be written
    kind : str
        default = "bar"
        kind of plot, possible options are box, violin and bar
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
def plot_correlogram(rf_dat, 
                     variable="features",
                     aggregate=True,
                     figure_name="correlogram",
                     figsize=(20, 20),
                     ):
    """ 
    generate a cluster map of the data Parameters
    ----------
    rf_dat : pl.DataFrame
        a pl.DataFrame with the preprocessed data
        obtained with the processData function
    variable: str
        features | any tax levels
    aggregate: bool
        default = True
        if true, means per group (variable) will be calculated and plotted
    figure_name: str
        a string that us used as filename for the resulting 
        figure
    Returns
    -------
    clusterMap : seaborn.matrix.ClusterGrid
        A matrix showing correlation among features
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

def plot_feature_importances(feature_importances, 
                             figure_name="feature_heatmap", 
                             highlight=False,
                             figsize=(12, 10)):
    """
    function to plot heatmap of feature importances
    features_wide       pl.DataFrame        obtained by model.get_features_importances(as_matrix=True)
    figure_name         str                 default="feature_heatmap", name of the file to be written
    highlight           pl.DataFrame        default=False, pl.DataFrame containing a column called "ex_em". these features will be highlighted in plot
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
def plot_cumulative_importance(feature_importances,
                               aggregate=True,
                               cutoff=False,
                               figure_name="cumulative_importance",
                               figsize=(8,5),
                               ):
    
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
def plot_fluorescent_response(rf_dat,
                              excitation_wavelength=345, 
                              color_by="strainID",
                              palette=None,
                              figure_name="fluorescent_response",
                              figsize=(8,5),
                              ):
    """
    funcion to plot fluorescent responses
    rf_dat                  pl.DataFrame            processed data ran through bacpy.preprocess_data
    excitation_wavelength   int                     default=345, which excitation wavelength to plot
    color_by                str                     default="strainID", color responses according to
    figure_name             str                     default="fluorescent_response", name of file to be written  
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
                            palette=palette,
                            height=figsize[1],
                            aspect=figsize[0]/figsize[1],
                            legend=False if len(ex_plot_data[color_by].unique().tolist()) > 20 else True,
                            )

    # adjust axis ranges and labels
    wv_plot.set(xlim=(ex_plot_data["em"].min(), ex_plot_data["em"].max()))
    wv_plot.set(xlabel='Emission wavelength', ylabel='Fluorescent response')

    wv_plot.figure.subplots_adjust(top=.9)
    wv_plot.figure.suptitle(f'Fluorescence responses upon excitation with {excitation_wavelength}nm')

    wv_plot.savefig(f"{figure_name}_ex{excitation_wavelength}.png", dpi=600)
    wv_plot.savefig(f"{figure_name}_ex{excitation_wavelength}.pdf")
    wv_plot.savefig(f"{figure_name}_ex{excitation_wavelength}.svg")
    plt.clf()
# ~~~~~~~~~~~~~~~~~~~~~~~~~ WAVELENGTHS ~~~~~~~~~~~~~~~~~~~~~~~~~ #



# ~~~~~~~~~~~~~~~~~~~~~~~~~ PCA ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def plot_dimensional_reduction(
                               rf_dat,
                               color_by        = False,
                               centroid_lines  = False,
                               method          = "pca",
                               distance        = "braycurtis",
                               kernel          = "linear",
                               perplexity      = False,
                               scale           = True,
                               components      = ["PC1", "PC2"],
                               color_map       = "tab20",
                               n_jobs          = -1,
                               figure_name     = "dimensional_reduction",
                               figsize         = (10, 10),
                               ):
    """
    function to perform dimensional reduction of processed features
    rf_dat          pl.DataFrame            processed data ran through bacpy.preprocess_data
    color_by        bool | str              default=False, color points according to feature in metadata
    centroid_line   bool | str              default=False, calculate & plot centroids for differently colored data-points
    method          str                     default="pca", which method to use, options are: pca | mds | tnse
    distance        str                     default="braycurtis", method to use to calculate distance
    kernel          str                     default="linear" kernel to use for pca
    components      list                    default=["PC1", "PC2"], which components to plot
    color_map       str                     default="tab20", which color map to use for color_by
    n_jobs          int                     default=-1, how many jobs to compute dimensional reduction
    figure_name     str                     default="dimensional_reduction", name of figure written to drive
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
        # get the colors
        grouped_df = (reduced_df
                        .group_by(color_by)
                        .agg(pl.col(components[0]).mean(), pl.col(components[1]).mean()))
        strains = reduced_df[color_by].unique()

        # create the dict
        colors = _get_color(color_map, len(strains), transparent=True)
        lut = {val: tuple(colors[idx]) for idx, val in enumerate(strains)}

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
                                 alpha=0.4)
    
    # plot the points
    g = sns.scatterplot(data=reduced_df.to_pandas(),
                        x=components[0], 
                        y=components[1],
                        alpha = 0.8,
                        s = 80,
                        ax = ax,
                        edgecolor=None,
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
def plot_heatmap(rf_dat,
                 annotation = False,
                 color_map = "tab20",
                 figure_name = "heatmap",
                 figsize=(16, 10),
                 ):
    
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
