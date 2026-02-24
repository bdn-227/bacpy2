import polars as pl
import numpy as np
from typing import Union, List, Optional



def print_text(log_text, print_logs=False):
    if print_logs:
        print(log_text)


def print_size(parsed_data, print_logs, pl):
    if print_logs:
        data_size = parsed_data.shape
        n_samples = parsed_data.with_columns((pl.col("filename") + "_" + pl.col("well")).alias("location"))["location"].unique().shape[0]
        n_features = parsed_data.select("ex", "em").unique().shape[0]
        print(f"dataset shape: {data_size}\nn_samples:      {n_samples}\nn_features:     {n_features}\n")


def normalize(fluroscence, normalize_by, indexcols, print_func, logging_func, pl, np):
    # perform normalization
    if normalize_by:
        print_func(f"Normalizing data using {normalize_by}:")
        if "scale" in normalize_by:
            if normalize_by == "scale":
                agg_cols =  indexcols
            elif normalize_by == "scale_ex":
                agg_cols =  indexcols + ["ex"]
            norm_data = fluroscence.group_by(agg_cols).agg(pl.col("response").mean().alias("mean"), pl.col("response").std().alias("std"))
            fluroscence = (fluroscence
                                .join(norm_data, how="left", on=agg_cols)
                                .with_columns( ( (pl.col("response") - pl.col("mean")) / pl.col("std") ).alias("response") )
                                .drop("mean", "std")
                                )
        if "auc" in normalize_by:
            from scipy.integrate import simpson
            def calculate_auc_safe(s):
                temp = s.struct.unnest()
                if len(temp) < 2:
                    return 0.0
                return simpson(y=temp["response"], x=temp["em"])
            agg_cols = indexcols + ["ex"]
            norm_data = (fluroscence
                            .group_by(agg_cols)
                            .agg(pl.struct(["em", "response"])
                                 .sort_by("em")
                                 .map_elements(calculate_auc_safe, return_dtype=pl.Float64)
                                 .alias("auc_simpson")))
            fluroscence = (fluroscence
                            .join(norm_data, how="left", on=agg_cols)
                            .with_columns((pl.col("response")/pl.col("auc_simpson")).alias("response"))
                            .drop("auc_simpson"))
        logging_func(fluroscence)
    return fluroscence


def check_medium(flu_medium, fluroscence):
    return flu_medium.select(['device', 'date', 'filename']).unique().shape[0]==fluroscence.select(['device', 'date', 'filename']).unique().shape[0]


def pivot(fluroscence, indexcols, print_func, pl):
    print_func(f"Pivoting dataframe..")
    rf_dat = (fluroscence
                    .with_columns(("wv" + pl.col("ex").cast(str) + "." + pl.col("em").cast(str)).alias("ex_em"))
                    .pivot(index=indexcols, on="ex_em", values="response")
                    )
    print_func(f"dataset shape: {rf_dat.shape}\n")
    return rf_dat


def impute(rf_dat, indexcols, impute_strategy, print_func, pl):
    # Step 1: Clean invalid values and compute row-wise mean
    numeric_cols = [col for col in rf_dat.columns if col not in indexcols]
    rf_dat = rf_dat.with_columns([pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)for c in numeric_cols])
    if impute_strategy:
        # convert features to numpy
        print_func(f"Imputing features..")
        # determine the inpute val
        if impute_strategy == "mean":
            inpute_df = rf_dat.unpivot(index=indexcols).filter(pl.col("value").is_not_null()).group_by(indexcols).agg(pl.col("value").mean().alias("impute_val"))
        elif impute_strategy == "min":
            inpute_df = rf_dat.unpivot(index=indexcols).filter(pl.col("value").is_not_null()).group_by(indexcols).agg(pl.col("value").min().alias("impute_val"))
        elif impute_strategy == "max":
            inpute_df = rf_dat.unpivot(index=indexcols).filter(pl.col("value").is_not_null()).group_by(indexcols).agg(pl.col("value").max().alias("impute_val"))
        elif impute_strategy == "median":
            inpute_df = rf_dat.unpivot(index=indexcols).filter(pl.col("value").is_not_null()).group_by(indexcols).agg(pl.col("value").median().alias("impute_val"))
        elif impute_strategy == "high_val":
            inpute_df = rf_dat.unpivot(index=indexcols).filter(pl.col("value").is_not_null()).group_by(indexcols).agg(pl.lit(1e6).alias("impute_val"))
        else:
            inpute_df = rf_dat.unpivot(index=indexcols).filter(pl.col("value").is_not_null()).group_by(indexcols).agg(pl.lit(0).alias("impute_val"))
        # add the impute val in
        rf_dat = (rf_dat
                    .join(inpute_df, how="left", on=indexcols)
                    .with_columns( pl.when(pl.col(col).is_null()).then("impute_val").otherwise(col).alias(col) for col in numeric_cols )
                    .drop("impute_val")
                    )
    else:
        # just remove features (i.e. cols)
        keep = rf_dat.select(pl.col(col).is_null().any() for col in numeric_cols).transpose(include_header=True, header_name="ex_em").filter(~pl.col("column_0"))["ex_em"].to_list()
        rf_dat = rf_dat.select(indexcols+keep)
    return rf_dat


# preprocess function
def preprocess_platereader(
                            parsed_data: pl.DataFrame,
                            feature_list: Union[List[str], np.ndarray, bool] = False,
                            filter_common_features: bool = False,
                            include_negative: bool = False,
                            include_over: bool = False,
                            scale_by_medium: Union[bool, str] = False,
                            arcsinh: bool = False,
                            normalize_by: Union[str, bool] = "scale_ex",
                            impute_strategy: Union[str, bool] = "mean",
                            batches: Union[List[str], bool] = ["device", "date"],
                            add_od: bool = False,
                            outlier_threshold: Union[bool, int] = False,
                            outlier_column: str = "strainID",
                            multicore: bool = True,
                            print_logs: bool = True,
                            return_after: Union[bool, str] = False,
                          ) -> pl.DataFrame:
    """
    Converts raw spectral plate reader data into a clean feature matrix for machine learning.

    This pipeline handles the transition from a 'long' observation format to a 'wide' 
    feature format. It includes specialized steps for biological data such as 
    background subtraction (using measured or estimated medium scattering), batch effect 
    correction via the ComBat algorithm, and PCA-based outlier filtering.

    Args:
        parsed_data: Input Polars DataFrame. Required columns: ["filename", "date", 
            "device", "well", "measurement_mode", "ex", "em", "ex_em", "response"].
        feature_list: Specific 'ex_em' features to keep. Accepts a list of strings, 
            a NumPy array, or False to keep all features.
        filter_common_features: If True, only keeps features present in every 
            filename (intersection).
        include_negative: If False, filters out negative fluorescence responses.
        include_over: If False, removes 'OVER' values (saturated detector signals).
        scale_by_medium: Background removal strategy.
            - "measured": Uses wells labeled 'medium'.
            - "estimated": Uses the lowest 5% quantile of absorbance as medium.
            - False: No background subtraction.
        arcsinh: If True, applies inverse hyperbolic sine transformation.
        normalize_by: Normalization method ("scale", "scale_ex", "auc", or False).
        impute_strategy: Logic for filling missing values ("mean", "min", etc.).
        batches: Columns to use for ComBat batch effect correction. Accepts a list 
            of column names or False to skip batch correction.
        add_od: If True, appends optical density (absorbance) data to the features.
        outlier_threshold: Z-score threshold for PCA-based outlier removal per strain. 
            Accepts an integer threshold or False to skip.
        outlier_column: The column used to group data for outlier detection.
        multicore: If False, restricts Polars to single-threaded execution.
        print_logs: If True, prints verbose progress and data shape updates.
        return_after: Debugging flag to return data early at specific steps 
            (e.g., "pivot", "impute", "batch").

    Returns:
        A wide-format Polars DataFrame ready for classification, or a dictionary 
        of dataframes if returning prematurely for debugging.
    """
    # handling imports
    if not multicore:
        import os
        os.environ["POLARS_MAX_THREADS"] = "1"
    from sklearn.decomposition import PCA
    from scipy.stats import zscore
    import warnings
    import polars as pl
    import numpy as np
    from .combat import combat
    from functools import partial, reduce


    # get the printing fucntion
    print_func = partial(print_text, print_logs=print_logs)
    logging_func = partial(print_size, print_logs=print_logs, pl=pl)


    # adjust types
    parsed_data = parsed_data.with_columns(pl.col("device").cast(str))


    # removal of "strains to remove"
    if "strainID" in parsed_data.columns:
        print_func("Removing strains to be removed..")
        parsed_data = (parsed_data
                            .filter(pl.col("strainID") != "remove")
                            .with_columns(pl.col("strainID").fill_null("unknown"))
                            )
        logging_func(parsed_data)


    # filter for features that are desired
    if feature_list is not False:
        print_func("subsetting features..")
        parsed_data = parsed_data.filter(pl.col("ex_em").is_in(feature_list))
        logging_func(parsed_data)


    if filter_common_features:
        # filter for common features, i.e. the intersection of all individual measurements
        print_func("subsetting common features..")
        
        # Get all unique values of "ex_em" for each "filename"
        grouped_df = parsed_data.group_by("filename").agg(pl.col("ex_em").unique().sort().alias("unique_ex_em"))

        # Find the intersection of "unique_ex_em" across all "filenames"
        intersection = reduce(np.intersect1d, grouped_df["unique_ex_em"])
        parsed_data = parsed_data.filter(pl.col("ex_em").is_in(intersection))
        logging_func(parsed_data)


    # split into absorbance and fluorescence data
    print_func(f"Splitting Fluorescent & Absorbance spectra..")
    absorbance   = parsed_data.filter(pl.col("measurement_mode") == "Absorbance")
    fluroscence  = parsed_data.filter(pl.col("measurement_mode") == "Fluorescence Bottom Reading")
    abs_medium   = parsed_data.filter(pl.col("measurement_mode") == "Absorbance").filter(pl.col("strainID")=="medium")
    flu_medium   = parsed_data.filter(pl.col("measurement_mode") == "Fluorescence Bottom Reading").filter(pl.col("strainID")=="medium")
    logging_func(parsed_data)


    # from now on, separate meta data from actual data & perform whole processing metadata free
    indexcols = ["device", "date", "filename", "well"]
    metadatacols = ['strainID', 'assay', 'medium', 'notes', "ratio", 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    fluroscence = fluroscence.select(indexcols + ["ex", "em", "response"])
    metadata =  parsed_data.select(indexcols + list(np.intersect1d(metadatacols, parsed_data.columns))).unique()


    # remove negative fluorescent values
    if not include_negative:
        print_func(f'Removing negative values..')
        fluroscence = fluroscence.filter(pl.col("response") >= 0)
        logging_func(fluroscence)


    # remove OVER values
    if not include_over:
        print_func(f'Removing OVER values..')
        fluroscence = fluroscence.filter(pl.col("response") != 1e10)
        logging_func(fluroscence)


    # premature returning
    if return_after == "remove_over":
        return {"fluroscence": fluroscence, "absorbance": absorbance, "flu_medium": flu_medium, "abs_medium": abs_medium}


    if scale_by_medium:
        """
        remove the background (i.e. device artifacts and background scattering)
        """
        print_func(f'Removing background..')
        medium_norm_cols = ['device', 'date', 'filename']
        scale_by_medium = "measured" if scale_by_medium == True else scale_by_medium
        if (scale_by_medium == "measured"):
            if check_medium(flu_medium, fluroscence):
                adjust_df = flu_medium
            else:
                print_func("not enough medium scans for background substraction. switching to estimated")
                scale_by_medium = "estimated"
        if scale_by_medium == "estimated":
            thresholds = (absorbance
                            .filter(pl.col("strainID")!="blank")
                            .group_by(medium_norm_cols)
                            .agg(pl.col("response").sort().quantile(0.05, "nearest").alias("threshold"))
                            .sort("filename", "threshold")
                        )
            norm_ids = (absorbance
                            .join(thresholds, on=medium_norm_cols)
                            .filter(pl.col("response") <= pl.col("threshold"))
                            .select(medium_norm_cols+["well"])
                            .unique()
                        )
            adjust_df =  (norm_ids.join(fluroscence, how="left", on=medium_norm_cols+["well"]))
        if scale_by_medium not in ["measured", "estimated"]:
            raise ValueError(f"set scale_by_medium = 'measured' or 'estimated' or False - {scale_by_medium}")
        norm_vals =  (adjust_df
                            .group_by(medium_norm_cols + ["ex", "em"])
                            .agg(pl.col("response").mean().alias("medium_mean"), 
                                 pl.col("response").std().alias("medium_std"))
                            )
        fluroscence = (fluroscence
                        .join(norm_vals, how="left", on=medium_norm_cols + ["ex", "em"])
                        .with_columns( ((pl.col("response")-pl.col("medium_mean"))/pl.col("medium_std")).alias("response") )
                        .drop("medium_mean", "medium_std")
                        .filter( ~(pl.col("response").is_nan() | pl.col("response").is_null() | pl.col("response").is_infinite()) )
                      )
        logging_func(fluroscence)


    # potential log transform
    if arcsinh:
        fluroscence = fluroscence.with_columns((pl.col("response").arcsinh()).alias("response"))
    if return_after == "arcsinh":
        return fluroscence


    # normalize data
    fluroscence = normalize(fluroscence, normalize_by, indexcols, print_func, logging_func, pl, np)


    # make df wider
    rf_dat = pivot(fluroscence, indexcols, print_func, pl)


    # premature returning
    if return_after == "pivot":
        return rf_dat


    # imputation of incomplete features
    rf_dat = impute(rf_dat, indexcols, impute_strategy, print_func, pl)
    if return_after == "impute":
        return rf_dat

    # remove batch effects from data
    if batches:
        for batch in batches:

            # print logs
            print_func(f"removing batch caused by: {batch}")
            numeric_cols = [col for col in rf_dat.columns if col not in indexcols]

            # test of more than two batches, if not, skip
            if len(rf_dat[batch].unique()) < 2:
                continue

            # extract features & metadata
            batch_dat = rf_dat.select(numeric_cols)
            meta_data = rf_dat.select(indexcols)
            batch_col = rf_dat[batch].cast(str)

            # remove columns with zero only
            batch_dat = batch_dat[:, (batch_dat.sum() != 0).to_numpy().reshape(-1)]

            # correct batch effects
            dBatchEffect = pl.DataFrame(combat(data = batch_dat.to_pandas().transpose(), batch = batch_col.to_pandas(), printFuct=print_func).T)

            # put data together
            rf_dat = pl.concat([meta_data, dBatchEffect], how="horizontal")
        print_func(f"dataset shape: {rf_dat.shape}\n")
    if return_after == "batch":
        return rf_dat


    # add optical density information to the dataframe
    if add_od:
        print_func(f'adding optical density information..')
        absorbance_wide = (absorbance
                                .with_columns(pl.col("ex_em").str.replace("wv", "od", literal=True))
                                .group_by(indexcols+["ex_em"])
                                .agg(pl.col("response").mean())
                                .pivot(index=indexcols, values="response", on="ex_em")
                                .with_columns(pl.col("device").cast(str))
                            )
        abs_cols = np.intersect1d(absorbance["ex_em"].unique().str.replace("wv", "od", literal=True), absorbance_wide.columns)
        rf_dat = rf_dat.join(absorbance_wide, how="left", on=indexcols, validate="1:1")
        drop = rf_dat.select(pl.col(col).is_null().any() for col in abs_cols).transpose(include_header=True, header_name="ex_em").filter(pl.col("column_0"))["ex_em"].to_list()
        rf_dat = rf_dat.drop(drop)
        print_func(f"dataset shape: {rf_dat.shape}\n")
    if return_after == "add_od":
        return rf_dat


    # add metadata
    rf_dat = rf_dat.join(metadata, how="left", on=indexcols)
    if return_after == "add_metadata":
        return rf_dat

    # remove outliers to get a crisp training-dataset
    if outlier_threshold:
        feature_cols = np.intersect1d(rf_dat.columns, parsed_data["ex_em"].unique())

        # error logging
        if type(outlier_threshold) != int:
            ValueError(f"VARIABLE `outlier_threshold` IS EXPECTED TO BE `int`; BUT GOT {type(outlier_threshold)}")
        
        # perform the actual correction
        if (outlier_column in rf_dat.columns) and (len(feature_cols)>2):
            filtered_list = []

            # logging
            print_func(f"Removing outliers using PCA..")

            # outlier detection can only be performed on training data --> only on bacteria
            for strain in rf_dat[outlier_column].unique():

                # subset strains specific data and perform pca
                strain_subset = rf_dat.filter(pl.col(outlier_column) == strain)

                # scipt for medium, unknown and blanks
                if (strain in ["unknown", pl.Null, None, np.nan, "background", "medium", "blank"]) or (strain_subset.shape[0]<2):
                    filtered_list.append(strain_subset)

                else:
                    # dimensional reduction
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        pca = PCA(n_components = 2)
                        transformed = pca.fit_transform(strain_subset.select(feature_cols))
                        transformed = zscore(transformed, axis=0)
                    keep = (np.abs(transformed) > outlier_threshold).sum(axis=1) == 0
                    strain_subset = strain_subset.filter(keep)
                    filtered_list.append(strain_subset)
            rf_dat = pl.concat(filtered_list, how="vertical")
            print_func(f"dataset shape: {rf_dat.shape}\n")

    return rf_dat
