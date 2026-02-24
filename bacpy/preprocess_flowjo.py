from typing import Union, Optional
import polars as pl

def print_text(log_text, print_logs=False):
    if print_logs:
        print(log_text)


# preprocess function
def preprocess_cytometry(
                            events_df: pl.DataFrame,
                            arcsinh: bool = False,
                            normalize_by: Union[str, bool] = "scale_ex",
                            impute_strategy: Union[str, bool] = "mean",
                            multicore: bool = True,
                            print_logs: bool = True,
                        ) -> pl.DataFrame:
    """
    Executes a comprehensive preprocessing pipeline on cytometry event data.

    The function processes data per-file to maintain memory efficiency, involving 
    unpivoting wide-format feature data into a long-format 'response' column, 
    applying optional transformations and Z-score normalizations, and finally 
    re-pivoting and imputing missing values.

    Args:
        events_df: The input Polars DataFrame containing cytometry events and metadata.
        arcsinh: If True, applies an inverse hyperbolic sine transformation to 
            the 'response' values to handle high dynamic range and skewness.
        normalize_by: The Z-score normalization strategy:
            - "scale": Standardizes by filename and event.
            - "scale_ex": Extracts excitation from feature names and standardizes.
            - "scale_ex_em": Extracts both excitation and emission for grouping.
            - False: Skips normalization.
        impute_strategy: Logic for handling nulls after re-pivoting:
            - "mean"/"min"/"max": Based on horizontal statistics across features.
            - "high_val": Fills with 1e6.
            - "zero": Fills with 0.
            - False: Drops columns containing any null values.
        multicore: If False, restricts Polars to a single thread by setting 
            POLARS_MAX_THREADS to "1".
        print_logs: If True, utilizes a partial print function to output 
            step-wise progress percentages to the console.

    Returns:
        A wide-format Polars DataFrame with transformed and cleaned features.

    Note:
        This function performs an 'unpivot' (melt) followed by a 'pivot' operation 
        inside the file loop, which is standard for applying column-dependent 
        normalizations across disparate signal channels.
    """
    
    # handling imports
    if not multicore:
        import os
        os.environ["POLARS_MAX_THREADS"] = "1"
    import polars as pl
    from functools import partial
    from .file_parser_flowjo import get_column_types_flowjo
    print_func = partial(print_text, print_logs=print_logs)

    events_ls = []
    filenames = events_df["filename"].unique()
    for idx, filename in enumerate(filenames):

        # susbet for faster and more memory efficient processing
        print_func(f"SUBSETTING: {filename} - {round(100*idx/len(filenames), 2)}%")
        events_subset = events_df.filter(pl.col("filename")==filename)

        print_func(f"UN-PIVOTING: {filename} - {round(100*idx/len(filenames), 2)}%")
        feature_cols, metadata_cols = get_column_types_flowjo(events_subset)
        events_subset = events_subset.unpivot(index=metadata_cols, on=feature_cols, variable_name="feature", value_name="response")

        # arcsinh transformation to reduce skewness of the data
        if arcsinh:
            print_func(f"ARCSINH TRANSFORMING: {filename} - {round(100*idx/len(filenames), 2)}%")
            events_subset = events_subset.with_columns((pl.col("response").arcsinh()).alias("response"))

        # normalization of the data
        if normalize_by:
            print_func(f"NORMALIZING: {filename} - {round(100*idx/len(filenames), 2)}%")
            drop_cols = ["mean", "std"]
            if normalize_by == "scale":
                agg_cols = ["filename", "event"]
            elif normalize_by == "scale_ex":
                agg_cols = ["filename", "event", "ex"]
                drop_cols = drop_cols + ["ex"]
                events_subset = events_subset.with_columns(pl.col("feature").str.split("_").list[0].alias("ex"))
            elif normalize_by == "scale_ex_em":
                agg_cols = ["filename", "event", "ex", "em"]
                drop_cols = drop_cols + ["ex", "em"]
                events_subset = (events_subset
                                .with_columns(pl.col("feature").str.split("_").list[0].alias("ex"))
                                .with_columns( pl.when(pl.col("feature").str.contains("-", literal=True))
                                                .then(pl.col("feature").str.split("-").list[-1]) 
                                                .otherwise(pl.lit("all"))
                                                .alias("em")))
            else:
                ValueError(f"INCOMPATIBLE NORMALIZATION CHOSEN: {normalize_by}")
            norm_df = (events_subset
                        .group_by(agg_cols)
                        .agg(pl.col("response").mean().alias("mean"), 
                            pl.col("response").std().alias("std")))
            events_subset = (events_subset
                                .join(norm_df, how="left", on=agg_cols)
                                .with_columns(((pl.col("response")-pl.col("mean"))/pl.col("std")).alias("response"))
                                .drop(drop_cols)
                                .with_columns( pl.when(pl.col("response").is_infinite() | pl.col("response").is_null() | pl.col("response").is_nan()).then(None).otherwise("response").alias("response") ))
        
        # unpivot now
        print_func(f"PIVOTING: {filename} - {round(100*idx/len(filenames), 2)}%")
        events_subset = events_subset.pivot(on="feature", values="response")
        events_ls.append(events_subset)
        
    # putting it together
    print_func(f"CONCATENATING ALL")
    events_df = pl.concat(events_ls, how="diagonal_relaxed")
    feature_cols, metadata_cols = get_column_types_flowjo(events_df)

    # inpute
    if impute_strategy:
        impute_strategy = "mean" if impute_strategy == True else impute_strategy
        print_func(f"IMPUTING ALL")
        if impute_strategy == "mean":
            inpute_df = events_df.group_by(metadata_cols).agg(pl.mean_horizontal(pl.col(feature_cols)).mean().alias("impute_val"))
        elif impute_strategy == "min":
            inpute_df = events_df.group_by(metadata_cols).agg(pl.min_horizontal(pl.col(feature_cols)).mean().alias("impute_val"))
        elif impute_strategy == "max":
            inpute_df = events_df.group_by(metadata_cols).agg(pl.max_horizontal(pl.col(feature_cols)).mean().alias("impute_val"))
        elif impute_strategy == "high_val":
            inpute_df = events_df.select(metadata_cols).unique().with_columns(pl.lit(1e6).alias("impute_val"))
        elif impute_strategy == "zero":
            inpute_df = events_df.select(metadata_cols).unique().with_columns(pl.lit(0).alias("zero"))
        else:
            ValueError(f"INCORRECT VALUE FOR ´impute_strategy´. got {impute_strategy}. options are: mean, min, max, median, zero, True (mean) and False")
        events_df = (events_df
                            .join(inpute_df, how="left", on=metadata_cols)
                            .with_columns( pl.when(pl.col(col).is_null()).then("impute_val").otherwise(col).alias(col) for col in feature_cols )
                            .drop("impute_val"))
    else:
        print_func(f"REMOVING NANs")
        keep = events_df.select(pl.col(col).is_null().any() for col in feature_cols).transpose(include_header=True, header_name="ex_em").filter(~pl.col("column_0"))["ex_em"].to_list()
        events_df = events_df.select(metadata_cols+keep)
    
    return events_df
