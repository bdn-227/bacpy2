
# ~~~~~~~ LIBRARIES ~~~~~~~ #
import polars as pl
from itertools import product, repeat
from multiprocessing import get_context
from .preprocess_tecan import preprocess_platereader
from .predictive_model import train_test_split, classifier_randomForest, BaseClassifier
from time import strftime, gmtime
from random import shuffle
from sklearn import clone
from ast import literal_eval
from typing import Union, List, Optional, Dict


# ~~~~~~~~ MODEL OPTIMIZATION ~~~~~~~~ #
def test_kwargs_platereader(kwargs, idx, kwargs_len, parsed_culture_collections, test_frac, equal, split_by, model, on, repeats):
    """
    function to test a combination of kwargs, implemented for compatility of polars with multiprocessing

    """
    print(f"\ntime: {strftime("%Y-%m-%d %H:%M:%S", gmtime())}\ncurrent percent: {round(100*idx*repeats/kwargs_len, 2)}%\ntesting:\n{kwargs}\n\n")
    stats_ls = []
    try:
        rf_dat = preprocess_platereader(parsed_culture_collections, **kwargs)
        for _ in range(repeats):
            train_set, validation_set = train_test_split(rf_dat, 
                                                         test_frac = test_frac,
                                                         equal=equal,
                                                         split_by=split_by)
            clf = clone(model)
            clf.train(train_set, predict=on)
            stats_res = clf.evaluate(validation_set, metric="stats")
            stats_ls.append(stats_res)
    except Exception as e:
        print(f"\n\nfailed: {str(kwargs)}\n{e}\n\n")
    stats_ls = [e for e in stats_ls if e is not None]
    if len(stats_ls)>0:
        return pl.concat(stats_ls, how="vertical_relaxed").with_columns(pl.lit(str(kwargs)).alias("kwargs"))


def optimize_preprocess_platereader(
                                        parsed: pl.DataFrame,
                                        n_kwargs: int = -1,
                                        outlier_column: str = "strainID",
                                        test_frac: float = 0.2,
                                        equal: Union[bool, str] = "strainID",
                                        split_by: Union[bool, str] = False,
                                        model: BaseClassifier = classifier_randomForest(n_jobs=1, n_estimators=100),
                                        on: str = "strainID",
                                        print_logs: bool = False,
                                        n_jobs: int = -1,
                                        repeats: int = 3,
                                        filename: Optional[str] = None,
                                   ) -> pl.DataFrame:
    """
    Performs a parameter grid search to find optimal preprocessing settings for plate reader data.

    This function generates a grid of preprocessing parameters, 
    subsets them if requested, and evaluates each combination in parallel. Each test 
    involves preprocessing the data and training a model to evaluate accuracy or 
    performance, averaged over multiple repeats to ensure robustness.

    Args:
        parsed: The raw parsed Polars DataFrame to be optimized.
        n_kwargs: Number of random parameter combinations to test. If -1, performs 
            a full grid search of the entire parameter space.
        outlier_column: The column to use for grouping during outlier detection.
        test_frac: The fraction of data to reserve for testing in each iteration.
        equal: The column name to ensure balanced representation during data splitting.
        split_by: Optional column name to define hard splits (e.g., by 'date' or 'plate').
        model: The classifier instance (inheriting from BaseClassifier) used for evaluation.
        on: The taxonomic rank or label column the model should predict.
        print_logs: Internal logging toggle for the preprocessing sub-functions.
        n_jobs: Number of CPU cores for parallel execution. Defaults to -1 (all cores).
        repeats: Number of times to repeat each parameter test with different 
            data splits to calculate average performance.
        filename: Optional path to save the resulting optimization table as a .tsv file.

    Returns:
        A Polars DataFrame containing the tested parameter sets and their 
        associated performance metrics.
    """
    
    # handle multi core processing
    model.__setattr__("n_jobs", 1)
    multicore = False

    kwargspace = {"feature_list": [False],
                  "filter_common_features": [True, False],
                  "include_negative": [True, False],
                  "include_over": [True, False],
                  "scale_by_medium": [False, "measured", "estimated"],
                  "normalize_by": [False, "scale", "scale_ex", "auc"],
                  "impute_strategy": ["mean", False],
                  "arcsinh": [True, False],
                  "batches": [False, ["device", "date"]],
                  "add_od": [False],
                  "outlier_threshold": [False, 1, 2, 3],
                  "outlier_column": [outlier_column],
                  "multicore": [multicore],
                  "print_logs": [print_logs],
                  "return_after": [False],} 

    # put together and shuffle order
    keys, values = zip(*kwargspace.items())
    kwargs_ls = [dict(zip(keys, v)) for v in product(*values)]
    shuffle(kwargs_ls)

    # now subset the final list
    if n_kwargs != -1:
        kwargs_ls = kwargs_ls[0:n_kwargs]

    # add repeats
    args_n = len(kwargs_ls)
    idx = range(args_n)
    total_tests = len(kwargs_ls)*repeats
    print(f"NUMBER OF COMBINATIONS TESTED: {args_n}")
    print(f"AVERAGING ACROSS {repeats} REPEATS")
    print(f"TOTAL ITERATIONS: {total_tests}")
    with get_context("spawn").Pool(processes=n_jobs) as pool:
        test_ls = pool.starmap(test_kwargs_platereader, zip(kwargs_ls, 
                                                        idx, 
                                                        repeat(total_tests), 
                                                        repeat(parsed), 
                                                        repeat(test_frac), 
                                                        repeat(equal), 
                                                        repeat(split_by), 
                                                        repeat(model),
                                                        repeat(on),
                                                        repeat(repeats)))

    # concat and write
    test_ls = [e for e in test_ls if e is not None]
    test_df = pl.concat(test_ls)
    if filename is not None:
        test_df.write_csv(f"{filename}.tsv", separator="\t")
    return test_df


def preprocess_optimized(
                            parsed: pl.DataFrame, 
                            optimization_result: pl.DataFrame
                        ) -> pl.DataFrame:
    """
    Applies the best-performing parameters from an optimization run to a dataset.

    This function acts as the bridge between the optimization search and the 
    final production pipeline. It extracts the dictionary of 
    parameters from the top-performing row of an optimization result and 
    executes the `preprocess_platereader` pipeline.

    Args:
        parsed: The raw parsed data to be processed.
        optimization_result: A Polars DataFrame containing the results from 
            `optimize_preprocess_platereader`. The first row is expected 
            to contain the target configuration in the "kwargs" column.

    Returns:
        A wide-format Polars DataFrame (rf_dat) processed with optimized parameters, 
        ready for training or final classification.
    """
    kwargs = literal_eval(optimization_result["kwargs"][0])
    kwargs["print_logs"] = True
    rf_dat_optimized = preprocess_platereader(parsed, **kwargs)
    return rf_dat_optimized
