
# ~~~~~~~ LIBRARIES ~~~~~~~ #
import polars as pl
from itertools import product, repeat
from multiprocessing import get_context
import bacpy
from time import strftime, gmtime
from random import shuffle



# ~~~~~~~~ MODEL OPTIMIZATION ~~~~~~~~ #
def test_kwargs_platereader(kwargs, idx, kwargs_len, parsed_culture_collections, test_frac, equal, split_by, model_type):
    """
    function to test a combination of kwargs, implemented for compatility of polars with multiprocessing

    """
    
    # some neat plotting
    print(f"\ntime: {strftime("%Y-%m-%d %H:%M:%S", gmtime())}\ncurrent percent: {round(100*idx/kwargs_len, 2)}%\ntesting:\n{kwargs}\n\n")

    try:
        # process the data
        rf_dat = bacpy.preprocess_platereader(parsed_culture_collections, **kwargs)


        # perform a train/test split of the data
        train_set, validation_set = bacpy.train_test_split(rf_dat, 
                                                           test_frac = test_frac,
                                                           equal=equal,
                                                           split_by=split_by)


        # now create and train a model on subset
        try:
            model = model_type(n_jobs=1)
        except:
            model = model_type()
        model.train(train_set)
        stats_validation = model.evaluate(validation_set, metric="stats")

        # put data together into dict and write to disk
        return stats_validation.with_columns(pl.lit(str(kwargs)).alias("kwargs"))
   
    except Exception as e:
        print(f"\n\nfailed: {str(kwargs)}\n{e}\n\n")


def optimize_preprocess_platereader(parsed, 
                                    n_kwargs=-1, 
                                    test_frac=0.2,
                                    equal="strainID",
                                    split_by=False,
                                    model_type=bacpy.classifier_randomForest,
                                    n_jobs=-1,
                                    outlier_column = "strainID",
                                    repeats=2,
                                    filename=None,
                                    ):

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
                  "multicore": [False],
                  "print_logs": [False],
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
    kwargs_ls = kwargs_ls*repeats

    # using parallel processing to read the files
    idx = range(len(kwargs_ls))
    total_tests = len(kwargs_ls)
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
                                                            repeat(model_type)))

    # concat and write
    test_df = pl.concat(test_ls)
    if filename is not None:
        test_df.write_csv(f"{filename}.tsv", separator="\t")
    return test_df
