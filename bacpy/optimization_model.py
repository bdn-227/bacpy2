

# ~~~~~~~ LIBRARIES ~~~~~~~ #
from .predictive_model import classifier_catboost, classifier_extraTrees, classifier_lightgbm, classifier_neuralnet, classifier_randomForest, classifier_svm, classifier_xgboost
from .utils import save_model
import polars as pl
import numpy as np
import random
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
from sklearn.model_selection import StratifiedKFold
import importlib
from scipy.stats import randint
from ast import literal_eval

# 2. define search spaces for models
param_grid_rf = {
    "n_estimators": randint(100, 1501),
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2", None, 0.2, 0.5, 0.8],
    "max_depth": [None] + list(randint(3, 101).rvs(10).tolist()),  
    "min_samples_split": randint(2, 51),
    "min_samples_leaf": randint(1, 21),
    "bootstrap": [True, False],
}

param_grid_svc = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": np.logspace(-3, 3, 7),
    "gamma": list(np.logspace(-4, 1, 6)) + ["scale", "auto"],
    "class_weight": [None, "balanced"]
}

param_grid_xgb = {
    "max_depth": [3, 5, 7, 9],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.5, 1],
    "reg_lambda": [0.5, 1, 5],
    "reg_alpha": [0, 0.1, 1],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 500, 1000]
}

param_grid_nn = {
    "hidden_layer_sizes": [(50,), (100,), (100,50), (100,100,50)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "solver": ["adam", "lbfgs"],
    "learning_rate_init": [1e-3, 1e-4, 1e-5],
    "max_iter": [300, 500, 800],
}

param_grid_nn = {
    "hidden_layer_sizes": [(50,), (100,), (100,50), (100,100,50)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "solver": ["adam", "lbfgs"],
    "learning_rate_init": [1e-3, 1e-4, 1e-5],
    "max_iter": [300, 500, 800],
}

param_grid_lightbgm = {
  "learning_rate": [0.01, 0.03, 0.05, 0.1],
  "n_estimators": [100, 200, 500, 1000],
  "num_leaves": [31, 63, 127],
  "min_child_samples": [10, 30, 100],
  "feature_fraction": [0.6, 0.8, 1.0],
  "bagging_fraction": [0.6, 0.8, 1.0]
}

param_grid_catboost = {
    "learning_rate": [None, 0.01, 0.03, 0.05, 0.1],
    "depth": [None, 4, 6, 8],
    "l2_leaf_reg": [None, 1, 3, 10],
    "bagging_temperature": [None, 0, 0.5, 1.0],
    "iterations": [None, 100, 200, 500],
    "early_stopping_rounds": [None, 20, 50, 100],
}



def sample_params(param_distributions):
    sampled = {}
    for key, val in param_distributions.items():
        if isinstance(val, list) or isinstance(val, np.ndarray):
            sampled[key] = random.choice(val)
        elif isinstance(val, rv_continuous_frozen) or isinstance(val, rv_discrete_frozen):
            sampled[key] = int(val.rvs(1)[0])
        else:
            raise ValueError(f"Unknown type for {key}: {type(val)}")
    return sampled


def optimize_model_platereader(rf_dat, 
                               on="strainID", 
                               by="f1",
                               cv_folds=5,
                               parameters_per_model=10,
                               n_jobs= -1,
                               filename=None,
                               ):

    # define cross-validation folds
    x = rf_dat.select(pl.selectors.starts_with("wv")).to_numpy()
    y = rf_dat.select(on).to_numpy()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    folds = [(train_idx, test_idx) for train_idx, test_idx in cv.split(np.zeros(len(y)), y)]

    # sample hyperparameters
    param_ls_rf = [sample_params(param_grid_rf) for _ in range(parameters_per_model)]
    param_ls_svc = [sample_params(param_grid_svc) for _ in range(parameters_per_model)]
    param_ls_xgb = [sample_params(param_grid_xgb) for _ in range(parameters_per_model)]
    param_ls_nn = [sample_params(param_grid_nn) for _ in range(parameters_per_model)]
    param_ls_lightgbm = [sample_params(param_grid_lightbgm) for _ in range(parameters_per_model)]
    param_ls_catboost = [sample_params(param_grid_catboost) for _ in range(parameters_per_model)]

    model_d = {
               #classifier_xgboost: param_ls_xgb, 
               classifier_randomForest: param_ls_rf, 
               classifier_extraTrees: param_ls_rf, 
               classifier_svm: param_ls_svc, 
               classifier_neuralnet: param_ls_nn,
               classifier_lightgbm: param_ls_lightgbm,
               classifier_catboost: param_ls_catboost,
            }

    total_tests = cv_folds*parameters_per_model*len(model_d)
    c = 1
    res_ls = []
    for model in model_d.keys():
        for idx in range(parameters_per_model):
            kwargs = model_d[model][idx]
            for i, (train_idx, test_idx) in enumerate(folds, start=1):
                try:
                    print(f"TESTING MODEL: {str(model)} - {c}/{total_tests} - {i}th fold - {round(100*c/total_tests, 2)}%")
                    print(f"KWARGS: {str(kwargs)}")

                    # get the data
                    train_df = rf_dat[train_idx].select(pl.selectors.starts_with("wv") | pl.selectors.by_name(on))
                    test_df  = rf_dat[test_idx].select(pl.selectors.starts_with("wv") | pl.selectors.by_name(on))

                    # train and evaluate model
                    m = model(n_jobs=n_jobs, **kwargs)
                    m.train(train_df)
                    stats_res = (m.evaluate(test_df.with_columns(pl.col(on).cast(str)), metric="stats")
                                    .with_columns(pl.lit(str(kwargs)).alias("kwargs"))
                                    .with_columns(pl.lit(str(model)).alias("model")))
                except ValueError as e:
                    print(f"ERROR ENCOUNTERED FOR {model} - {str(kwargs)} - {e}")
                    # taxonomic_level	accuracy	f1	mcc	kwargs	model
                    stats_res = pl.DataFrame({"taxonomic_level": on, "accuracy": 0.0, "f1": 0.0, "mcc": 0.0, "kwargs": str(kwargs), "model": str(model)})
                res_ls.append(stats_res)
                c+=1
    optimization = (pl.concat(res_ls)
                        .with_columns(pl.col("model").str.split(".").list[-1].str.strip_chars_end("'>").alias("model_str"))
                        .sort(by, descending=True))
    if filename is not None:
        optimization.write_csv(f"{filename}.tsv", separator="\t")
    return optimization





def get_optimized_model(optimization_result, n_jobs=-1, filename=None):
    best_model_type = optimization_result["model"][0]
    best_model_params = literal_eval(optimization_result["kwargs"][0])
    full_class_path = best_model_type.strip("<class '").strip("'>")
    module_name, class_name = full_class_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
        BestModelClass = getattr(module, class_name)
        best_model_instance = BestModelClass(n_jobs=n_jobs, **best_model_params)
        print(f"Successfully loaded class: {BestModelClass}")
        print(f"Instantiated model: {best_model_instance}")
        if filename is not None:
            save_model(best_model_instance, filename)
        return best_model_instance
    except (ImportError, AttributeError) as e:
        print(f"Error loading class '{full_class_path}': {e}")