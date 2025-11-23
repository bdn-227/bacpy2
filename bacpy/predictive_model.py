# import libraries
from abc import ABC
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
import numpy as np
import joblib
import polars as pl
from itertools import product
import warnings
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# load bacpy modules
from bacpy.taxonomy import taxonomy_dict


class BaseClassifier(ABC):
    def __init__(self):
        pass

    # train method
    def train(self, train_set, predict=None):
        """
        function to train the model
        train_set   pl.DataFrame    obtained using bacpy.preprocess_data and/or bacpy.train_test_split
        """

        # get the set of features & labels
        labels = ["kingdom", "phylum", "class", "order", "family", "genus", "strainID"]
        feature_set = [feature for feature in train_set.columns if (feature.startswith("wv") or feature.startswith("od"))]
        label_set   = [label for label in labels if label in train_set.columns]

        # subset the restepctive columns if only predicting specific label
        if predict is not None:
            label_set = [label for label in label_set if label in predict]

        # save features
        self.features = feature_set
        self.labels = label_set

        # splitting dataframe into training and testing
        train_x = train_set.select(feature_set)
        train_y = train_set.select(label_set)

        # reformat labels to prevent warning
        shape = train_y.shape[1]

        # perform label encoding of model is xgboost
        if self.model_type in ["neural_net", "multi_xgboost"]:
            train_y_ls = []
            self.le = {}
            for label in self.labels:
                self.le[label] = LabelEncoder()
                self.le[label].fit(train_y[label])
                train_y_ls.append(self.le[label].transform(train_y[label]))
            train_y = pl.DataFrame({label: dat for label, dat in zip(self.labels, train_y_ls)})
        
        # because all are multioutput now
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.fit(train_x.to_pandas(), train_y.to_pandas())

        # train the model
        if shape == 1:
            self.taxonomic_classes = {level: class_ for level, class_ in zip(self.labels, [self.classes_])}
        else:
            self.taxonomic_classes = {level: class_ for level, class_ in zip(self.labels, self.classes_)}


    def predict_strains(self, 
                        test_set):
        """
        function to predict the taxonomy of bacterial spectra
        test_set            pl.DataFrame    a dataframe containing feautres (generated with bacpy.preprocess_data)
        probability         bool            default: False; uses class probability and a look-up table for more precise predictions at the cost of computing speed
        predict_all_ranks   bool            default: False; if True, use class probability to predict all taxonomic ranks that were present in training data
        
        returns:            pl.DataFrame    containing predictions
        """

        # add missing columns
        missing = [feature for feature in self.features if feature not in test_set.columns]
        test_set = test_set.with_columns(pl.lit(0).alias(col) for col in missing)

        # subset the relevant columns for predictions & extract metadata
        test_x      = test_set.select(self.features)
        metadata    = test_set.select(pl.exclude(self.features)).select(pl.exclude(self.labels))

        if self.model_type in ["multi_catboost"]:
            pred_d = {label: pred.ravel() for label, pred in zip(self.labels, [est.predict(test_x.to_numpy()) for est in self.estimators_])}
            pred_df = pl.DataFrame(pred_d)

        else:
            pred = self.predict(test_x.to_pandas())
            pred_df = pl.DataFrame(pred)
            pred_df.columns = self.labels
        
        # convert cats
        if self.model_type in ["neural_net", "multi_xgboost"]:
            for label in self.labels:
                transformed = self.le[label].inverse_transform(pred_df[label])
                pred_df = pred_df.drop(label).with_columns(pl.Series(name=label, values=transformed))

        return pl.concat([metadata, pred_df], how="horizontal")



    def evaluate(self, 
                 validation_set, 
                 metric="cm", # cm | stats | both
                 average="weighted",
                 ):
        """
        function to perform evaluation of the model
        validation_set      pl.DataFrame    validation dataset obtained using bacpy.preprocess_data, must contain the same taxonomic labels as training data
        probability         bool            using probability-predictions and look-up for evaluation
        predict_all_ranks   bool            predict all possible taxonomic ranks or just confident ones
        metric              str             either: cm | stats | both determines which statistics should be returned
        """

        # subset the relevant columns for predictions & extract metadata
        validation_set = validation_set.filter(pl.col(col).is_not_null() for col in self.labels)
        validation_x = validation_set.select(self.features)
        truth = validation_set.select(self.labels)

        # perform the predictions
        pred = self.predict_strains(validation_x)
        

        # nulls in predictions probably need to be replaced
        pred = pred.with_columns(pl.all().fill_null("n.p."))


        # conditional branch for return
        if metric == "cm":
            return self._get_cm(truth, pred)
        if metric == "both":
            return self._get_stats(truth, pred, average), self._get_cm(truth, pred)
        else:
            return self._get_stats(truth, pred, average)
    

    def _get_cm(self, truth, pred):

        # store results in a dictionary
        cm_dict = {}

        # iterate through taxonomic levels
        for label in np.intersect1d(truth.columns, pred.columns):

            # extract the labels
            labels = self.taxonomic_classes[label]

            # convert for xgboost
            if "xgboost" in self.model_type:
                labels = self.le[label].inverse_transform(labels)

            # get the calsses used for predictions
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cm = pl.DataFrame(confusion_matrix(y_true=truth[label], y_pred=pred[label], labels=labels))
            cm.columns = labels
            cm = cm.with_columns(pl.Series(labels).alias(label))
            cm_dict[label] = cm
        
        return cm_dict
    

    def _get_stats(self, truth, pred, average):
        stats_ls = []
        for label in np.intersect1d(truth.columns, pred.columns):
            labels = np.union1d(truth[label], pred[label])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                accuracy = accuracy_score(truth[label], pred[label])
                mcc = matthews_corrcoef(truth[label],pred[label])
                f1 = f1_score(truth[label], pred[label], average = average, labels=labels)
            if average is None:
                stats_res = pl.DataFrame({"taxonomic_level": label,
                                          "f1": f1,
                                          "taxonomy": labels
                                        })
            else:
                stats_res = pl.DataFrame({"taxonomic_level": label, 
                                          "accuracy": accuracy, 
                                          "f1": f1, 
                                          "mcc": mcc})
            # format final table
            stats_ls.append(stats_res)

        stats_df = (pl.concat(stats_ls, how="diagonal_relaxed")
                        .sort(pl.col("taxonomic_level").cast(pl.Enum(self.labels)), descending=True)
                   )
        return stats_df



    def get_features_importances(self, as_matrix=False):
        """
        function to retrieve the feature importances of the model
        as_matrix  bool    default: False; return feature importances as matrix instead of table
        """

        if self.model_type == "multi_svc":
            ValueError(f"MULTI-OUTPUT SVM DOES NOT ALLOW FOR DETERMINATION OF FEATURE IMPORTANCES..")

        # extract the information
        importances = self.feature_importances_
        features    = self.features

        # create dataframe and perform some formatting
        importance_df = (pl.DataFrame({"importance": importances})

                                # add features
                                .with_columns(pl.Series(features).alias("feature"))
                                .with_columns(pl.col("feature").str.replace("od", "0.0"))
                                .sort("importance", descending=True)

                                # reformat to obtain raw feature
                                .with_columns(pl.col("feature").str.replace("wv", ""))
                                .with_columns(pl.col("feature").str.split_exact(".", 1).struct.rename_fields(["ex", "em"]).alias("wv"))
                                .unnest("wv")
                                .with_columns(pl.col("em").str.splitn("_", 2).struct.rename_fields(["em", "fft"]))
                                .unnest("em")
                                .with_columns( (pl.col("ex") + "." + pl.col("em")).alias("ex_em") )
                                .group_by("ex_em", "ex", "em").agg(pl.col("importance").sum())

                                # fix od-values and types
                                .with_columns( pl.col("ex").cast(int), pl.col("em").cast(int) )
                                .sort("importance", descending=True)
                                )

        # return data
        if as_matrix:

            # convert feature importances into a matrix --> for plottung and stuff
            importance_mat = (importance_df
                                    .select("ex", "em", "importance")
                                    .group_by("ex", "em")
                                    .agg(pl.col("importance").sum())
                                    .sort("ex", descending=True)
                                    .pivot(on="em", index="ex", values="importance", sort_columns=True)
                                    )
            
            return importance_mat
        else:
            return importance_df


    def extract_useful_features(self, n=50):
        """
        function to extract n useful features for successive measurements. As 
        features are generally normalized across a range of values to estimate a baseline,
        this function also patches these informative features so that they can be used
        n       int     default=50  number of informative features 
        returns         dataframe containing n features + some addition ones required for normalization
        """

        # first, get the feature importances
        importance_df = self.get_features_importances(as_matrix=False)
        
        # determine steps of the spectra taken
        em_values = importance_df["em"].unique().sort().to_numpy()
        em_values = np.array(list(product(em_values, em_values)))
        em_values = em_values[em_values[:,0] != em_values[:,1]]
        step_size = np.abs(em_values[:,0] - em_values[:,1]).min()

        # aggregate and subset
        importance_df = (importance_df.select("ex", "em", "importance")
                                    .group_by("ex", "em")
                                    .agg(pl.col("importance").sum())
                                    .sort("importance", descending=True)
                                    .select("ex", "em"))[0:n]
        

        # now determine set size and add values to path
        patched_features = []
        for ex in importance_df["ex"].unique():

            # subset
            ex_subset = importance_df.filter(pl.col("ex") == ex)
            
            # condition branch to fix stuff
            if ex_subset.shape[0] < 3:
                new_min = ex_subset["em"].min() - step_size
                new_max = ex_subset["em"].max() + step_size
                em_range = np.arange(new_min, new_max+1, step_size)
                ex_range = np.full(em_range.shape[0], ex)
                patched_features.append(pl.DataFrame({"ex": ex_range, "em": em_range}))
                
            else:
                patched_features.append(ex_subset)

        # put together and sort
        patched_df = (pl.concat(patched_features, how="vertical_relaxed")
                                .sort("ex", "em")
                                .with_columns( ("wv" + pl.col("ex").cast(str) + "." + pl.col("em").cast(str) ).alias("ex_em") )
                                .unique()
                                .join(self.get_features_importances().select("ex", "em", "importance").unique(), how="left", on =["ex", "em"]))
        
        return patched_df



class classifier_randomForest(RandomForestClassifier, BaseClassifier):
    def __init__(self, 
                 n_jobs = 1, 
                 n_estimators = 500,
                 criterion = "gini", 
                 max_features = "sqrt",
                 max_depth = None,
                 min_samples_split = 2,
                 min_samples_leaf = 1,
                 bootstrap = True
                 ):
        self.model_type = "tree"
        super().__init__(n_estimators=n_estimators, 
                         n_jobs=n_jobs, 
                         criterion=criterion,
                         max_features=max_features, 
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         bootstrap=bootstrap)



class classifier_extraTrees(ExtraTreesClassifier, BaseClassifier):
    def __init__(self, 
                 n_jobs = 1, 
                 n_estimators = 500,
                 criterion = "gini", 
                 max_features = "sqrt",
                 max_depth = None,
                 min_samples_split = 2,
                 min_samples_leaf = 1,
                 bootstrap = True,
                 ):
        self.model_type = "tree"
        super().__init__(n_estimators=n_estimators, 
                         n_jobs=n_jobs, 
                         max_features=max_features, 
                         criterion = criterion, 
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         bootstrap=bootstrap,
                         )



class classifier_svm(MultiOutputClassifier, BaseClassifier):
    def __init__(self, 
                 n_jobs = 1,
                 kernel = "rbf",
                 C = 1,
                 gamma = "scale",
                 class_weight = None,
                 ):
        self.n_jobs = n_jobs
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.model_type = "multi_svm"
        super().__init__(estimator=SVC(kernel=self.kernel,
                                       C=self.C,
                                       gamma=self.gamma,
                                       class_weight=self.class_weight,
                                       probability=True),
                         n_jobs=self.n_jobs)


class classifier_xgboost(MultiOutputClassifier, BaseClassifier):
    def __init__(self, 
                 n_jobs = -1,
                 max_depth = None,
                 min_child_weight = None,
                 subsample = None,
                 colsample_bytree = None,
                 gamma = None,
                 reg_lambda = None,
                 reg_alpha = None,
                 learning_rate = None,
                 n_estimators = 100,
                 ):
        self.n_jobs = n_jobs
        self.nthread = n_jobs
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.model_type = "multi_xgboost"
        super().__init__(estimator=XGBClassifier(n_estimators=n_estimators, 
                                                 max_depth=max_depth, 
                                                 min_child_weight=min_child_weight,
                                                 subsample=subsample,
                                                 colsample_bytree=colsample_bytree,
                                                 gamma=gamma,
                                                 reg_lambda=reg_lambda,
                                                 reg_alpha=reg_alpha,
                                                 learning_rate=learning_rate,
                                                 nthread=n_jobs,
                                                 n_jobs=n_jobs,
                                                 ),
                         n_jobs=1
                         )


class classifier_catboost(MultiOutputClassifier, BaseClassifier):
    def __init__(self, 
                 iterations = None,
                 learning_rate = None,
                 early_stopping_rounds = None,
                 n_jobs=1,
                 **kwargs
                 ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.allow_writing_files = False
        self.save_snapshot = False
        self.model_type = "multi_catboost"
        super().__init__(estimator=CatBoostClassifier(iterations=iterations,
                                                      learning_rate=learning_rate,
                                                      early_stopping_rounds=early_stopping_rounds,
                                                      thread_count=n_jobs,
                                                      allow_writing_files=False,
                                                      save_snapshot=False,
                                                      **kwargs
                                                      ),
                         n_jobs=1)


class classifier_lightgbm(MultiOutputClassifier, BaseClassifier):
    def __init__(self,
                 num_leaves=31,
                 n_estimators=100,
                 max_depth=-1,
                 boosting_type = "gbdt",
                 learning_rate=0.1,
                 n_jobs=None,
                 **kwargs
                 ):
        self.n_estimators = n_estimators
        self.boosting_type = boosting_type
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.model_type = "multi_lightbgm"
        super().__init__(estimator=LGBMClassifier(n_estimators=n_estimators,
                                                  num_leaves=num_leaves,
                                                  max_depth=max_depth,
                                                  boosting_type=boosting_type,
                                                  learning_rate = learning_rate,
                                                  n_jobs = n_jobs,
                                                  **kwargs
                                                  ),
                         n_jobs=1)


class classifier_neuralnet(MLPClassifier, BaseClassifier):
    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation = "relu",
                 alpha=0.0001,
                 solver="adam",
                 learning_rate_init=0.001,
                 max_iter=200,
                 early_stopping=True,
                 n_jobs=None,
                 ):
        self.model_type = "neural_net"
        self.n_jobs = n_jobs
        super().__init__(
                         hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,
                         alpha=alpha,
                         solver=solver,
                         learning_rate_init=learning_rate_init,
                         max_iter=max_iter,
                         early_stopping=early_stopping,
                         )



def train_test_split(
                     rf_dat, 
                     test_frac = 0.2, 
                     split_by = False, 
                     equal = False, 
                     ):
    """
    function to split a parsed and processed dataset (rf_dat) into
    two datasets, one for validation and one for testing
    rf_dat      pl.DataFrame    processed dataset to be splitted
    test_frac   float           default: 0.2; fraction of the dataset that will be used for testing
    split_by    bool | str      default: False; if False, splitting occurs randomly, if str,
                                dataset separation will be according to split_by
                                i.e. split_by="strainID" means that a given strain is either testing or training, but not both
                                used to predict novel strains, does not work with equal
    equal       bool | str      default: False; if str, equal instances of each str will be in testing set, does not work with split_by
    """

    # general information for splitting
    rows = rf_dat.shape[0]
    index = np.arange(rf_dat.shape[0])


    if split_by and equal:
        raise ValueError(f"cannot use split_by: {split_by} and equal: {equal} together")


    if not split_by and not equal:
        
        # split the data solely according to the test_frac
        test_rows_n = int(test_frac * rows)
        test_rows = np.random.choice(index, test_rows_n, replace=False)

        # now translate into index-mask
        test_index = np.isin(index, test_rows)


    if split_by:

        # branch to split the data according to label set
        labels = rf_dat.filter( ~pl.col(split_by).is_in(["medium", "blank"]) )[split_by].unique()

        # now sample split into train & test
        test_classes_n = max(np.round(labels.shape[0] * test_frac).astype(int), 1)
        test_classes = np.random.choice(labels, test_classes_n, replace=False)

        # now translate these indices into a index-mask
        test_index = rf_dat[split_by].is_in(test_classes)
    

    if equal:

        # here, a strain can be in both, but we want to have equal instances in test set
        labels = rf_dat[equal].unique()
        min_class_instance = rf_dat.group_by(equal).len()["len"].min()

        # now, determine the size of the test set
        rows_per_class = min(round((rows * test_frac) / len(labels)), min_class_instance)
        test_rows = np.array([np.random.choice(index[rf_dat[equal] == label], rows_per_class, replace=False) for label in labels]).reshape(-1)
        
        # now translate into index-mask
        test_index = np.isin(index, test_rows)


    # and now, just split the dataset according to the masks
    test_set  = rf_dat.filter( test_index )
    train_set = rf_dat.filter( ~test_index )

    return train_set, test_set

