
# some adjustment to make the package compatible with hpc
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

# function to preprocess the data
from bacpy.preprocess_tecan import preprocess_platereader
from bacpy.preprocess_flowjo import preprocess_cytometry

# parser for icontrol
from .file_parser_icontrol import parse_file_icontrol
from .dataset_parser_icontrol import parse_dataset_icontrol

# parser for evoware
from .file_parser_evoware import parse_file_evoware
from .dataset_parser_evoware import parse_dataset_evoware

# flowjo
from .file_parser_flowjo import parse_file_flowjo, get_column_types_flowjo
from .dataset_parser_flowjo import parse_dataset_flowjo

# taxonomy look-up
from .taxonomy import taxonomy_dict, taxonomy_df

# train a model
from .predictive_model import classifier_randomForest, classifier_extraTrees, train_test_split, classifier_xgboost, classifier_neuralnet, classifier_svm, classifier_catboost, classifier_lightgbm

# misc
from .utils import abs_path, load_model, save_model

# plotting
from .plotting import plot_confusion_matrix, plot_stats, plot_correlogram, plot_feature_importances, plot_fluorescent_response, plot_dimensional_reduction, plot_heatmap, plot_reisolation, plot_cumulative_importance

# these ones still need to fixed
from .predict_reisolation import predictReisolation

# scripts to create tecan scripts
from .create_tecan_protocols import create_tecan_scripts

# optimization
from .optimization_preprossing import optimize_preprocess_platereader

