
# modules
import os
import re
import joblib


def abs_path(dir):
    wd = os.getcwd()
    return re.sub("biodata.*", dir, wd)


def save_model(model, filename = 'model'):
    joblib.dump(model, filename)
    print("saved model as: " + filename)


def load_model(filename):
    return joblib.load(filename)



