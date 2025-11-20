
# load imports
from bacpy.predictive_model import classifier_randomForest, train_test_split
from bacpy.preprocess_tecan import preprocess_platereader
from bacpy.plotting import plot_confusion_matrix
from scipy.stats import hmean
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ~~~~~~~~~~~~~~~~~~~~~~~~~ SYNCOM SPECIFIC MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def plotMetrics(df):
    meltedMetrics = df.melt(id_vars = ["tax"])
    measurements = meltedMetrics.groupby("tax") ["tax"].count().unique()[0] / 2
    meltedMetrics = meltedMetrics.rename(columns={"value": "Score", "variable": "Metric", "tax": "Taxonomic level"})

    palette = "tab10"
    #palette = "Set3"

    if measurements < 3:
        g, ax = plt.subplots()
        ax = sns.lineplot(data=meltedMetrics, 
                        x = "Taxonomic level", 
                        y = "Score", 
                        hue = "Metric",
                        palette = palette,
                        markers=True, 
                        style="Metric")
    else:
        g = sns.catplot(data=meltedMetrics, 
                        x = "Taxonomic level", 
                        y = "Score", 
                        hue = "Metric", 
                        kind="box",
                        palette = palette,
                        aspect=1.2)
    g.tight_layout()
    return g


def predictSynComSpecific(trainSetMedium, testSetMedium, syncom, splitSize, gaussian, model, metricsPlot, probability=False, taxonomy=False):

    #print(f"predicting: {syncom}")

    # prepare syncom specific data
    idx = (testSetMedium["syncom"] == syncom) | (testSetMedium["syncom"] == "medium") # is on both conditions
    testSetMedium = testSetMedium.loc[idx, :]

    # subset train set
    if syncom == "at":
        idx = (trainSetMedium["Sample_ID"].str.startswith("Root")) | (trainSetMedium["Sample_ID"].str.startswith("medium")) | (trainSetMedium["Sample_ID"] == "LjNodule218") # is on both conditions
    if syncom == "lj":
        idx = (trainSetMedium["Sample_ID"].str.startswith("Lj")) | (trainSetMedium["Sample_ID"].str.startswith("medium"))
    
    # add medium to test and training set
    trainSetMedium = trainSetMedium.loc[list(idx),:]
    xTrain, xTest, yTrain, yTest, labelSet = train_test_split(trainSetMedium, splitSize = splitSize, equal = "Sample_ID", splitBy=False)
    model.train(xTrain, yTrain, gaussian)

    # perform evaluation of the model
    metrics, cmDict = model.evaluate(xTest, yTest, returnBoth=True, taxonomy=taxonomy, probability=probability)

    if metricsPlot:
        metricsPlot = "" if metricsPlot == True else metricsPlot
        plot = plotMetrics(metrics)
        plot.savefig(f'metrics_{metricsPlot}_{syncom}.pdf')
        plot.savefig(f'metrics_{metricsPlot}_{syncom}.png')

        # and confusion matrix
        confusionMatrix = plot_confusion_matrix(cmDict, taxLevel = "Sample_ID")
        confusionMatrix.savefig(f'confusionMatrix_{metricsPlot}_{syncom}.pdf')
        confusionMatrix.savefig(f'confusionMatrix_{metricsPlot}_{syncom}.png')

    # extend the metrics with medium predictions
    medium = testSetMedium.loc[testSetMedium["Sample_ID"] == "medium",:]
    mediumMetrics = model.evaluate(medium[model.features], medium[model.labelSet], taxonomy=taxonomy, probability=probability)
    metricsVec = np.hstack([list(mediumMetrics["f1"]), list(metrics["f1"])])


    # calculate the harmonic mean of training data and labeled testing data (medium scans) F1 scores
    #totalModelScore = np.mean(metricsVec)
    totalModelScore = hmean(metricsVec)
    
    # extract meta data
    testSetMedium = testSetMedium.loc[testSetMedium["Sample_ID"] != "medium",:]
    metaCols = testSetMedium.columns
    features = np.hstack([model.features, model.labelSet])
    metaCols = metaCols[~metaCols.isin(features)]
    metaData = testSetMedium.loc[:, metaCols]
    index = metaData.index
    metaData = metaData.reset_index(drop=True)

    # make predictions and add meta data
    testSetMedium = model.predictStrains(testSetMedium, taxonomy=taxonomy, probability=probability).reset_index(drop=True)
    
    if type(probability) == str:
        cols = model.labelSet
        cols.append(probability)
        testSetMedium = testSetMedium.loc[:,cols]
    else:
        testSetMedium = testSetMedium.loc[:,model.labelSet]
    
    testSetMedium = pd.concat([metaData, testSetMedium], axis=1)
    testSetMedium.index = index
    
    return totalModelScore, testSetMedium
# ~~~~~~~~~~~~~~~~~~~~~~~~~ SYNCOM SPECIFIC MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~ #


# ~~~~~~~~~~~~~~~~~~~~~~~~~ PREDICT REISOLATED BACTERIA ~~~~~~~~~~~~~~~~~~~~~~~~~ #
def predictReisolation(
                       trainSet, 
                       testSet, 
                       mapping=False, 
                       filename=False, 
                       return_metrics=False, 
                       model=classifier_randomForest(), 
                       gaussian=False,
                       metricsPlot = False,
                       probability=None,
                       taxonomy=False,
                       ):

    # add mapping information
    if type(mapping) == str:
        mapping = pd.read_excel(mapping).astype(str)


    # add syncom information to dataframe
    if type(mapping) == pd.DataFrame:
        testSet["line"] = testSet["Filename"].str.split("_").str[1]
        testSet["generation"] = testSet["Filename"].str.split("_").str[0].str.replace("g", "")
        testSet = testSet.merge(mapping, on = "line", how = "left")

    # move subset of labelled test data to training data
    mediumSubset = testSet.loc[testSet["Sample_ID"] == "medium",:]
    testSet = testSet.loc[testSet["Sample_ID"] != "medium",:]
    trainIdx = np.random.choice(mediumSubset.reset_index(drop=True).index, size = int(mediumSubset.index.shape[0]/2))
    trainSet = pd.concat([trainSet, mediumSubset.iloc[trainIdx]]).reset_index()
    testSet = pd.concat([testSet, mediumSubset.iloc[~trainIdx]])


    # perform syncom specific predictions
    predList = []
    metricList = []
    syncoms = list(pd.Series(testSet["syncom"].unique()[testSet["syncom"].unique() != "medium"]).dropna())
    for sc in syncoms:
        metrics, pred = predictSynComSpecific(trainSet, testSet, sc, splitSize=0.2, gaussian=gaussian, model=model, metricsPlot=metricsPlot, probability=probability, taxonomy=taxonomy)
        predList.append(pred)
        metricList.append(metrics)


    # calculate the harmonic mean between the F1 score of the two syncoms
    modelMetrics = hmean(metricList)

    # convert to memory efficient data-types
    predDf = pd.concat(predList)

    if filename:
        predDf.to_csv(f'{filename}.tsv', sep="\t", index=False)

    if return_metrics:
        return modelMetrics

    return predDf, modelMetrics
# ~~~~~~~~~~~~~~~~~~~~~~~~~ PREDICT REISOLATED BACTERIA ~~~~~~~~~~~~~~~~~~~~~~~~~ #





