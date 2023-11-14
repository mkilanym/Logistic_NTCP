import os.path

import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from statsmodels.formula.api import glm
import statsmodels.api as sm
from scipy.stats import chi2#.cdf as pchisq # given a quantile, calculate the area underneath the chi-square distribution to that value. this returns the probability (alpha) where alpha refers to how much of data will lie beneath the given quantile.
import math

def Classification_Report_Plot(Classification_Report, PredictorsList, SaveDir = ""):
    # create empty dataframe
    df = pd.DataFrame(columns=["precision", "sensitivity", "f1_score", "predictor"])

    # fill the df
    for i, result in enumerate(Classification_Report):
        precision = result["precision"]
        sensitivity = result["recall"]
        f1 = result["f1-score"]
        predictor = PredictorsList[i]

        df = df.append({"precision": precision, "sensitivity": sensitivity, "f1_score": f1, "predictor": predictor}, ignore_index = True)

    print(df)
    # plot
    ax = sn.scatterplot(x=df["sensitivity"], y=df["precision"], hue=df["predictor"], style=df["predictor"], s=150)
    sn.move_legend(ax, "upper right", bbox_to_anchor=(1.1, 1))#plt.legend(bbox_to_anchor=(1.35, 1))
    plt.xlabel("sensitivity")
    plt.ylabel("precision")
    plt.xlim((0,1.05))
    plt.ylim((0, 1.05))

    if(SaveDir != ""):
        plt.savefig(os.path.join(SaveDir, "Classification_Report.png"))

    plt.show()

def Calibration_Plot(model, Data_of_Interest, predictor, endPoint, CategoryOutput=False, SaveDir = ""):
    # get probability estimates
    if(CategoryOutput):
        pred = model.predict_proba(Data_of_Interest[predictor])
    else:
        pred = model.predict(Data_of_Interest[predictor])

    y,x = calibration_curve(Data_of_Interest[endPoint], pred.values, n_bins=10)

    fig, ax = plt.subplots()
    plt.plot(x, y, marker='o')
    plt.xlabel("prediction rate")
    plt.ylabel("observed rate")
    plt.title(predictor)
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

    if (SaveDir != ""):
        plt.savefig(os.path.join(SaveDir,"{}.png".format(predictor)))

    plt.show()


def Get_Classification_Report(model, inDataFrame, predictor, reference):
    pred = model.predict(inDataFrame[predictor])
    threshold = 0.2
    binaryPredict = (pred >= threshold).astype(int)
    return classification_report(binaryPredict, reference, output_dict=True, zero_division=0.0, target_names=["class_0", "class_1"])


def Perform_Univariable_glm(Data_of_Interest, Predictors, EndPoint="Retinopathy_Flag", python_glm_file="", Target_Report="1.0"):

    dev_AIC_BIC_list = []
    Classification_Reports_list = []

    print("the loaded predictors are: {}".format(Predictors))

    for predictor in Predictors:  # ["Overall_V20_trans"]:
        print("Analyze predictor: {}".format(predictor))
        model = glm("{} ~ {}".format(EndPoint, predictor), data=Data_of_Interest, family=sm.families.Binomial())
        results = model.fit()  # (cov_type= 'hac', use_t=False,  cov_kwds=dict(maxlags=2))

        dev = results.null_deviance - results.deviance  # gain in -2*log-likelihood
        #pvalue = pchisq(dev, results.)

        if (math.isinf(results.bic_llf) or math.isnan(results.bic_llf)):
            print("nan BIC detected")
            # the idea behind this is shown in: (statmodels GLM Binomial  is the same model as Logit for binary endog but uses a different numerical approach)
            # https://stackoverflow.com/questions/46173061/statsmodels-throws-overflow-in-exp-and-divide-by-zero-in-log-warnings-and-ps

            model = sm.Logit(Data_of_Interest[EndPoint], Data_of_Interest[predictor])
            logitResults = model.fit(cov_type='hac', use_t=False, cov_kwds=dict(maxlags=2))

            AIC = results.aic
            BIC = results.bic
            PValue = logitResults.pvalues.loc[predictor] # A high p-value means that a coefficient is unreliable (insignificant), while a low p-value suggests that the coefficient is statistically significant
        else:
            AIC = results.aic
            BIC = results.bic_llf
            PValue = results.pvalues.loc[predictor] # A high p-value means that a coefficient is unreliable (insignificant), while a low p-value suggests that the coefficient is statistically significant
        
        PChi = 1-chi2.cdf(results.null_deviance-results.deviance, len(Data_of_Interest)-results.df_resid)
        
        # use Calibration_Evaluation_Plots_Class to plot and save calibration plots
        CalibrationDir = ""
        if(python_glm_file != ""):
            StudyDir, txt = os.path.split(python_glm_file)
            CalibrationDir = os.path.join(StudyDir,"Calibration_Plots")
            os.makedirs(CalibrationDir, exist_ok=True)
        Calibration_Plot(results, Data_of_Interest, predictor, EndPoint, CategoryOutput=False, SaveDir=CalibrationDir)

        # retrieve classification_report to plot accuracy vs. precission per predictor
        Classification_Report = Get_Classification_Report(results, Data_of_Interest, predictor, Data_of_Interest[EndPoint])
        
        # append all info in one dictionary
        dev_AIC_BIC_list.append({"AIC": AIC, "BIC": BIC, "dev": dev, "PValue":PValue, "PChi": PChi, "Predictor": predictor,
                                 "Precision": Classification_Report[Target_Report]["precision"], "Recall": Classification_Report[Target_Report]["recall"]})
        print(Classification_Report)
        
        Classification_Reports_list.append(Classification_Report[Target_Report])

    return dev_AIC_BIC_list, Classification_Reports_list
