# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:45:16 2023

@author: mkmahassan
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, average_precision_score
from scipy.stats import chi2
import matplotlib.pyplot as plt

def Logistic_Model_Performance_at_Variuos_BinaryThreshold(reference, prediction, binsNo=10):
    
    Performance_List = []
    for percent in [99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]:
        Thr = np.percentile(prediction, percent)
        
        LL          = Get_LogLeikelihood(reference, prediction)
        AUC_ROC     = Get_AUC_ROC_Score(reference, prediction, Thr=Thr)
        AUC_PR      = average_precision_score(reference, (prediction>=Thr)*1.0)
        Pvalue      = Get_HosmerLemeshow_Goodness_of_Fit_Test(reference, prediction, binsNo=binsNo)
        report      = Get_Classification_Report(reference, prediction, Thr)
        precision   = report["class_1"]["precision"]
        sensitivity = report["class_1"]["recall"]
        f1          = report["class_1"]["f1-score"]
        CM          = Get_Confusion_Matrix(reference, prediction, Thr)
        
        Performance_List.append({"percent": percent, "Thr": Thr, "LL": LL, "AUC_ROC": AUC_ROC, 
                                 "Pvalue": Pvalue, "precision": precision, "f1": f1, "AUC_PR": AUC_PR,
                                 "sensitivity": sensitivity, "Confusion_Matrix": CM})
        
    Performance_List = sorted(Performance_List, key=lambda d: d['AUC_PR']) 
    
    return Performance_List
    
def Get_LogLeikelihood(reference, prediction):
    # reference is binary, prediction is probabilities
    return np.sum(np.log(reference*prediction + (1-reference)*(1-prediction)))

def Get_AUC_ROC_Score(reference, prediction, Thr=0.5):
    # reference is binary, prediction is probabilities
    binaryPredict = (prediction>=Thr)*1.0
    
    logit_roc_auc = roc_auc_score(reference, binaryPredict)
    fpr, tpr, thresholds = roc_curve(reference, prediction)
    
    
    plt.figure()
    plt.plot(fpr, tpr, label='AUC = %0.2f ' % logit_roc_auc + 'at Thr = %0.2f' %Thr )
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    return logit_roc_auc

def Get_Confusion_Matrix(reference, prediction, Thr=0.5,  labels=["observed", "predicted"]):
    # only reference is binary. prediction is probability
    
    binaryPrediction = (prediction >= Thr)*1.0
    cm = confusion_matrix(reference, binaryPrediction)
    
    #plotting the heatmap for correlation
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    disp.ax_.set_title('Confusion Matrix at Threshold = {}'.format(Thr))
    
    return cm

def Get_Classification_Report(reference, prediction, Thr= 0.5):
    # reference is binary, prediction is probabilities
    binaryPrediction = (prediction >= Thr)*1.0
    
    report = classification_report(binaryPrediction, reference, output_dict=True, 
                                   zero_division=0.0, target_names=["class_0", "class_1"])
    
    return report

def Get_HosmerLemeshow_Goodness_of_Fit_Test(reference, prediction, binsNo=10):
    # only reference is binary, , prediction is probabilities
    
    # this code is implemented based on:
    # https://stackoverflow.com/questions/40327399/hosmer-lemeshow-goodness-of-fit-test-in-python
    # https://jbhender.github.io/Stats506/F18/GP/Group5.html
    
    if binsNo < 2:
        binsNo = 2
        
    hl_df = pd.DataFrame({"predict": prediction, "observed": reference})
    hl_df["decile"] = pd.qcut(hl_df["predict"],binsNo, duplicates="drop")
    
    
    #We will calculate all the observed ones in every decile
    obsevents_1 = hl_df["observed"].groupby(hl_df.decile).sum()

    #We will find all the observed zeroes of every decile if we substract the obsevents_1 from the
    #number of elements in every decile
    obsevents_0 = hl_df["observed"].groupby(hl_df.decile).count() - obsevents_1
    
    expevents_1 = hl_df["predict"].groupby(hl_df.decile).sum()

    #We will find the expected number of events Y = 0 for every decile by substracting the
    #expevents_1 from the number of elements in every decile
    expevents_0 = hl_df["predict"].groupby(hl_df.decile).count() - expevents_1
    
    
    hl = (((obsevents_0 - expevents_0)**2)/(expevents_0)).sum() + (((obsevents_1 - expevents_1)**2)/(expevents_1)).sum()
    pvalue = 1 - chi2.cdf(hl , binsNo - 2)
    
    return pvalue