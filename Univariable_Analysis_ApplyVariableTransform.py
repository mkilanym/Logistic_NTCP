# -*- coding: utf-8 -*-
# https://blog.rtwilson.com/regression-in-python-using-r-style-formula-its-easy/
"""
Created on Mon Jul 10 19:07:42 2023

@author: mkmahassan
"""

import os
import shap
import math
import sklearn
import itertools
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# from collections import Counter
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer, SimpleImputer

#from R_Routine import *
from Print_Results import *
from Calibration_Evaluation_Plots_Class import *
# import utils_imbalanced_Classification as utils

from statsmodels.formula.api import glm
import statsmodels.api as sm


# Load Data
StudyDir = r"E:\NTCP_trials\NTCP_Per_Sector_2_Targets\farPeriphery_I"
Data_of_Interest = pd.read_csv(os.path.join(StudyDir, "Transformed_Data_from_R.csv"))
X_Columns = list(set(list(Data_of_Interest.columns)) - set(["Retinopathy_Flag"]))



# use python to perform univariate analysis (only - no transformations)
python_glm_file = os.path.join(StudyDir, "python_glm_file_bic_llf_Overall.txt")

print("Start glm")
dev_AIC_BIC_list, Classification_Reports_list = Perform_Univariable_glm(Data_of_Interest, X_Columns, EndPoint="Retinopathy_Flag", python_glm_file=python_glm_file, Target_Report='class_1')

dev_AIC_BIC_list = sorted(dev_AIC_BIC_list, key=lambda x: x['BIC'])
with open(python_glm_file, 'a') as writeResults:
    for dict in dev_AIC_BIC_list:
        writeResults.write("Feature: {},  Gain: {},  AIC: {},  BIC: {}, PChi: {}, PValue: {}, Precision: {}, Recall: {}\n".format(
            dict["Predictor"], dict["dev"], dict["AIC"], dict["BIC"], dict["PChi"], dict["PValue"], 
            dict["Precision"], dict["Recall"]))


Classification_Report_Plot(Classification_Reports_list, PredictorsList=[x['Predictor'] for x in dev_AIC_BIC_list], SaveDir=StudyDir)

# Data_of_Interest.to_csv(os.path.join(StudyDir, "Transformed_Data_of_Interest.csv"), index=False)

## save dev_AIC_BIC_list for the next step
# with open(os.path.join(StudyDir, "dev_AIC_BIC_list.pkl"), 'wb') as f:
#     pickle.dump(dev_AIC_BIC_list, f)

print("Done. go to Preselect_One_Predictor_Per_Sector.py in case of too many sectors features.")
print("Done. go to Multicolinearity_Considering_ImbalancedData.py")
