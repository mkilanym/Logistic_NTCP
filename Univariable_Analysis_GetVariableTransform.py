# -*- coding: utf-8 -*-
# https://blog.rtwilson.com/regression-in-python-using-r-style-formula-its-easy/
"""
Created on Mon Jul 10 19:07:42 2023

@author: mkmahassan
"""

import os
import math
import sklearn
import itertools
#import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from IPython.display import Image
from six import StringIO

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

import NTCP_Descriptive_Analysis_Class as NTCPClass
#from R_Routine import *
#from Print_Results import *
from Calibration_Evaluation_Plots_Class import *
# import utils_imbalanced_Classification as utils

from statsmodels.formula.api import glm
import statsmodels.api as sm


# Load Data
Features_Dir = "E:\RT-Data\Extracted_Dosimetric_Features_Complications"
Sectors_MinMaxMeanSTD = pd.read_csv(os.path.join(Features_Dir, "Retinal_3MS_Features.txt")        ,sep="\t")
Sectors_DVH           = pd.read_csv(os.path.join(Features_Dir, "Retinal_DVH_Sectors_Features.txt"),sep="\t")
Global_DVH            = pd.read_csv(os.path.join(Features_Dir, "Retinal_DVH_Overall_Features.txt"),sep="\t")
Retinopathy           = pd.read_csv(os.path.join(Features_Dir, "Retinopathy_Conditions_Kilany2Cases.txt")      ,sep="\t")

# Quick clean to dataframes
# Remove NaN column & Remove the space in the column's name
NTCPClass.Remove_Nan_Column_And_Space_From_ColumnName(Sectors_MinMaxMeanSTD)
NTCPClass.Remove_Nan_Column_And_Space_From_ColumnName(Sectors_DVH)
NTCPClass.Remove_Nan_Column_And_Space_From_ColumnName(Global_DVH)
NTCPClass.Remove_Nan_Column_And_Space_From_ColumnName(Retinopathy)

# Create new column
Sectors_MinMaxMeanSTD["Case_Eye"] = Sectors_MinMaxMeanSTD["Case_Name"]+Sectors_MinMaxMeanSTD["Eye"]
Sectors_DVH["Case_Eye"] = Sectors_DVH["Case_Name"]+Sectors_DVH["Eye"]
Global_DVH["Case_Eye"] = Global_DVH["Case_Name"]+Global_DVH["Eye"]
Retinopathy["Case_Eye"] = Retinopathy["Case_Name"]+Retinopathy["Eye"]

# Drop the unnecessary columns
Sectors_MinMaxMeanSTD.drop(["Case_Name", "Eye"], axis=1, inplace=True)
Sectors_DVH.drop(["Case_Name", "Eye"], axis=1, inplace=True)
Global_DVH.drop(["Case_Name", "Eye"], axis=1, inplace=True)
Retinopathy.drop(["Case_Name", "Eye"], axis=1, inplace=True)

# Get the cases with complications
Cases_With_ComplicationsDF = NTCPClass.Get_Rows_With_Complication(Retinopathy)
Case_Condition_MaxDose_DF = NTCPClass.Get_MaxDose_Per_Complicated_Sector(Cases_With_ComplicationsDF, Sectors_MinMaxMeanSTD, Threshold=20)


# Joint the Case_Condition_MaxDose_DF with Sectors_MinMaxMeanSTD, Sectors_DVH, and Global_DVH
print("Case_Condition_MaxDose_DF \n {}".format(Case_Condition_MaxDose_DF.info()))
print("Sectors_MinMaxMeanSTD \n {}".format(Sectors_MinMaxMeanSTD.info()))

Join_Sectors_MinMaxSTD = Sectors_MinMaxMeanSTD.merge(Case_Condition_MaxDose_DF[["Case_Eye","Retinopathy_Flag"]], on =["Case_Eye"], how="left", indicator=False)
Join_Sectors_DVH       = Sectors_DVH.merge(Case_Condition_MaxDose_DF[["Case_Eye", "Retinopathy_Flag"]], on =["Case_Eye"], how="left", indicator=False)
Join_Global_DVH        = Global_DVH.merge(Case_Condition_MaxDose_DF[["Case_Eye", "Retinopathy_Flag"]], on =["Case_Eye"], how="left", indicator=False)

Join_Whole             = pd.merge(Join_Sectors_DVH, Join_Sectors_MinMaxSTD, on =["Case_Eye"], how="left", indicator=False)
Join_Whole             = pd.merge(Join_Whole, Join_Global_DVH, on =["Case_Eye"], how="left", indicator=False)
Join_Whole.drop(["Retinopathy_Flag_x", "Retinopathy_Flag_y"], axis=1, inplace=True)


# Fill the nan cells in Retinopathy_Flag with 0
Join_Sectors_MinMaxSTD["Retinopathy_Flag"] = Join_Sectors_MinMaxSTD["Retinopathy_Flag"].fillna(0)
Join_Sectors_DVH["Retinopathy_Flag"]       = Join_Sectors_DVH["Retinopathy_Flag"].fillna(0)
Join_Global_DVH["Retinopathy_Flag"]        = Join_Global_DVH["Retinopathy_Flag"].fillna(0)
Join_Whole["Retinopathy_Flag"]             = Join_Whole["Retinopathy_Flag"].fillna(0)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# select the dataframe of interest. Clean it for training
DummyDF = Join_Sectors_DVH.copy(deep=True)
Data_of_Interest = DummyDF.copy(deep=True)

Columns_of_Interest = set(list(Data_of_Interest.columns)) - set(["Case_Eye"])
X_Columns = list(Columns_of_Interest - set(["Retinopathy_Flag"]))
Data_of_Interest = Data_of_Interest[list(Columns_of_Interest)]
print("Data_of_Interest type = {}".format(type(Data_of_Interest)))

StudyDir = r"E:\NTCP_trials\Sectors_2_Targets"
Data_of_Interest.to_csv(os.path.join(StudyDir, "Data_of_Interest_before_R.csv"), index=False)
# #----------------------------------------------------------------------------------
# # run univariable Logistic regression analysis
# # this code is inspired from https://medium.com/analytics-vidhya/calling-r-from-python-magic-of-rpy2-d8cbbf991571
# from rpy2 import robjects
# from rpy2.robjects import pandas2ri
# #from rpy2.robjects.packages import importr
#
# # Defining the R script and loading the instance in Python
# r = robjects.r
#
# #converting df into r object for passing into r function
# with robjects.robject.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
#     df_r = robjects.conversion.py2ri(Data_of_Interest)
#
#     # transform all features
#     CodePath = 'C:\Program Files Extra\RT_Work\R_Code\mfp.R'
#     FunctionName = 'Transform_Features'
#     InputsDict = {"data": df_r, "candidates": X_Columns, "endpoint": "Retinopathy_Flag"}
#     Features_Transformation = R_Routine(r, CodePath, FunctionName, InputsDict)
#
#     iOCT_Print_Results_Univariable_Transformation(Features_Transformation, os.path.join(StudyDir, "Transform_OverallFeatures.txt"))
#
#
#
#
# # use python to perform univariate analysis (only - no transformations)
# python_glm_file = os.path.join(StudyDir, "python_glm_file_bic_llf_Overall.txt")
# print("Start glm")
# dev_AIC_BIC_list, Classification_Reports_list = Perform_Univariable_glm(Data_of_Interest, X_Columns, EndPoint="Retinopathy_Flag", python_glm_file=python_glm_file, Target_Report="class_1")
#
# Classification_Report_Plot(Classification_Reports_list, PredictorsList=[x['Predictor'] for x in dev_AIC_BIC_list], SaveDir=StudyDir)
#
# Data_of_Interest.to_csv(os.path.join(StudyDir, "Data_of_Interest.csv"), index=False)
#
# # CR = Classification_Reports_list[0]
print("Done: go use test.Rmd in R_Code folder then back to python to use Univariable_Analysis_ApplyVariableTransform.py")
