# -*- coding: utf-8 -*-
# https://blog.rtwilson.com/regression-in-python-using-r-style-formula-its-easy/
"""
Created on Mon Jul 10 19:07:42 2023

@author: mkmahassan
"""

import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from imblearn.over_sampling import SMOTE

import NTCP_Descriptive_Analysis_Class as NTCPClass
from Calibration_Evaluation_Plots_Class import *


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


print("Done: go use test.Rmd in R_Code folder then back to python to use Univariable_Analysis_ApplyVariableTransform.py")
