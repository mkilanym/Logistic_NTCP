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

import NTCP_Descriptive_Analysis_Class as NTCPClass
from Feature_Selection_Class_MultiColinearity import Split_Predictors_into_Groups, forward_selected
from Feature_Selection_Class_MultiColinearity import Prepare_PredictorsSet_AfterSplitFunction
from Feature_Selection_Class_MultiColinearity import Select_Potential_Submodels

from Model_Composition_Class import Average_SubModels_Coeffs, Analytical_Logistic_Model_Prediction

from Model_Evaluation import Get_LogLeikelihood, Get_AUC_Score, Get_HosmerLemeshow_Goodness_of_Fit_Test
from Model_Evaluation import Get_Confusion_Matrix, Get_Classification_Report
from Model_Evaluation import Logistic_Model_Performance_at_Variuos_BinaryThreshold

from Print_Results import iOCT_Print_Results_From_List_of_models

# from Calibration_Evaluation_Plots_Class import *
# import utils_imbalanced_Classification as utils




# Load Data
StudyDir = r"E:\NTCP_trials\Sectors_2_Targets"
Data_of_Interest = pd.read_csv(os.path.join(StudyDir, "Transformed_Data_from_R.csv"))
X_Columns = list(set(list(Data_of_Interest.columns)) - set(["Retinopathy_Flag"]))

# Use selected predictor per sector if exist
Selected_Preds_Path = os.path.join(StudyDir, "Selected_Predictor_Per_Sector.pkl") 
if(os.path.isfile(Selected_Preds_Path)):
    with open(Selected_Preds_Path, 'rb') as f:
        Selected_Predictor_Per_Sector = pickle.load(f)
    X_Columns = [item["Predictor"] for item in Selected_Predictor_Per_Sector]

# calculate the colinearity between predictors
corr_matrix = Data_of_Interest[X_Columns].corr()
corr_matrix = np.round(corr_matrix,decimals=1)

#plotting the heatmap for correlation
ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("coolwarm", as_cmap=True))
plt.show()

# create groups of predictors
Groups = Split_Predictors_into_Groups(corr_matrix.values, Threshold=0.7) 

Groups_dict = Prepare_PredictorsSet_AfterSplitFunction(X_Columns, Groups)

subModels = []
for k in Groups_dict.keys():
    groupPredictor = Groups_dict[k]
    subModels.append(forward_selected(Data_of_Interest[groupPredictor+["Retinopathy_Flag"]], "Retinopathy_Flag"))

iOCT_Print_Results_From_List_of_models(subModels, os.path.join(StudyDir, "Results_of_Stepwise_Logistic_Regression_BIC.txt"))

# filter out the submodels
FilteredModels, Selected_Groups = Select_Potential_Submodels(subModels, Groups_dict)

# compose a model and predict
meanCoeff = Average_SubModels_Coeffs(FilteredModels)
analyticalPredict = Analytical_Logistic_Model_Prediction(meanCoeff, Data_of_Interest, type = 'response')


# look for the best threshold to the analytical prediction that results in highest performance
print("Find the best threshold to binarizr the logistic model output")
Performance_List = Logistic_Model_Performance_at_Variuos_BinaryThreshold(Data_of_Interest["Retinopathy_Flag"], 
                                                                          analyticalPredict, binsNo=3)

Best_Threshold_Performance = Performance_List[-1]

# save model info
Composite_Model_Folder = os.path.join(StudyDir, "Composite_Model_Folder")
os.makedirs(Composite_Model_Folder, exist_ok=True)

meanCoeff.to_csv(os.path.join(Composite_Model_Folder, "meanCoeff.csv"))
with open(os.path.join(Composite_Model_Folder, "FilteredModels.pkl"), 'wb') as f:
    pickle.dump(FilteredModels, f)
with open(os.path.join(Composite_Model_Folder, "Selected_Groups.pkl"), 'wb') as f:
    pickle.dump(Selected_Groups, f)
with open(os.path.join(Composite_Model_Folder, "Best_Threshold_Performance.pkl"), 'wb') as f:
    pickle.dump(Best_Threshold_Performance, f)



# # with open(os.path.join(Composite_Model_Folder, "Best_Threshold_Performance.pkl"), 'rb') as f:
# #     loaded_list_of_dicts = pickle.load(f)

print("Done: go to internal validation code")



# test code to find unique models based on the variables names
# A = [1,2,3]
# B = [10,2,3,5]
# C = [1,2,3]
# unique = [list(x) for x in set(tuple(x) for x in [A,B,C])]
# print(unique)