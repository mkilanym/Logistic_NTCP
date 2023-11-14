# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:46:22 2023

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
from imblearn.over_sampling import SMOTE
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


from Feature_Selection_Class_MultiColinearity import Split_Predictors_into_Groups, forward_selected
from Feature_Selection_Class_MultiColinearity import Prepare_PredictorsSet_AfterSplitFunction
from Feature_Selection_Class_MultiColinearity import Select_Potential_Submodels

from Model_Composition_Class import Average_SubModels_Coeffs, Analytical_Logistic_Model_Prediction

from Model_Evaluation import Logistic_Model_Performance_at_Variuos_BinaryThreshold

from Print_Results import iOCT_Print_Results_From_List_of_models
import NTCP_Descriptive_Analysis_Class as NTCPClass
# from Calibration_Evaluation_Plots_Class import *
# import utils_imbalanced_Classification as utils




# Load Data
StudyDir = r"E:\NTCP_trials\Sectors_2_Targets"

Composite_Model_Folder = os.path.join(StudyDir, "Composite_Model_Folder_ConsideringImbalancedData")
os.makedirs(Composite_Model_Folder, exist_ok=True)


Data_of_Interest = pd.read_csv(os.path.join(StudyDir, "Transformed_Data_from_R.csv"))

# Remove NaN column & Remove the space in the column's name
NTCPClass.Remove_Nan_Column_And_Space_From_ColumnName(Data_of_Interest)


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

# create virtual balanced data
over = SMOTE(k_neighbors=1)
Transformed_X, Transformed_y = over.fit_resample(Data_of_Interest[X_Columns].values, Data_of_Interest["Retinopathy_Flag"].values)
Transformed_y = np.reshape(Transformed_y,(-1,1))
Appended_Transformed_Data = np.append(Transformed_X, Transformed_y, axis=1)
OverSampled_DF = pd.DataFrame(Appended_Transformed_Data, columns=X_Columns+["Retinopathy_Flag"])        
        
subModels = []
for k in Groups_dict.keys():
    print("work on group {}".format(k))
    groupPredictor = Groups_dict[k]
    subModels.append(forward_selected(OverSampled_DF[groupPredictor+["Retinopathy_Flag"]], "Retinopathy_Flag"))

iOCT_Print_Results_From_List_of_models(subModels, os.path.join(Composite_Model_Folder, "Results_of_Stepwise_Logistic_Regression_BIC.txt"))

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

# Data_of_Interest.plot.scatter("midPeriphery_IN_D50","Retinopathy_Flag")
# save model info

meanCoeff.to_csv(os.path.join(Composite_Model_Folder, "meanCoeff.csv"))
with open(os.path.join(Composite_Model_Folder, "FilteredModels.pkl"), 'wb') as f:
    pickle.dump(FilteredModels, f)
with open(os.path.join(Composite_Model_Folder, "Selected_Groups.pkl"), 'wb') as f:
    pickle.dump(Selected_Groups, f)
with open(os.path.join(Composite_Model_Folder, "Best_Threshold_Performance.pkl"), 'wb') as f:
    pickle.dump(Best_Threshold_Performance, f)




print("Done: go to internal validation code")