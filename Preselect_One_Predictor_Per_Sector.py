# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 23:17:28 2023

@author: mkmahassan
"""

import os
import numpy as np
import pandas as pd
import pickle
from Calibration_Evaluation_Plots_Class import Perform_Univariable_glm
import NTCP_Descriptive_Analysis_Class as NTCPClass



# Load Data
StudyDir = r"E:\NTCP_trials\Sectors_2_Targets"
Data_of_Interest = pd.read_csv(os.path.join(StudyDir, "Transformed_Data_from_R.csv"))

# Remove NaN column & Remove the space in the column's name
NTCPClass.Remove_Nan_Column_And_Space_From_ColumnName(Data_of_Interest)


X_Columns = list(set(list(Data_of_Interest.columns)) - set(["Retinopathy_Flag"]))

Sectors_Names =["macular_", "perimacular_ST_", "perimacular_SN_",
                "perimacular_IN_", "perimacular_IT_", "midPeriphery_ST_",
                "midPeriphery_SN_", "midPeriphery_IN_", "midPeriphery_IT_",
                "farPeriphery_S_", "farPeriphery_I_"]


python_glm_file = os.path.join(StudyDir, "python_glm_file_bic_llf_Sectors_2.txt")
dev_AIC_BIC_list, Classification_Reports_list = Perform_Univariable_glm(Data_of_Interest, X_Columns, EndPoint="Retinopathy_Flag", python_glm_file=python_glm_file, Target_Report='class_1')
    
Selected_Predictor_Per_Sector = []

for sector in Sectors_Names:
    Collected_Preds = []
    print("sector: {}".format(sector))
    # 1- collect all predictors for a sector
    for x in X_Columns:
        if(("{}V".format(sector) in x) or ("{}D".format(sector) in x)):
            TempDict = next(item for item in dev_AIC_BIC_list if item["Predictor"] == x)
            
            # penalize the transformed versions more than the originals
            if("trans" in x):
                TempDict["penalised_BIC"] = TempDict["BIC"] + 2*np.log(45)
            else:
                TempDict["penalised_BIC"] = TempDict["BIC"] 
                
            Collected_Preds.append(TempDict)
            
            print("{} , BIC= {}, pvalue= {}, Penlaized_BIC= {}".format(x, TempDict["BIC"], TempDict["PChi"], TempDict["penalised_BIC"]))
    print('*'*50)      
    
    # 2- sort them according to BIC
    Collected_Preds = sorted(Collected_Preds, key=lambda x: x['penalised_BIC']) # it was originally BIC 
    
    # 3- pick the lowest BIC
    Selected_Predictor_Per_Sector.append(Collected_Preds[0])
    
for item in Selected_Predictor_Per_Sector:
    print("{} , penalised_BIC= {}".format(item["Predictor"], item["penalised_BIC"]))
    
# the previously selected predictors were based on BIC not penalised_BIC    
with open(os.path.join(StudyDir, "Selected_Predictor_Per_Sector.pkl"), 'wb') as f:
    pickle.dump(Selected_Predictor_Per_Sector, f)