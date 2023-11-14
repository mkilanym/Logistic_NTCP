# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:00:16 2022

@author: mkmahassan
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn import tree #from sklearn.tree import export_graphviz
#import pydotplus
from IPython.display import Image
from six import StringIO


def Remove_Nan_Column_And_Space_From_ColumnName(DF):
    DF.dropna(axis=1, inplace=True)
    Correct_Columns_Names(DF)

def Correct_Columns_Names(DF):
    Columns = DF.columns.to_list()
    Dict = {}
    for col in Columns:
        Dict[col] = col.replace(" ","")
    DF.rename(columns=Dict, inplace=True)
    
def Get_Rows_With_Complication(RetinopathyDF):
    Complicated_Cases = RetinopathyDF.loc[RetinopathyDF["Condition"] != "None "]
    return Complicated_Cases

def Get_Sectors_Names_List(DF):
    Columns = DF.columns.to_list()
    #Columns = [c.replace(" ","") for c in Columns]
    Unwanted_Columns = ["Case_Name", "Eye", "Condition", "Case_Eye", "Retinopathy_Flag"]
    Sectors = [name for name in Columns if name not in Unwanted_Columns]
    return Sectors

def Get_MaxDose_Per_Complicated_Sector(Complication_DF, Sectors_Features_DF, Threshold = 40):
    Sectors_Names = Get_Sectors_Names_List(Complication_DF)
    Dict = {"Case_Eye": [], "Condition": [],"maxDose": [], "Retinopathy_Flag": [], "maxCovering": []}
    
    for r in range (Complication_DF.shape[0]): #iterate over rows

      TempRow = Complication_DF.iloc[[r]]
      
      # pick up the index of sectors that have complications
      Target_Sector_Index = np.transpose(np.array(np.where(TempRow[Sectors_Names].values > 0)))
      Target_Sector_Index = Target_Sector_Index[:,1]
      
      # get the patient name
      TargetCaseName = (TempRow["Case_Eye"].values)
      
      # get the condition
      Target_Condition = (TempRow["Condition"].values)
      
      Dict["Case_Eye"].append(TargetCaseName[0])
      Dict["Condition"].append(Target_Condition[0])
      
      Value_List = []
      Area_List = []
      for i in Target_Sector_Index:
          Value = Sectors_Features_DF["{}_max".format(Sectors_Names[i])].loc[Sectors_Features_DF["Case_Eye"] == TargetCaseName[0]] 
          Value_List.append(Value.values[0])
          
          Value = TempRow["{}".format(Sectors_Names[i])]
          Area_List.append(Value.values[0])
    
      Dict["maxDose"].append(np.amax(Value_List))
      Dict["maxCovering"].append(np.amax(Area_List)*100.0)
            
      Flag = lambda x: 1 if (x >=Threshold) else 0
      Dict["Retinopathy_Flag"].append(Flag(np.amax(Value_List)))
      
      del Value_List
      del Area_List
                
      
    return pd.DataFrame(Dict)
          

def Plot_Features_Histo_Per_Class(Data_of_Interest, SaveFolder):
    # visual check the features overlaping between classes before using SMOTE to synthesis minority class data 
    Retino_Group = Data_of_Interest.loc[Data_of_Interest["Retinopathy_Flag"] == 1]
    Non_Retino_Group = Data_of_Interest.loc[Data_of_Interest["Retinopathy_Flag"] == 0]

    Columns_of_Interest = list(Data_of_Interest.columns)
    for col in Columns_of_Interest:
        sns.distplot(Retino_Group.loc[:, col], norm_hist=True, kde=False, label='Retino', color="r")
        sns.distplot(Non_Retino_Group.loc[:, col], norm_hist=True, kde=False, label='Non_Retino', color="b")
        plt.legend()
        plt.savefig(os.path.join(SaveFolder,"{}.png".format(col)))
        plt.show()      

def Plot_Confusion_Matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    RF_Pred_df = pd.DataFrame(cm, columns=["Normal", "Complication"],index=["Normal", "Complication"])
    ax = sns.heatmap(RF_Pred_df, annot=True, linewidth=.5)
    ax.tick_params(colors='b', which='both')  # 'both' refers to minor and major axes
    plt.xlabel('Predicted',fontsize=15, color="r")
    plt.ylabel('Actual',fontsize=15, color="r")
    
    plt.show()     
    
def Plot_Feature_Importance(models_list, Features_Names, Plot_Trees = False):
        
    for i,model in enumerate(models_list):
        importances = model.feature_importances_
        STD = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        
        # select the significant features
        ind = np.where(importances > 0)
        ind_of_interest = np.where(importances >= 0.5*np.amax(importances) )
        
        significant_importances = importances[ind]
        significant_STD = STD[ind]
        significant_features = np.array(Features_Names)[ind]
        Features_of_Interest = np.array(Features_Names)[ind_of_interest]
        
        forest_importances = pd.Series(significant_importances, index=significant_features)
        
            
        fig, ax = plt.subplots()
        forest_importances.plot.bar(ax=ax) #(yerr=significant_STD, ax=ax)
        ax.set_title("Feature importances")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
        
        if(Plot_Trees):
            Plot_RandomForest(model, Features_Names, class_names=["Normal", "Complication"])

    return Features_of_Interest

def Plot_RandomForest(model, feature_names, class_names=["Normal", "Complication"]):

    # os.environ["PATH"] += os.pathsep + r'C:\Users\mkmahassan\.conda\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz'

    for estimator in model.estimators_:
        
        tree.plot_tree(estimator, feature_names=feature_names, class_names=class_names,
                       filled=True, rounded=True, proportion=True, impurity=False, precision=2)
        plt.show()
        #dot_data = StringIO()
    
        #export_graphviz(estimator, feature_names=feature_names,out_file=dot_data,
        #            filled=True, rounded=True, proportion=True,special_characters=True,
        #            impurity=False, class_names=class_names, precision=2)
    
        #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        #Image(graph.create_png())
        #plt.show()
def BoxPlot_TPTNFPFN_Features(TP,TN,FP,FN):
    df1 = pd.DataFrame(TP).assign(Trial="TP")
    df2 = pd.DataFrame(TN).assign(Trial="TN")
    df3 = pd.DataFrame(FP).assign(Trial="FP")
    df4 = pd.DataFrame(FN).assign(Trial="FN")
    
    cdf = pd.concat([df1, df2, df3, df4])                                # CONCATENATE
    mdf = pd.melt(cdf, id_vars=['Trial'], var_name=['Category'])      # MELT
    
    ax = sns.boxplot(x="Trial", y="value", hue="Category", data=mdf)  # RUN PLOT   
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()
    
def General_BoxBlot(inList, NamesList, SaveFolder="",FeatureName=""):
    dfList = list()
    for i in range(len(inList)):
        dfList.append( pd.DataFrame(inList[i]).assign(Trial=NamesList[i]))
        
    cdf = pd.concat(dfList)                                # CONCATENATE
    mdf = pd.melt(cdf, id_vars=['Trial'], var_name=['Category'])      # MELT
    
    ax = sns.boxplot(x="Trial", y="value", data=mdf)  # RUN PLOT   #, hue="Category"
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(label=FeatureName, loc="center")
    
    if(len(SaveFolder) > 0):
        plt.savefig(os.path.join(SaveFolder,"{}.png".format(FeatureName)))
        
    plt.show()
    
def General_BarPlot(x,y):
    dataSeries = pd.Series(y, index=x)
        
    fig, ax = plt.subplots()
    dataSeries.plot.bar(ax=ax) #(yerr=significant_STD, ax=ax)
    ax.set_ylim(0.95,1.0)
    plt.xticks(rotation=80) #plt.xlabel(ax.get_xlabel(), rotation=30)
    ax.set_title("Feature importances")
    ax.set_ylabel("1-pValue")
    fig.tight_layout()
    plt.show()
        