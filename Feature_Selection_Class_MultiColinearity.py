# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:46:31 2023

@author: mkmahassan
"""
import numpy as np
from statsmodels.formula.api import glm
import statsmodels.api as sm

def Recursive_Split(correlationMatrix, pointer, status, noColumns):
    
        
    if (pointer > noColumns-1):
        # pointer is advanced beyond the last element
        result = np.reshape(np.array(status),[1,-1]) #list(status)
        
    elif ((pointer > 0) and (np.any(correlationMatrix[pointer, 0:pointer].astype(int) & status[0:pointer].astype(int)))):
        
        conflict_with_included_var = np.any(correlationMatrix[pointer, 0:pointer].astype(int) & status[0:pointer].astype(int))
        no_conflict_with_future_var =  not np.any(correlationMatrix[pointer, (pointer+1):noColumns])
        #print("pointer = {}\n, sub-status= {}\n, status= {}\n, conflict_with_included_var= {}\n, no_conflict_with_future_var= {}\n".format(pointer, status[0:pointer], status, conflict_with_included_var, no_conflict_with_future_var))


        # do not select pointed element because of conflict to the left
        #print("recursive\n")
        result = Recursive_Split(correlationMatrix, pointer + 1, status, noColumns)
        
    elif ((pointer == noColumns-1) or (not (np.any(correlationMatrix[pointer, (pointer+1):noColumns]))) ):
        
        conflict_with_included_var = np.any(correlationMatrix[pointer, 0:pointer].astype(int) & status[0:pointer].astype(int))
        no_conflict_with_future_var =  not (np.any(correlationMatrix[pointer, (pointer+1):noColumns]))
        #print("pointer = {}\n, sub-status= {}\n, status= {}\n, conflict_with_included_var= {}\n, no_conflict_with_future_var= {}\n".format(pointer, status[0:pointer], status, conflict_with_included_var, no_conflict_with_future_var))


        # select pointed element because of no possible conflict to the right
        #print("add element\n")
        status[pointer] = True
        result = Recursive_Split(correlationMatrix, pointer + 1, status, noColumns)
  
    else: 
        
        conflict_with_included_var = np.any(correlationMatrix[pointer, 0:pointer].astype(int) & status[0:pointer].astype(int))
        no_conflict_with_future_var =  not np.any(correlationMatrix[pointer, (pointer+1):noColumns])
        #print("pointer = {}\n, sub-status= {}\n, status= {}\n, conflict_with_included_var= {}\n, no_conflict_with_future_var= {}\n".format(pointer, status[0:pointer], status, conflict_with_included_var, no_conflict_with_future_var))


        # split combination
        #print("split - before one\n")
        Temp = status.copy()
        one = Recursive_Split(correlationMatrix, pointer + 1, status, noColumns)
        
        status = Temp.copy()
        status[pointer] = True
        #print("split - after one === status= {}\n".format(status))
        two = Recursive_Split(correlationMatrix, pointer + 1, status, noColumns)
        result =np.append(one, two, axis=0)
        #print("split - after two\n")

    return result[:]

def Split_Predictors_into_Groups(correlationMatrix, Threshold=0.8):
#' Returns all combinations of elements that do not conflict eachother and that
#' cannot contain more elements without conflict.
#'
#' The algorithm searches recursively, starting from an empty combination (all
#' elements not selected) with a pointer to the first element. If the pointed
#' element has a conflict with a selected elements to the left, then the element
#' is not inserted and the pointer is advanced. Else, if the pointed element has
#' no possible conflict with any element to the right (with higher index), then
#' the element is inserted and the pointer is advanced to the right (index
#' increased by one). Otherwise, the combination is split: one combination does
#' not contain the pointed element and one does. In both cases the pointer is
#' advanced. The combination is full if the pointer is advanced beyond the last
#' element. This procedure may produce combinations that are subsets of other
#' combinations. These are removed at the end.

    absCorrelationMatrix = (np.absolute(correlationMatrix) > Threshold)    
    noColumns = correlationMatrix.shape[0] #len(correlationMatrix.columns)
    
    pointer = 0
    status = np.repeat(False, noColumns)
    
    Groups = Recursive_Split(absCorrelationMatrix, pointer, status, noColumns)
    Groups = np.unique(Groups, axis=0) 
    
    
    subsets = np.repeat(False, Groups.shape[0])
    for i in range(Groups.shape[0]):
        for j in range(Groups.shape[0]): 
            if ((i != j) and (not np.any(Groups[i,:] & np.logical_xor(Groups[i,:], Groups[j,:])))):
                subsets[i] = True
      
    

    return Groups[(1-subsets).astype(bool),:]


def Prepare_PredictorsSet_AfterSplitFunction(PredictorsList, SplitGroups):
    
    Groups_dict = {}
    
    for g in range(SplitGroups.shape[0]):
        Groups_dict["G{}".format(g+1)] = [x for i,x in enumerate(PredictorsList) if SplitGroups[g,i]==True]

    return Groups_dict


def forward_selected(data, response):
    # from: https://planspace.org/20150423-forward_selection_with_statsmodels/
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    
    # # the base model
    # formula = "{} ~ 1".format(response)
    # model = glm(formula, data=data, family=sm.families.Binomial())
    # results = model.fit()
    # current_score = best_new_score = results.bic_llf
    
    current_score, best_new_score = 1000000.0, 1000000.0
    while remaining and (current_score == best_new_score):
        scores_with_candidates = []
        
        for candidate in remaining:
            formula = "{} ~ {}".format(response, ' + '.join(selected + [candidate]))
            model = glm(formula, data=data, family=sm.families.Binomial())
            results = model.fit()
            
            score = results.bic_llf
            if(np.isnan(score)):
                score = results.bic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0] #scores_with_candidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            
    formula = "{} ~ {}".format(response,  ' + '.join(selected))
    model = glm(formula, data, family=sm.families.Binomial()).fit()
    return model


def Select_SubModels_With_Positive_Coeff(SubModels_List):
    subModels_with_positive_coeff = []
    for model in SubModels_List:
        varCount = 0
        Positive = 0
        for var in model.params.index:
            if("Intercept" not in var):
                varCount = varCount + 1
                if(model.params[var] > 0):
                    Positive = Positive + 1
        if(varCount == Positive):
            subModels_with_positive_coeff.append(model)
            
    return subModels_with_positive_coeff[:]

def Reject_Transformed_Variables(SubModels_List, Predictors_Groups_dict):
    selectedModels = []
    Selected_Groups = []
    for i,k in enumerate(Predictors_Groups_dict.keys()):
        # accept models with multi predictors at this stage
        if(len(Predictors_Groups_dict[k]) > 1):
            selectedModels.append(SubModels_List[i])
            Selected_Groups.append(Predictors_Groups_dict[k])
        else:
        # else: ignore the transformed version if it performs as same as the original form of the predictor    
            if("trans" in Predictors_Groups_dict[k][0]):
                Trans_Bic = np.round(SubModels_List[i].bic_llf, decimals=3)
                
                Original_Pred = Predictors_Groups_dict[k][0].replace("_trans", "") # get the original name 
                # search for the original form
                for j,Ok in enumerate(Predictors_Groups_dict.keys()):
                    if(len(Predictors_Groups_dict[Ok]) == 1): # make sure you look for a model that has single predictor
                        if(Original_Pred == Predictors_Groups_dict[Ok][0]):
                            Orig_Bic = np.round(SubModels_List[j].bic_llf, decimals=3)
                            
                            if(Orig_Bic <= Trans_Bic):
                                selectedModels.append(SubModels_List[j])
                                Selected_Groups.append(Predictors_Groups_dict[Ok])
                                
    return selectedModels[:], Selected_Groups[:]
                
    
def Select_SubModels_With_BIC_Less_Than_Threshold(SubModels_List):
    # sort models according to bIC ascendingly
    #SortedModels = sorted(SubModels_List, key=lambda x: x.bic_llf)
    Bic_List = []
    for model in SubModels_List:
        if(np.isnan(model.bic_llf)):
            Bic_List.append(model.bic)
        else:
            Bic_List.append(model.bic_llf)
    
    BIC_Thr = np.amin(Bic_List) + np.log(SubModels_List[0].nobs) #BIC_Thr = SortedModels[0].bic_llf + np.log(SortedModels[0].nobs)
    selectedModels = []
    Selected_Groups = []
    
    for i,model in enumerate(SubModels_List):
        if(((model.bic_llf <= BIC_Thr) and not np.isnan(model.bic_llf)) or ((model.bic <= BIC_Thr) and not np.isnan(model.bic))):
            selectedModels.append(model)
            
    return selectedModels[:]
    
def Extract_PredictorsNames_From_Model(SubModels_List):
    Selected_Groups = {}
    
    for i,model in enumerate(SubModels_List):
        Temp = []
        for var in model.params.index:
            if("Intercept" not in var):
                Temp.append(var)
                
        Selected_Groups["G{}".format(i+1)] = Temp #Selected_Groups.append(np.array(Temp))
        del(Temp)
        
    return Selected_Groups

        
def Select_Potential_Submodels(SubModels_List, Predictors_Groups_dict):
    # this function clean up the submodels, and gives the potential submodels that can be used for NTCP model composition
    
    # 1- neglect the submodels that have negative coefficient for any predictoe
    Positive_SubModels = Select_SubModels_With_Positive_Coeff(SubModels_List)
    
    # # 2- select models that have untransformed parameters
    # Positive_SubModels, Selected_Groups = Reject_Transformed_Variables(Positive_SubModels, Predictors_Groups_dict)   
    
    # 3- select the models that have BIC value < Threshold BIC
    Positive_SubModels = Select_SubModels_With_BIC_Less_Than_Threshold(Positive_SubModels)
    Selected_Groups = Extract_PredictorsNames_From_Model(Positive_SubModels)
    
    # NOTE: I have to provide the original predictor group where the selected predictors came from
    
    return Positive_SubModels[:], Selected_Groups           
    