import pandas as pd
import numpy as np
import os

def R_Print_Results_Univariable_Analysis(results, summary_candidates = True,
                                       number_summary_print = None, LL = True,
                                               number_LL_print = None):
    if(summary_candidates):
        candidates = Get_Candidates_names(results, number_summary_print)

        for i in range(len(candidates)):
            print(candidates[i])

            dev_index = results[i].names.index('dev')
            pvalue_index = results[i].names.index('p.value')

            dev = results[i][dev_index]  # [4]
            dev = float(dev[0])
            pvalue = results[i][pvalue_index]  # [5]
            pvalue = float(pvalue[0])

            if("numeric" in results[i][0]): # type

                dev_trans_index = results[i].names.index('dev.trans')

                dev_trans = results[i][dev_trans_index]#[9]
                dev_trans = float(dev_trans[0])

                print('Gain in -2logLikelihood: {} (p = {})\n\n'.format(dev, pvalue))
                if((dev_trans - dev) >= 2):
                    if ("trans" in results[i].names):

                        trans_index = results[i].names.index('trans')
                        trans = results[i][trans_index]#[8]

                        print("{} \n".format(trans[0])) # print the proposed transform

                    else:
                        pvalue_trans_index = results[i].names.index('p.value.trans')
                        formula_trans_index = results[i].names.index('formula.trans')

                        pvalue_trans = results[i][pvalue_trans_index]#[10]
                        pvalue_trans = float(pvalue_trans[0])
                        formula_trans = results[i][formula_trans_index]

                        print(' ... transformed: {} (p = {})\n   ... transform: {}\n\n'.format(dev_trans, pvalue_trans, formula_trans[0]))

                #----------------------------------------------------------------------
                mean_index  = results[i].names.index('mean')
                sd_index    = results[i].names.index('sd')
                coeff_index = results[i].names.index('coeff')

                Mean       = results[i][mean_index]
                STD        = results[i][sd_index]
                Coeff      = results[i][coeff_index][0]
                Std_Error  = results[i][coeff_index][1]
                pval_coeff = results[i][coeff_index][2]

                Mean       = float(Mean[0])
                STD        = float(STD[0])
                #Coeff      = float(Coeff[0])
                #Std_Error  = float(Std_Error[0])
                #pval_coeff = float(pval_coeff[0])

                print(pd.DataFrame({'Mean': Mean, 'STD': STD, 'Coeff': Coeff, 'Std_Error': Std_Error, 'pval_coeff': pval_coeff}, index=[0]))
                # ----------------------------------------------------------------------

            else: # not numeric
                print('Gain in -2logLikelihood: {} (p = {})\n\n'.format(dev, pvalue))

                # if (any(sublist$levels["n.events"] == 0)):
                #     cat('** Warning: level without any events\n\n')
                #     print(sublist$levels)
                #     cat('\n')
                #     print(sublist$coeff)
                # else:
                #     print(data.frame(sublist$levels, rbind(rep(0, 3), sublist$coeff)))
                #
                # cat('\n')
                # if (length(sublist$dropped.levels) != 0):
                #     cat(c('Dropped levels:', sublist$dropped.levels), sep = '\n')

            print("\n\n")

    if LL == True:
        if (number_LL_print == None):
            number_LL_print = len(results) - 1
        print(results[-1]) #print(results[-1][0:number_LL_print])


def iOCT_Print_Results_Univariable_Transformation(results, save_file, summary_candidates=True):
    if(summary_candidates):
        candidates = results.names

        with open(save_file, 'a') as writeResults:

            writeResults.write("candidate\t")
            writeResults.write("formula\t")
            writeResults.write("gain in -2*loglikelihood\t")
            writeResults.write("p.value\t")
            writeResults.write("AIC\t")
            writeResults.write("BIC\n")

            for i in range(len(candidates)):
                writeResults.write("{}\t".format(candidates[i])) # print the original feature name


                trans_index = results[i].names.index('formula.trans')
                trans = results[i][trans_index]
                writeResults.write("{}\t".format(trans[0]))  # print the proposed transform


                dev_trans = float(Get_value_From_Rlist(results[i], 'dev.trans', candidates[i]))
                writeResults.write("%6.2f\t" % dev_trans) # print the gain in -2*log-likelihood
                #print('Gain in -2logLikelihood: {} (p = {})\n\n'.format(dev, pvalue))


                pvalue_trans = float(Get_value_From_Rlist(results[i], 'p.value.trans', candidates[i]))
                writeResults.write("%6.2f\t" % pvalue_trans) # print the p-value

                AIC_trans = float(Get_value_From_Rlist(results[i], 'AIC.trans', candidates[i]))
                writeResults.write("%6.2f\t" % AIC_trans)  # print the p-value

                BIC_trans = float(Get_value_From_Rlist(results[i], 'BIC.trans', candidates[i]))
                writeResults.write("%6.2f\n" % BIC_trans)  # print the p-value

    print("Done printing Univariable_Transformations")

def iOCT_Print_Results_Univariable(results, save_file, summary_candidates=True):
    if(summary_candidates):
        candidates = Get_Candidates_names(results, None)

        with open(save_file, 'a') as writeResults:

            writeResults.write("candidate\t")
            writeResults.write("gain in -2*loglikelihood\t")
            writeResults.write("p.value\t")
            writeResults.write("AIC\t")
            writeResults.write("BIC\n")

            for i in range(len(candidates)):
                writeResults.write("{}\t".format(candidates[i])) # print the original feature name


                dev_trans = float(Get_value_From_Rlist(results[i], 'dev', candidates[i]))
                writeResults.write("%6.2f\t" % dev_trans) # print the gain in -2*log-likelihood
                #print('Gain in -2logLikelihood: {} (p = {})\n\n'.format(dev, pvalue))


                pvalue_trans = float(Get_value_From_Rlist(results[i], 'p.value', candidates[i]))
                writeResults.write("%6.2f\t" % pvalue_trans) # print the p-value

                AIC_trans = float(Get_value_From_Rlist(results[i], 'AIC', candidates[i]))
                writeResults.write("%6.2f\t" % AIC_trans)  # print the p-value

                BIC_trans = float(Get_value_From_Rlist(results[i], 'BIC', candidates[i]))
                writeResults.write("%6.2f\n" % BIC_trans)  # print the p-value

    print("Done printing Univariable analysis")


def Get_Candidates_names(results, number_summary_print):
    # if (number_summary_print == None):
    #     candidates = results.names[:-1]
    # else:
    #     candidates = results[-1][1:number_summary_print, 1]

    candidates = results.names[:-1]
    return candidates

def Get_value_From_Rlist(RList, name, candidate):
    index = RList.names.index(name) if name in RList.names else -1
    if(index > -1):
        value = RList[index]
        return value[0]
    else:
        print('{} is not element in RList of candidate {}'.format(name, candidate))
        return -1

def Condition(model):
    if(np.isnan(model.bic_llf)):
        return model.bic
    else:
        return model.bic_llf
    
def iOCT_Print_Results_From_List_of_models(modelsList, save_file):
    # modelsList: list of glm models where data will be extracted from
    # save_file: the text file to write information of each model
    
    #modelsList = sorted(modelsList, key=lambda x: x.bic)
    modelsList = sorted(modelsList, key=Condition)
    
    with open(save_file, 'a') as writeResults:
        
        # loop over model
        for model in modelsList:
            
            # write the best selected coeff in this subgroup
            Variables = model.model.exog_names[:]
            Variables.remove("Intercept")
            writeResults.write("Variables: {}\n".format(Variables))
            
            # write the coeff. values
            TempFrame = pd.DataFrame({"Coef":model.params, "P_value":model.pvalues, "STD_Error":model.bse }, index=model.params.index)
            dfAsString = TempFrame.to_string(header=True, index=True)
            writeResults.write(dfAsString)
            writeResults.write("\n\n")
            
            # write BIC and deviance
            writeResults.write("BIC: {}\n".format(model.bic_llf))
            writeResults.write("Deviance: {}\n".format(model.deviance))
            writeResults.write("*"*50)
            writeResults.write("\n")