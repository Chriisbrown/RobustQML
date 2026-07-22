import os
import pickle
import scipy
# Third parties
import numpy as np
import pandas as pd
# Import from other modules
from model.common import fromYaml
from data.EOSdataset import DataSet
from model.common import fromFolder

from model.gpu_utils import setup_gpu_memory_growth
from plot.basic import error_residual, plot_histo, rates,efficiency, clusters, plot_2d,ROC_curve
from plot.plot_latent import plot_latent, plot_PCA
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

from argparse import ArgumentParser

import matplotlib.pyplot as plt

import time

import gc


parser = ArgumentParser()
    

parser.add_argument(
        '-m', '--model_type', default='transformer'
    )

parser.add_argument(
        '-d', '--dataset', default='C2V'
    )

args = parser.parse_args()

Model_type = args.model_type
Dataset = args.dataset

models = ['CAE','QAE','HW_QAE']

######## GLOBAL PARAMETERS for ML Models ############
if Dataset == 'C2V':
    if Model_type == 'MLP':
        embedding_model_path = 'embeddings/C2V/minbias/MLP/'
        output_path = 'AnomalyDetection/C2V/minbias/MLP'
    if Model_type == 'transformer':
        embedding_model_path = 'embeddings/C2V/minbias/Transformer/'
        output_path = 'AnomalyDetection/C2V/minbias/Transformer'

if Dataset == 'AD': 
    if Model_type == 'MLP':
        embedding_model_path = 'embeddings/AD/background/MLP'
        output_path = 'AnomalyDetection/AD/background/MLP'
    if Model_type == 'transformer':
        embedding_model_path = 'embeddings/AD/background/Transformer'
        output_path = 'AnomalyDetection/AD/background/Transformer'

########## GLOBAL PARAMETERS for Collide 2V ##########

if Dataset == 'C2V':
    train_labels = {"QCD_HT50tobb":2,"QCD_HT50toInf":1,"minbias":0}
    test_labels = {"minbias" : 0, "QCD_HT50toInf" :1, "HH_4b" : 2, 'HH_bbgammagamma':3,'HH_bbtautau':4,'QCD_HT50tobb':5}
    augment_labels = {"minbias" : 0, "HH_4b" : 2, "minbias_augment" : 0, "HH_4b_augment" : 2}


    output_dict = {'CAE' : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}},
                   'QAE' : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}},
                   'HW_QAE' : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}}}

    augment_output_dict = {'CAE' : {"minbias" : {}, "HH_4b" : {}, "minbias_augment" : {}, "HH_4b_augment" : {} },
                           'QAE' : {"minbias" : {}, "HH_4b" : {}, "minbias_augment" : {}, "HH_4b_augment" : {} },
                           'HW_QAE' : {"minbias" : {}, "HH_4b" : {}, "minbias_augment" : {}, "HH_4b_augment" : {} }}

    input_path = '/scratch/RobustQML_Datasets/C2V_dataset'

    background_name = 'minbias'
    background_augment_name = background_name + '_augment'

    signal_name = 'HH_4b'
    signal_augment_name = signal_name + '_augment'

##################

########## GLOBAL PARAMETERS for Collide AD ##########
if Dataset == 'AD':

    train_labels = {"background" : 0, "ato4l" :1, "hChToTauNu" : 2, 'hToTauTau':3,'leptoquark':4}
    test_labels =  {"background" : 0, "ato4l" :1, "hChToTauNu" : 2, 'hToTauTau':3,'leptoquark':4, 'blackbox':5}
    augment_labels = {"background" : 0, "ato4l" :1, "background_augment":0, "ato4l_augment":1}

    output_dict = {'CAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}, "blackbox": {}},
                   'QAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}, "blackbox": {}},
                   'HW_QAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}, "blackbox": {}}}

    augment_output_dict = {'CAE' : {"background" : {}, "ato4l" : {}, "background_augment" : {}, "ato4l_augment" : {} },
                           'QAE' : {"background" : {}, "ato4l" : {}, "background_augment" : {}, "ato4l_augment" : {} },
                           'HW_QAE' : {"background" : {}, "ato4l" : {}, "background_augment" : {}, "ato4l_augment" : {} }}

    input_path = '/scratch/RobustQML_Datasets/AD_dataset/'

    background_name = 'background'
    background_augment_name = background_name + '_augment'

    signal_name = 'ato4l'
    signal_augment_name = signal_name + '_augment'

##################


os.makedirs(output_path, exist_ok=True)

setup_gpu_memory_growth()
embedding_model = fromFolder(embedding_model_path)
    
CAE_model = fromFolder(output_path+'/models/CAE')
QAE_model = fromFolder(output_path+'/models/QAE')
HW_QAE_model = fromFolder(output_path+'/models/HW_QAE')
    
num_samples = 6000
num_CV = 10
print("Running augment scan")
    
os.makedirs(output_path+"/augment/", exist_ok=True)
f = open(output_path+"/augment/scan_classical_only.csv",'w')
f.write(f"model,smear_percent,auc_loss_non_augmented,auc_loss_non_augmented_err,auc_loss_augmented ,auc_loss_augmented_err,background_wd,background_wd_err,signal_wd,signal_wd_err,background_mse,background_mse_err,signal_mse,signal_mse_err,background_mae,background_mae_err,signal_mae,signal_mae_err,background_cos,background_cos_err,signal_cos,signal_cos_err,background_euc,background_euc_err,signal_euc,signal_euc_err\n")

non_augmented_model_results = {background_name: {'CAE':[],
                                                 'QAE':[],'HW_QAE':[],
                                                 'embeddings':[]},
                                   signal_name: {'CAE':[],
                                                 'QAE':[],'HW_QAE':[],
                                                 'embeddings':[]}}
    
    

for iCV in range(num_CV):
    start_time = time.time()
    for datasets in [background_name,signal_name]:  
        temp_dataset = DataSet.fromH5(input_path+datasets+'/test/')
        temp_dataset.normalise()
        data_test_dataframe = temp_dataset.data_frame
        temp_dataset.set_label(test_labels[datasets])
        
        embeddings = embedding_model.encoder_predict(data_test_dataframe[iCV*num_samples:(iCV+1)*num_samples],temp_dataset.training_columns) 

        non_augmented_model_results[datasets]['CAE'].append(CAE_model.only_CAE_predict(embeddings))
        non_augmented_model_results[datasets]['QAE'].append(QAE_model.only_QAE_predict(embeddings))
        non_augmented_model_results[datasets]['HW_QAE'].append(HW_QAE_model.only_QAE_predict(embeddings))  
        non_augmented_model_results[datasets]['embeddings'].append(embeddings)
                
    print(f"Cross validation {iCV} background finishing in {time.time() - start_time} s")

for smear_percent in [0.0001,0.001,0.01,0.1,1.0]:
    augmented_model_results = {background_name:{'CAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                       'WD_signal':[],'WD_background':[],
                                                       'MSE_signal':[],'MSE_background':[],
                                                       'MAE_signal':[],'MAE_background':[],
                                                       'cos_signal':[],'cos_background':[],
                                                       'euc_signal':[],'euc_background':[]},
                                                'QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                       'WD_signal':[],'WD_background':[],
                                                       'MSE_signal':[],'MSE_background':[],
                                                       'MAE_signal':[],'MAE_background':[],
                                                       'cos_signal':[],'cos_background':[],
                                                       'euc_signal':[],'euc_background':[]},
                                                'HW_QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                          'WD_signal':[],'WD_background':[],
                                                       'MSE_signal':[],'MSE_background':[],
                                                       'MAE_signal':[],'MAE_background':[],
                                                       'cos_signal':[],'cos_background':[],
                                                       'euc_signal':[],'euc_background':[]},},                                
                               signal_name:{'CAE':{ 'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                    'WD_signal':[],'WD_background':[],
                                                    'MSE_signal':[],'MSE_background':[],
                                                    'MAE_signal':[],'MAE_background':[],
                                                    'cos_signal':[],'cos_background':[],
                                                    'euc_signal':[],'euc_background':[]},
                                             'QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                    'WD_signal':[],'WD_background':[],
                                                    'MSE_signal':[],'MSE_background':[],
                                                    'MAE_signal':[],'MAE_background':[],
                                                    'cos_signal':[],'cos_background':[],
                                                    'euc_signal':[],'euc_background':[]},
                                              'HW_QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                        'WD_signal':[],'WD_background':[],
                                                       'MSE_signal':[],'MSE_background':[],
                                                       'MAE_signal':[],'MAE_background':[],
                                                       'cos_signal':[],'cos_background':[],
                                                       'euc_signal':[],'euc_background':[]},
                                              }}
        
    for iCV in range(num_CV): 
        start_time = time.time()
        for datasets in [background_name,signal_name]:
            temp_dataset = DataSet.fromH5(input_path+datasets+'/test/')

            temp_data_frame = temp_dataset.data_frame[iCV*num_samples:(iCV+1)*num_samples]
            temp_dataset.data_frame = temp_data_frame
            # temp_dataset.drop_a_soft_one('jet',pt_threshold[0])
            # temp_dataset.drop_a_soft_one('e',pt_threshold[1])
            # temp_dataset.drop_a_soft_one('mu',pt_threshold[2])
            temp_dataset.eta_smear(smear_percent)
            temp_dataset.pt_smear(smear_percent)
            temp_dataset.phi_smear(smear_percent)
            #temp_dataset.eta_flip()
            temp_dataset.normalise()
                
            data_test_dataframe = temp_dataset.data_frame

            temp_dataset.set_label(test_labels[datasets])
            start_time = time.time()
            augmented_embeddings = embedding_model.encoder_predict(data_test_dataframe,temp_dataset.training_columns) 
            print(f"Embedding predict {iCV} {datasets} finishing in {time.time() - start_time} s")
            start_time = time.time()
            augmented_model_results[datasets]['CAE']['predictions'] = (CAE_model.only_CAE_predict(augmented_embeddings))
            print(f"CAE predict {iCV} {datasets} finishing in {time.time() - start_time} s")
            start_time = time.time()
            augmented_model_results[datasets]['QAE']['predictions'] = (QAE_model.only_QAE_predict(augmented_embeddings))
            print(f"QAE predict {iCV} {datasets} finishing in {time.time() - start_time} s")
            start_time = time.time()
            augmented_model_results[datasets]['HW_QAE']['predictions'] = (HW_QAE_model.only_QAE_predict(augmented_embeddings))  
            print(f"HW QAE predict {iCV} {datasets} finishing in {time.time() - start_time} s")
            gc.collect()
            
        for model in non_augmented_model_results[background_name].keys():
            if model == 'embeddings':
                continue
            start_time = time.time()
            background_wd = scipy.stats.wasserstein_distance(non_augmented_model_results[background_name][model][iCV],augmented_model_results[background_name][model]['predictions'])
            sig_wd = scipy.stats.wasserstein_distance(non_augmented_model_results[signal_name][model][iCV],augmented_model_results[signal_name][model]['predictions'])
            
            background_mse = root_mean_squared_error(non_augmented_model_results[background_name][model][iCV],augmented_model_results[background_name][model]['predictions'])
            sig_mse = root_mean_squared_error(non_augmented_model_results[signal_name][model][iCV],augmented_model_results[signal_name][model]['predictions'])
                
            background_mae = mean_absolute_error(non_augmented_model_results[background_name][model][iCV],augmented_model_results[background_name][model]['predictions'])
            sig_mae = mean_absolute_error(non_augmented_model_results[signal_name][model][iCV],augmented_model_results[signal_name][model]['predictions'])

            background_cos = distance.cosine(non_augmented_model_results[background_name][model][iCV],augmented_model_results[background_name][model]['predictions'])
            sig_cos = distance.cosine(non_augmented_model_results[signal_name][model][iCV],augmented_model_results[signal_name][model]['predictions'])
                
            background_euc = distance.euclidean(non_augmented_model_results[background_name][model][iCV],augmented_model_results[background_name][model]['predictions'])
            sig_euc = distance.euclidean(non_augmented_model_results[signal_name][model][iCV],augmented_model_results[signal_name][model]['predictions'])
                
                
            auc_loss_list = ROC_curve([non_augmented_model_results[background_name][model][iCV], augmented_model_results[background_name][model]['predictions']],
                                        [non_augmented_model_results[signal_name][model][iCV],     augmented_model_results[signal_name][model]['predictions']],
                                        [background_name,signal_name],plot=False)
                
            augmented_model_results[background_name][model]['augmented_ROC'].append( auc_loss_list[1])
            augmented_model_results[background_name][model]['non_augmented_ROC'].append( auc_loss_list[0])
            augmented_model_results[background_name][model]['WD_signal'].append( sig_wd)
            augmented_model_results[background_name][model]['WD_background'].append( background_wd )
            augmented_model_results[background_name][model]['MSE_signal'].append( sig_mse)
            augmented_model_results[background_name][model]['MSE_background'].append( background_mse )
            augmented_model_results[background_name][model]['MAE_signal'].append( sig_mae)
            augmented_model_results[background_name][model]['MAE_background'].append( background_mae )
            augmented_model_results[background_name][model]['cos_signal'].append( sig_cos)
            augmented_model_results[background_name][model]['cos_background'].append( background_cos )
            augmented_model_results[background_name][model]['euc_signal'].append( sig_euc)
            augmented_model_results[background_name][model]['euc_background'].append( background_euc )
            
    for model in non_augmented_model_results[background_name].keys():   
        if model == 'embeddings':
                continue 
                
        non_augment_roc = sum(augmented_model_results[background_name][model]['non_augmented_ROC']) /num_CV
        non_augment_roc_err = np.std(augmented_model_results[background_name][model]['non_augmented_ROC'])
        augment_roc = sum(augmented_model_results[background_name][model]['augmented_ROC'])/num_CV
        augment_roc_err = np.std(augmented_model_results[background_name][model]['augmented_ROC']) 
        
        background_WD = sum(augmented_model_results[background_name][model]['WD_background'])/num_CV
        background_WD_err = np.std(augmented_model_results[background_name][model]['WD_background'])
        signal_WD = sum(augmented_model_results[background_name][model]['WD_signal'])/num_CV
        signal_WD_err = np.std(augmented_model_results[background_name][model]['WD_signal'])
            
        background_MSE = sum(augmented_model_results[background_name][model]['MSE_background'])/num_CV
        background_MSE_err = np.std(augmented_model_results[background_name][model]['MSE_background'])
        signal_MSE = sum(augmented_model_results[background_name][model]['MSE_signal'])/num_CV
        signal_MSE_err = np.std(augmented_model_results[background_name][model]['MSE_signal'])
            
        background_MAE = sum(augmented_model_results[background_name][model]['MAE_background'])/num_CV
        background_MAE_err = np.std(augmented_model_results[background_name][model]['MAE_background'])
        signal_MAE = sum(augmented_model_results[background_name][model]['MAE_signal'])/num_CV
        signal_MAE_err = np.std(augmented_model_results[background_name][model]['MAE_signal'])
            
        background_cos = sum(augmented_model_results[background_name][model]['cos_background'])/num_CV
        background_cos_err = np.std(augmented_model_results[background_name][model]['cos_background'])
        signal_cos = sum(augmented_model_results[background_name][model]['cos_signal'])/num_CV
        signal_cos_err = np.std(augmented_model_results[background_name][model]['cos_signal'])
            
        background_euc = sum(augmented_model_results[background_name][model]['euc_background'])/num_CV
        background_euc_err = np.std(augmented_model_results[background_name][model]['euc_background'])
        signal_euc = sum(augmented_model_results[background_name][model]['euc_signal'])/num_CV
        signal_euc_err = np.std(augmented_model_results[background_name][model]['euc_signal'])
              
            
        print(model)
        print(smear_percent)
        print(f"Non Augmented ROC {non_augment_roc} +/- {non_augment_roc_err}")
        print(f"Augmented ROC {augment_roc} +/- {augment_roc_err}")
        print("Wasserstein Distance")
        print(f"Background sample {background_WD} +/- {background_WD_err}")
        print(f"Signal sample {signal_WD} +/- {signal_WD_err}")
        print("============")
        
        print("RMSE")
        print(f"Background sample {background_MSE} +/- {background_MSE_err}")
        print(f"Signal sample {signal_MSE} +/- {signal_MSE_err}")
        print("============")
            
        print("MAE")
        print(f"Background sample {background_MAE} +/- {background_MAE_err}")
        print(f"Signal sample {signal_MAE} +/- {signal_MAE_err}")
        print("============")
            
        print("Cosine similarity")
        print(f"Background sample {background_cos} +/- {background_cos_err}")
        print(f"Signal sample {signal_cos} +/- {signal_cos_err}")
        print("============")
            
        print("Euclidean Distance")
        print(f"Background sample {background_euc} +/- {background_euc_err}")
        print(f"Signal sample {signal_euc} +/- {signal_euc_err}")
        print("============")
                    
        f.write(f"{model},{smear_percent},"
                    f"{non_augment_roc},"
                    f"{non_augment_roc_err},"
                    f"{augment_roc} , "
                    f"{augment_roc_err},"
                    f"{background_WD}, " 
                    f"{background_WD_err} , "
                    f"{signal_WD} , "
                    f"{signal_WD_err}, "
                    f"{background_MSE}, " 
                    f"{background_MSE_err} , "
                    f"{signal_MSE} , "
                    f"{signal_MSE_err}, "
                    f"{background_MAE}, " 
                    f"{background_MAE_err} , "
                    f"{signal_MAE} , "
                    f"{signal_MAE_err}, "
                    f"{background_cos}, " 
                    f"{background_cos_err} , "
                    f"{signal_cos} , "
                    f"{signal_cos_err}, "
                    f"{background_euc}, " 
                    f"{background_euc_err} , "
                    f"{signal_euc} , "
                    f"{signal_euc_err}\n")
    
f.close()
            
        
    
