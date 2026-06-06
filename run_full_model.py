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
        '-e', '--rerun_embedding', action='store_true'
    )

parser.add_argument(
        '-a', '--retrain_AE', action='store_true'
    )

parser.add_argument(
        '-p', '--run_predictions', action='store_true'
    )

parser.add_argument(
        '-aug', '--run_augment_predictions', action='store_true'
    )

parser.add_argument(
        '-pl', '--plot', action='store_true'
    )

parser.add_argument(
        '-s', '--augment_scan', action='store_true'
    )

parser.add_argument(
        '-m', '--model_type', default='transformer'
    )

parser.add_argument(
        '-d', '--dataset', default='C2V'
    )

args = parser.parse_args()

rerun_embedding = args.rerun_embedding
retrain_AE_models = args.retrain_AE
rerun_predictions = args.run_predictions
rerun_augment_predictions = args.run_augment_predictions
plot = args.plot
run_augment_scan = args.augment_scan
Model_type = args.model_type
Dataset = args.dataset

models = ['CAE','QAE','HW_QAE']

######## GLOBAL PARAMETERS for ML Models ############
if Dataset == 'C2V':
    if Model_type == 'MLP':
        embedding_model_path = 'C2V_Contrastive_Embedding/MLP/minbias/ContrastiveEmbedding'
        output_path = 'C2V/MLP/minbias'
    if Model_type == 'transformer':
        embedding_model_path = 'C2V_Contrastive_Embedding/Transformer/minbias/TransformerEmbedding'
        output_path = 'C2V/transformer/minbias'

if Dataset == 'AD': 
    if Model_type == 'MLP':
        embedding_model_path = 'AD_Contrastive_Embedding/MLP/background/ContrastiveEmbedding'
        output_path = 'AD/MLP/background'
    if Model_type == 'transformer':
        embedding_model_path = 'AD_Contrastive_Embedding/Transformer/background/TransformerEmbedding'
        output_path = 'AD/transformer/background'

########## GLOBAL PARAMETERS for Collide 2V ##########

if Dataset == 'C2V':
    train_labels = {"QCD_HT50tobb":2,"QCD_HT50toInf":1,"minbias":0}
    test_labels = {"minbias" : 0, "QCD_HT50toInf" :1, "HH_4b" : 2, 'HH_bbgammagamma':3,'HH_bbtautau':4,'QCD_HT50tobb':5}
    augment_labels = {"minbias" : 0, "HH_4b" : 2, "minbias_augment" : 0, "HH_4b_augment" : 2}


    output_dict = {'CAE' : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}},
                #'SVM'  : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}},
                'QAE' : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}},
                'HW_QAE' : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}}}

    augment_output_dict = {'CAE' : {"minbias" : {}, "HH_4b" : {}, "minbias_augment" : {}, "HH_4b_augment" : {} },
                            #'SVM' : {"minbias" : {}, "HH_4b" : {}, "minbias_augment" : {}, "HH_4b_augment" : {} },
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
                #'SVM' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}, "blackbox": {}},
                'QAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}, "blackbox": {}},
                'HW_QAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}, "blackbox": {}}}

    augment_output_dict = {'CAE' : {"background" : {}, "ato4l" : {}, "background_augment" : {}, "ato4l_augment" : {} },
                            #'SVM'  : {"background" : {}, "ato4l" : {}, "background_augment" : {}, "ato4l_augment" : {} },
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

if rerun_embedding:
    print(f" Rerunning train embedding from: {embedding_model_path}")
    training_columns = []
    dataset_list = []
    for datasets in train_labels.keys():
        data_test = DataSet.fromH5(input_path+datasets+'/train/')
        data_test.normalise()

        data_test.set_label(train_labels[datasets])
        dataset_list.append(data_test)
        training_columns = data_test.training_columns
            
    train_data_frame = pd.concat([dataset.data_frame for dataset in dataset_list])
    train_data_frame = train_data_frame.sample(frac=1)

    train_embeddings = embedding_model.encoder_predict(train_data_frame[training_columns].to_numpy(),training_columns) 
    
    os.makedirs(output_path+'/embeddings/train', exist_ok=True)
    np.save(output_path+'/embeddings/train/train_embeddings.npy',train_embeddings)
    train_data_frame.to_pickle(output_path+'/embeddings/train/dataset.pkl')

train_embeddings = np.load(output_path+'/embeddings/train/train_embeddings.npy')
train_data_frame = pd.read_pickle(output_path+'/embeddings/train/dataset.pkl')

background_indices = np.where(train_data_frame["event_label"] == 0)
background_embeddings = train_embeddings[background_indices]
background_data_frame = train_data_frame.iloc[background_indices]

embedding_max, embedding_min = np.percentile(train_embeddings,100), np.percentile(train_embeddings,0)

if retrain_AE_models:
    
    print("Training CAE")
    
    CAE_model = fromYaml('model/configs/EmbeddingClassicalAEModel.yaml',output_path+'/models/CAE')
    input_shape = background_embeddings.shape[0]
    input_length = len(background_data_frame)
    CAE_model.build_model(input_shape,embedding_min, embedding_max )
    CAE_model.compile_model(input_length)
    CAE_model.only_CAE_fit(background_embeddings)
    CAE_model.save()
    CAE_model.plot_loss()
     
    # print("Training SVM")
    
    # SVM_model = fromYaml('model/configs/SupportVectorMachine.yaml',output_path+'/models/SVM')
    # os.makedirs(output_path+'/models/SVM/plots', exist_ok=True)
    # input_shape = background_embeddings.shape[0]
    # input_length = len(background_data_frame)
    # SVM_model.build_model(input_shape, embedding_min, embedding_max)
    # SVM_model.compile_model(input_length)
    # SVM_model.fit(background_embeddings)
    # SVM_model.save()
    
    print("Training QAE")
    
    QAE_model = fromYaml('model/configs/EmbeddingPennyLaneQAEModel.yaml',output_path+'/models/QAE')
    input_shape = background_embeddings.shape[0]
    input_length = len(background_data_frame)
    input_length = len(background_data_frame)
    QAE_model.build_model(input_shape,embedding_min, embedding_max )
    QAE_model.compile_model(input_length)
    QAE_model.only_QAE_fit(background_embeddings)
    QAE_model.save()
    QAE_model.plot_loss()
    
    print("Training HW QAE")
    
    HW_QAE_model = fromYaml('model/configs/EmbeddingHWPennyLaneQAEModel.yaml',output_path+'/models/HW_QAE')
    input_shape = background_embeddings.shape[0]
    input_length = len(background_data_frame)
    input_length = len(background_data_frame)
    HW_QAE_model.build_model(input_shape,embedding_min, embedding_max )
    HW_QAE_model.compile_model(input_length)
    HW_QAE_model.only_QAE_fit(background_embeddings)
    HW_QAE_model.save()
    HW_QAE_model.plot_loss()
    
CAE_model = fromFolder(output_path+'/models/CAE')
#SVM_model = fromFolder(output_path+'/models/SVM')
QAE_model = fromFolder(output_path+'/models/QAE')
HW_QAE_model = fromFolder(output_path+'/models/HW_QAE')

if rerun_embedding:
    print(f" Rerunning test embedding from: {embedding_model_path}")
    
    training_columns = []
    dataset_list = []
    for datasets in test_labels.keys():
        data_test = DataSet.fromH5(input_path+datasets+'/test/')
        data_test.normalise()

        data_test.set_label(test_labels[datasets])
        dataset_list.append(data_test)
        training_columns = data_test.training_columns
            
    test_data_frame = pd.concat([dataset.data_frame for dataset in dataset_list])
    test_data_frame = test_data_frame.sample(frac=1)

    test_embeddings = embedding_model.encoder_predict(test_data_frame[training_columns].to_numpy(),training_columns) 
    
    os.makedirs(output_path+'/embeddings/test', exist_ok=True)
    np.save(output_path+'/embeddings/test/test_embeddings.npy',test_embeddings)
    test_data_frame.to_pickle(output_path+'/embeddings/test/dataset.pkl')
    
test_embeddings = np.load(output_path+'/embeddings/test/test_embeddings.npy')
test_data_frame = pd.read_pickle(output_path+'/embeddings/test/dataset.pkl')

if rerun_predictions:
    
    print("Running testing")

    for label in test_labels.keys():
        print("==== Predicting for "+label+" ====")
        indices = np.where(test_data_frame["event_label"] == test_labels[label])
        test_index = np.random.randint(0, len(indices[0]), size=10000)
        embeddings = test_embeddings[indices[0][test_index]]
        data_frame = test_data_frame.iloc[indices[0][test_index]]

        print("Classical Autoencoder Predict")
        output_dict['CAE'][label] = {'predictions' : CAE_model.only_CAE_predict(embeddings)}
        # print("Support Vector Machine Predict")
        # output_dict['SVM'][label]  = {'predictions' : SVM_model.predict(embeddings)}
        print("Quantum Autoencoder Predict")
        output_dict['QAE'][label] = {'predictions' : QAE_model.only_QAE_predict(embeddings)}
        print("HW Quantum Autoencoder Predict")
        output_dict['HW_QAE'][label] = {'predictions' : HW_QAE_model.only_QAE_predict(embeddings)}    
    
    with open(output_path+'/models/output_dict.pkl', 'wb') as f:
        pickle.dump(output_dict, f)
        
with open(output_path+'/models/output_dict.pkl', 'rb') as f:
    output_dict = pickle.load(f)
    
    
if plot:
    print("Plotting output histograms")
    
    for model in models:
    
        plot_histo([output_dict[model][dataset]['predictions'] for dataset in output_dict[model].keys()], 
                [dataset for dataset in output_dict[model].keys()], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
        os.makedirs(f"{output_path}/models/{model}/plots/testing/", exist_ok=True)
        plt.savefig(f"{output_path}/models/{model}/plots/testing/outputscores.png", bbox_inches='tight')
        plt.close() 

    
    print("Plotting output ROCs")
    
    for model in models:
        
        ROC_curve([output_dict[model][background_name]['predictions'] for dataset in output_dict[model].keys()],
                [output_dict[model][dataset]['predictions'] for dataset in output_dict[model].keys()],
                test_labels)
        plt.savefig(f"{output_path}/models/{model}/plots/testing/ROC.png", bbox_inches='tight')
        plt.close() 

    
    for label in test_labels.keys():
        
        ROC_curve([output_dict[model][background_name]['predictions'] for model in models],
                  [output_dict[model][label]['predictions'] for model in models],
                  models)
        os.makedirs(f"{output_path}/plots/testing/", exist_ok=True)
        plt.savefig(f"{output_path}/plots/testing/{label}_ROC.png", bbox_inches='tight')
        plt.close() 
        
        
if rerun_augment_predictions and rerun_embedding:
    
    print(f" Rerunning augment embedding from: {embedding_model_path}")
    
    training_columns = []
    dataset_list = []

    for datasets in test_labels.keys():
        data_test = DataSet.fromH5(input_path+datasets+'/test/')
        data_test_dataframe = data_test.data_frame.sample(n=10000)
        
        #augment_test.drop_a_soft_one('jet')
        data_test.eta_smear()
        data_test.pt_smear()
        data_test.phi_smear()
        #data_test.phi_rotate()
        
        data_test.normalise()

        data_test.set_label(test_labels[datasets])
        dataset_list.append(data_test)
        training_columns = data_test.training_columns
            
    augment_data_frame = pd.concat([dataset.data_frame for dataset in dataset_list])
    augment_data_frame = augment_data_frame.sample(frac=1)

    augment_embeddings = embedding_model.encoder_predict(augment_data_frame[training_columns].to_numpy(),training_columns) 
    
    os.makedirs(output_path+'/embeddings/augment', exist_ok=True)
    np.save(output_path+'/embeddings/augment/augment_embeddings.npy',augment_embeddings)
    augment_data_frame.to_pickle(output_path+'/embeddings/augment/dataset.pkl')
    
augment_embeddings = np.load(output_path+'/embeddings/augment/augment_embeddings.npy')
augment_data_frame = pd.read_pickle(output_path+'/embeddings/augment/dataset.pkl')

if plot:
    
    print("Plotting augment latent spaces")
    
    background_indices = np.where(test_data_frame["event_label"] == 0)
    test_index = np.random.randint(0, len(background_indices[0]), size=60000)
    background_test_embeddings = test_embeddings[background_indices[0][test_index]]
    background_test_data_frame = test_data_frame.iloc[background_indices[0][test_index]]

    background_indices = np.where(augment_data_frame["event_label"] == 0)
    test_index = np.random.randint(0, len(background_indices[0]), size=60000)
    background_augment_embeddings = augment_embeddings[background_indices[0][test_index]]
    background_augment_data_frame = augment_data_frame.iloc[background_indices[0][test_index]]
        
    latent_vector = np.concatenate([background_test_embeddings,background_augment_embeddings], axis=0)
    event_label_vector = np.concatenate([np.zeros(len(background_test_embeddings)),np.ones(len(background_augment_embeddings))], axis=0)

    os.makedirs(f"{output_path}/embeddings/plots/{background_name}", exist_ok=True)

    plot_latent(latent_vector, event_label_vector, {'test':'test','augment':'augment'}, f"{output_path}/embeddings/plots/{background_name}")
        
    pca = PCA(n_components=2)
    event_principle_components = pca.fit_transform(latent_vector)

    event_xmax, event_xmin = np.percentile(event_principle_components[:,0],99), np.percentile(event_principle_components[:,0],1)
    event_ymax, event_ymin = np.percentile(event_principle_components[:,1],99), np.percentile(event_principle_components[:,1],1)
        
    plot_PCA(event_principle_components, event_label_vector,((event_xmin, event_xmax),(event_ymin, event_ymax)), {'test':'test','augment':'augment'}, f"{output_path}/embeddings/plots/{background_name}")

    hh4b_indices = np.where(test_data_frame["event_label"] == 2)
    test_index = np.random.randint(0, len(hh4b_indices[0]), size=60000)
    hh4b_test_embeddings = test_embeddings[hh4b_indices[0][test_index]]

    hh4b_indices = np.where(augment_data_frame["event_label"] == 2)
    test_index = np.random.randint(0, len(hh4b_indices[0]), size=60000)
    hh4b_augment_embeddings = augment_embeddings[hh4b_indices[0][test_index]]
        
    latent_vector = np.concatenate([hh4b_test_embeddings,hh4b_augment_embeddings], axis=0)
    event_label_vector = np.concatenate([np.zeros(len(hh4b_test_embeddings)),np.ones(len(hh4b_augment_embeddings))], axis=0)

    os.makedirs(f"{output_path}/embeddings/plots/{signal_name}", exist_ok=True)

    plot_latent(latent_vector, event_label_vector, {'test':'test','augment':'augment'}, f"{output_path}/embeddings/plots/{signal_name}")
        
    pca = PCA(n_components=2)
    event_principle_components = pca.fit_transform(latent_vector)

    event_xmax, event_xmin = np.percentile(event_principle_components[:,0],99), np.percentile(event_principle_components[:,0],1)
    event_ymax, event_ymin = np.percentile(event_principle_components[:,1],99), np.percentile(event_principle_components[:,1],1)
        
    plot_PCA(event_principle_components, event_label_vector,((event_xmin, event_xmax),(event_ymin, event_ymax)), {'test':'test','augment':'augment'}, f"{output_path}/embeddings/plots/{signal_name}")

if rerun_augment_predictions:
    
    print("Rerunning augment predictions")
    
    for label in augment_labels.keys():
        print("==== Predicting for "+label+" ====")
        
        if 'augment' in label:
            indices = np.where(augment_data_frame["event_label"] == augment_labels[label])
            test_index = np.random.randint(0, len(indices[0]), size=10000)
            embeddings = augment_embeddings[indices[0][test_index]]
        else:
            indices = np.where(test_data_frame["event_label"] == augment_labels[label])
            test_index = np.random.randint(0, len(indices[0]), size=10000)
            embeddings = test_embeddings[indices[0][test_index]]
            

        print("Classical Autoencoder Predict")
        augment_output_dict['CAE'][label] = {'predictions' : CAE_model.only_CAE_predict(embeddings)}
        # print("Support Vector Machine Predict")
        # augment_output_dict['SVM'][label]  = {'predictions' : SVM_model.predict(embeddings)}
        print("Quantum Autoencoder Predict")
        augment_output_dict['QAE'][label] = {'predictions' : QAE_model.only_QAE_predict(embeddings)}
        print("HW Quantum Autoencoder Predict")
        augment_output_dict['HW_QAE'][label] = {'predictions' : HW_QAE_model.only_QAE_predict(embeddings)}  
        
    with open(output_path+'/models/output_augment_dict.pkl', 'wb') as f:
        pickle.dump(augment_output_dict, f)

with open(output_path+'/models/output_augment_dict.pkl', 'rb') as f:
    augment_output_dict = pickle.load(f)
    
if plot:
    print("Plotting augment output histograms")
    
    for model in models:
        plot_histo([augment_output_dict[model][dataset]['predictions'] for dataset in augment_output_dict[model].keys()], 
                [dataset for dataset in augment_output_dict[model].keys()], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
        os.makedirs(f"{output_path}/models/{model}/plots/augment/testing/", exist_ok=True)
        plt.savefig(f"{output_path}/models/{model}/plots/augment/testing/outputscores.png", bbox_inches='tight')
        plt.close() 

    
    os.makedirs(output_path+"/plots/augment/testing/", exist_ok=True)
    for label in augment_labels.keys():
        plot_histo([augment_output_dict[model][label]['predictions'] for model in models], 
                models, 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
        plt.savefig(f"{output_path}/plots/augment/testing/{label}_outputscores.png", bbox_inches='tight')
        plt.close()  
    
    print("Plotting augment ROCs")
    
    for model in models:
        
        ROC_curve([augment_output_dict[model][background_name]['predictions'] for dataset in augment_output_dict[model].keys()],
            [augment_output_dict[model][dataset]['predictions'] for dataset in augment_output_dict[model].keys()],
            augment_labels)
        plt.savefig(f"{output_path}/models/{model}/plots/augment/testing/ROC.png", bbox_inches='tight')
        plt.close() 

    
    
    for label in augment_labels.keys():
        if 'augment' in label:
            ROC_curve([augment_output_dict[model][background_augment_name]['predictions'] for model in models],
                      [augment_output_dict[model][label]['predictions'] for model in models],
                      models)
            
        else:
            ROC_curve([augment_output_dict[model][background_name]['predictions'] for model in models],
                      [augment_output_dict[model][label]['predictions'] for model in models],
                      models)
        
        plt.savefig(f"{output_path}/plots/augment/testing/{label}_ROC.png", bbox_inches='tight')
        plt.close() 
        
        
for model in models:
    background = augment_output_dict[model][background_name]['predictions']
    background_a = augment_output_dict[model][background_augment_name]['predictions']

    sig = augment_output_dict[model][signal_name]['predictions']
    sig_a = augment_output_dict[model][signal_augment_name]['predictions']
    
    background_wd = scipy.stats.wasserstein_distance(background,background_a)
    sig_wd = scipy.stats.wasserstein_distance(sig,sig_a)
    
    auc_loss_list = ROC_curve([background, background_a],
                              [sig,     sig_a],
                              [background_name,signal_name],plot=False)
    
    print(model)
    print(f"Wasserstein Distance for background samples {background_wd}")
    print(f"Wasserstein Distance for Signal sample {sig_wd}")
    print(f"Non Augmented ROC {auc_loss_list[0]}")
    print(f"Augmented ROC {auc_loss_list[1]}")
    
    
if run_augment_scan:
    
    num_samples = 2000
    num_CV = 10
    print("Running augment scan")
    
    os.makedirs(output_path+"/augment/", exist_ok=True)
    f = open(output_path+"/augment/scan_classical_only.csv",'w')
    f.write(f"model,smear_percent,auc_loss_non_augmented,auc_loss_non_augmented_err,auc_loss_augmented ,auc_loss_augmented_err,embedding_wd, embedding_wd_err,background_wd,background_wd_err,signal_wd,signal_wd_err\n")

    non_augmented_model_results = {background_name: {'CAE':[],#'SVM':[],
                                                     'QAE':[],'HW_QAE':[],
                                                     'embeddings':[]},
                                   signal_name: {'CAE':[],#'SVM':[],
                                                 'QAE':[],'HW_QAE':[],
                                                 'embeddings':[]}}
    
    background_test = DataSet.fromH5(input_path+background_name+'/test/')
    
    signal_test = DataSet.fromH5(input_path+signal_name+'/test/')
    
    dataset_dict = {background_name: background_test, signal_name: signal_test}
    
    for iCV in range(num_CV):
        start_time = time.time()
        for datasets in [background_name,signal_name]:
                temp_dataset = dataset_dict[datasets]
                temp_dataset.normalise()
                data_test_dataframe = temp_dataset.data_frame
                
                temp_dataset.set_label(test_labels[datasets])

                embeddings = embedding_model.encoder_predict(data_test_dataframe[iCV*num_samples:(iCV+1)*num_samples],temp_dataset.training_columns) 

                non_augmented_model_results[datasets]['CAE'].append(CAE_model.only_CAE_predict(embeddings))
                #non_augmented_model_results[datasets]['SVM'].append(SVM_model.predict(embeddings))
                non_augmented_model_results[datasets]['QAE'].append(QAE_model.only_QAE_predict(embeddings))
                non_augmented_model_results[datasets]['HW_QAE'].append(HW_QAE_model.only_QAE_predict(embeddings))  
                non_augmented_model_results[datasets]['embeddings'].append(embeddings)
                
                gc.collect()
                
        print(f"Cross validation {iCV} background finishing in {time.time() - start_time} s")

    for smear_percent in [0.0001,0.001,0.01,0.1,1.0]:
        
        augmented_model_results = {background_name:{'CAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],'WD_signal':[],'WD_background':[]},
                                                    #'SVM':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],'WD_signal':[],'WD_background':[]},
                                                    'QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],'WD_signal':[],'WD_background':[]},
                                                    'HW_QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],'WD_signal':[],'WD_background':[]},
                                                    'embeddings':{'embeddings':0,'WD':[]}},
                                   signal_name:{'CAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],'WD_signal':[],'WD_background':[]},
                                                #'SVM':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],'WD_signal':[],'WD_background':[]},
                                                'QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],'WD_signal':[],'WD_background':[]},
                                                'HW_QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],'WD_signal':[],'WD_background':[]},
                                                'embeddings':{'embeddings':0,'WD':[]}}}
        
        for iCV in range(num_CV): 
            start_time = time.time()
            for datasets in [background_name,signal_name]:
                temp_dataset =  dataset_dict[datasets]
                #augment_test.drop_a_soft_one('jet')
                temp_dataset.eta_smear(smear_percent)
                temp_dataset.pt_smear(smear_percent)
                temp_dataset.phi_smear(smear_percent)
                #data_test.phi_rotate()
                temp_dataset.normalise()
                
                data_test_dataframe = temp_dataset.data_frame

                temp_dataset.set_label(test_labels[datasets])
                print(f"Data set loading {iCV} {datasets} {smear_percent} finishing in {time.time() - start_time} s")
                start_time = time.time()
                augmented_embeddings = embedding_model.encoder_predict(data_test_dataframe[iCV*num_samples:(iCV+1)*num_samples],temp_dataset.training_columns) 
                print(f"Embedding predict {iCV} {datasets} finishing in {time.time() - start_time} s")
                start_time = time.time()
                print(augmented_embeddings.shape)
                augmented_model_results[datasets]['CAE']['predictions'] = (CAE_model.only_CAE_predict(augmented_embeddings))
                print(f"CAE predict {iCV} {datasets} finishing in {time.time() - start_time} s")
                start_time = time.time()
                # augmented_model_results[datasets]['SVM']['predictions'] = (SVM_model.predict(augmented_embeddings))
                # print(f"SVM predict {iCV} {datasets} finishing in {time.time() - start_time} s")
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
                auc_loss_list = ROC_curve([non_augmented_model_results[background_name][model][iCV], augmented_model_results[background_name][model]['predictions']],
                                        [non_augmented_model_results[signal_name][model][iCV],     augmented_model_results[signal_name][model]['predictions']],
                                        [background_name,signal_name],plot=False)
                
                augmented_model_results[background_name][model]['augmented_ROC'].append( auc_loss_list[1])
                augmented_model_results[background_name][model]['non_augmented_ROC'].append( auc_loss_list[0])
                augmented_model_results[background_name][model]['WD_signal'].append( sig_wd)
                augmented_model_results[background_name][model]['WD_background'].append( background_wd )
            
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
            
            
            print(model)
            print(smear_percent)
            print(f"Non Augmented ROC {non_augment_roc} +/- {non_augment_roc_err}")
            print(f"Augmented ROC {augment_roc} +/- {augment_roc_err}")
            print("Wasserstein Distance")
            print(f"Background sample {background_WD} +/- {background_WD_err}")
            print(f"Signal sample {signal_WD} +/- {signal_WD_err}")
            print("============")
                    
            f.write(f"{model},{smear_percent},"
                    f"{non_augment_roc},"
                    f"{non_augment_roc_err},"
                    f"{augment_roc} , "
                    f"{augment_roc_err},"
                    f"{background_WD}, " 
                    f"{background_WD_err} , "
                    f"{signal_WD} , "
                    f"{signal_WD_err}\n")
    
    f.close()
            
        
    
