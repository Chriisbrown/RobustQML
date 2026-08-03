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
from sklearn.metrics import roc_curve, auc,root_mean_squared_error, mean_absolute_error

import scipy.spatial.distance as distance

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



args = parser.parse_args()

rerun_embedding = args.rerun_embedding
retrain_AE_models = args.retrain_AE
rerun_predictions = args.run_predictions
rerun_augment_predictions = args.run_augment_predictions
plot = args.plot
run_augment_scan = args.augment_scan


models = ['CAE','VAE','QAE','HW_QAE']

######## GLOBAL PARAMETERS for ML Models ############

embedding_model_path = 'embeddings/AD/background/MLP'
output_path = 'AnomalyDetection/AD/background/MLP'


########## GLOBAL PARAMETERS for  AD ##########

background_name = 'background'
background_augment_name = background_name + '_augment'

signal_name = 'ato4l'
signal_augment_name = signal_name + '_augment'

train_labels = {"ato4l" :1, "hChToTauNu" : 2, 'hToTauTau':3,'leptoquark':4,"background" : 0,}
test_labels =  {"ato4l" :1, "hChToTauNu" : 2, 'hToTauTau':3,'leptoquark':4,"background" : 0,}
augment_labels = {background_name : 0, signal_name :1, background_augment_name:0, signal_augment_name:1}

output_dict = {'CAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}},
                   'VAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}},
                   'QAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}},
                   'HW_QAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}}}

augment_output_dict = {'CAE' : {background_name : {},signal_name : {}, background_augment_name : {}, signal_augment_name : {} },
                           'VAE' : {background_name : {},signal_name : {}, background_augment_name : {}, signal_augment_name : {} },
                           'QAE' : {background_name : {},signal_name : {}, background_augment_name : {}, signal_augment_name : {} },
                           'HW_QAE' : {background_name : {},signal_name : {}, background_augment_name : {}, signal_augment_name : {} }}


fancy_dataset = {
    'background' : 'background',
    'ato4l': r'$A \to 4\ell$',
    'hChToTauNu': r'$H^{\pm} \to \tau\nu$',
    'hToTauTau': r'$H \to \tau\tau$',
    'leptoquark': 'leptoquark'
}

input_path = '/scratch/RobustQML_Datasets/AD_dataset/'


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
    print(background_embeddings.shape)
    input_length = len(background_data_frame)
    CAE_model.build_model(input_shape,embedding_min, embedding_max )
    CAE_model.compile_model(input_length)
    CAE_model.only_CAE_fit(background_embeddings)
    CAE_model.save()
    CAE_model.plot_loss()
    
    print("Training CAE")
    
    VAE_model = fromYaml('model/configs/EmbeddingClassicalVAEModel.yaml',output_path+'/models/VAE')
    input_shape = background_embeddings.shape[0]
    input_length = len(background_data_frame)
    VAE_model.build_model(input_shape,embedding_min, embedding_max )
    VAE_model.compile_model(input_length)
    VAE_model.only_VAE_fit(background_embeddings)
    VAE_model.save()
    VAE_model.plot_loss()

    print("Training QAE")
    
    QAE_model = fromYaml('model/configs/EmbeddingPennyLaneQAEModel.yaml',output_path+'/models/QAE')
    input_shape = background_embeddings.shape[0]
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
    HW_QAE_model.build_model(input_shape,embedding_min, embedding_max )
    HW_QAE_model.compile_model(input_length)
    HW_QAE_model.only_QAE_fit(background_embeddings)
    HW_QAE_model.save()
    HW_QAE_model.plot_loss()
    
CAE_model = fromFolder(output_path+'/models/CAE')
VAE_model = fromFolder(output_path+'/models/VAE')
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
        embeddings = test_embeddings[indices[0]]
        data_frame = test_data_frame.iloc[indices[0]]

        print("Classical Autoencoder Predict")
        output_dict['CAE'][label] = {'predictions' : CAE_model.only_CAE_predict(embeddings)}
        print("Classical Variational Autoencoder Predict")
        output_dict['VAE'][label] = {'predictions' : VAE_model.only_VAE_predict(embeddings)}
        print("Quantum Autoencoder Predict")
        
        use_max_events = True
        if label == "background":
            use_max_events = False
            
        output_dict['QAE'][label] = {'predictions' : QAE_model.only_QAE_predict(embeddings,use_max_events)}
        print("HW Quantum Autoencoder Predict")
        output_dict['HW_QAE'][label] = {'predictions' : HW_QAE_model.only_QAE_predict(embeddings,use_max_events)}    
    
    with open(output_path+'/models/output_dict.pkl', 'wb') as f:
        pickle.dump(output_dict, f)
        
with open(output_path+'/models/output_dict.pkl', 'rb') as f:
    output_dict = pickle.load(f)
    
    
if plot:
    print("Plotting output histograms")
    
    for model in models:
        plot_histo([output_dict[model][dataset]['predictions'] for dataset in output_dict[model].keys()], 
                [fancy_dataset[dataset] for dataset in output_dict[model].keys()], 
                '', 
                'Anomaly Score', 
                'a.u.', 
                log = 'log', 
                x_range=(0, 1), 
                bins = 50)
        os.makedirs(f"{output_path}/models/{model}/plots/testing/", exist_ok=True)
        plt.savefig(f"{output_path}/models/{model}/plots/testing/outputscores.png", bbox_inches='tight')
        plt.savefig(f"{output_path}/models/{model}/plots/testing/outputscores.pdf", bbox_inches='tight')
        plt.close() 

    
    print("Plotting output ROCs")
    
    for model in models:
        
        ROC_curve([output_dict[model][background_name]['predictions'] for dataset in output_dict[model].keys()],
                [output_dict[model][dataset]['predictions'] for dataset in output_dict[model].keys()],
                [fancy_dataset[key] for key in fancy_dataset.keys()])
        plt.savefig(f"{output_path}/models/{model}/plots/testing/ROC.png", bbox_inches='tight')
        plt.savefig(f"{output_path}/models/{model}/plots/testing/ROC.pdf", bbox_inches='tight')
        plt.close() 

    
    for label in test_labels.keys():
        
        ROC_curve([output_dict[model][background_name]['predictions'] for model in models],
                  [output_dict[model][label]['predictions'] for model in models],
                  models)
        os.makedirs(f"{output_path}/plots/testing/", exist_ok=True)
        plt.savefig(f"{output_path}/plots/testing/{label}_ROC.png", bbox_inches='tight')
        plt.savefig(f"{output_path}/plots/testing/{label}_ROC.pdf", bbox_inches='tight')
        plt.close() 
        
        
if rerun_augment_predictions and rerun_embedding:
    
    print(f" Rerunning augment embedding from: {embedding_model_path}")
    
    training_columns = []
    dataset_list = []

    for datasets in test_labels.keys():
        data_test = DataSet.fromH5(input_path+datasets+'/test/')        
        #augment_test.drop_a_soft_one('jet')
        #data_test.eta_smear()
        data_test.pt_smear()
        #data_test.phi_smear()
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
    test_index = np.random.randint(0, len(background_indices[0]), size=10000)
    background_test_embeddings = test_embeddings[background_indices[0][test_index]]
    background_test_data_frame = test_data_frame.iloc[background_indices[0][test_index]]

    background_indices = np.where(augment_data_frame["event_label"] == 0)
    test_index = np.random.randint(0, len(background_indices[0]), size=10000)
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

    signal_indices = np.where(test_data_frame["event_label"] == 1)
    test_index = np.random.randint(0, len(signal_indices[0]), size=10000)
    signal_test_embeddings = test_embeddings[signal_indices[0][test_index]]

    signal_indices = np.where(augment_data_frame["event_label"] == 1)
    test_index = np.random.randint(0, len(signal_indices[0]), size=10000)
    signal_augment_embeddings = augment_embeddings[signal_indices[0][test_index]]
        
    latent_vector = np.concatenate([signal_test_embeddings,signal_augment_embeddings], axis=0)
    event_label_vector = np.concatenate([np.zeros(len(signal_test_embeddings)),np.ones(len(signal_augment_embeddings))], axis=0)

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
            embeddings = augment_embeddings[indices[0]]
        else:
            indices = np.where(test_data_frame["event_label"] == augment_labels[label])
            embeddings = test_embeddings[indices[0]]
            

        print("Classical Autoencoder Predict")
        augment_output_dict['CAE'][label] = {'predictions' : CAE_model.only_CAE_predict(embeddings)}
        print("Classical Variational Autoencoder Predict")
        augment_output_dict['VAE'][label] = {'predictions' : VAE_model.only_VAE_predict(embeddings)}
  
        use_max_events = True
        if "background" in label:
            use_max_events = False
                    
        print("Quantum Autoencoder Predict")
        augment_output_dict['QAE'][label] = {'predictions' : QAE_model.only_QAE_predict(embeddings,use_max_events)}
        print("HW Quantum Autoencoder Predict")
        augment_output_dict['HW_QAE'][label] = {'predictions' : HW_QAE_model.only_QAE_predict(embeddings,use_max_events)}  
        
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
        plt.savefig(f"{output_path}/models/{model}/plots/augment/testing/outputscores.pdf", bbox_inches='tight')
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
        plt.savefig(f"{output_path}/plots/augment/testing/{label}_outputscores.pdf", bbox_inches='tight')
        plt.close()  
    
    print("Plotting augment ROCs")
    
    for model in models:
        
        ROC_curve([augment_output_dict[model][background_name]['predictions'] for dataset in augment_output_dict[model].keys()],
            [augment_output_dict[model][dataset]['predictions'] for dataset in augment_output_dict[model].keys()],
            augment_labels)
        plt.savefig(f"{output_path}/models/{model}/plots/augment/testing/ROC.png", bbox_inches='tight')
        plt.savefig(f"{output_path}/models/{model}/plots/augment/testing/ROC.pdf", bbox_inches='tight')
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
        plt.savefig(f"{output_path}/plots/augment/testing/{label}_ROC.pdf", bbox_inches='tight')
        plt.close() 
        
        
for model in models:
    background = augment_output_dict[model][background_name]['predictions']
    background_a = augment_output_dict[model][background_augment_name]['predictions']

    sig = augment_output_dict[model][signal_name]['predictions']
    sig_a = augment_output_dict[model][signal_augment_name]['predictions']
    
    background_wd = scipy.stats.wasserstein_distance(background,background_a)
    sig_wd = scipy.stats.wasserstein_distance(sig,sig_a)
    
    auc_loss_list,tpr_loss_list = ROC_curve([background, background_a],
                                            [sig,     sig_a],
                                            [background_name,signal_name],plot=False)
    
    print(model)
    print(f"Wasserstein Distance for background samples {background_wd}")
    print(f"Wasserstein Distance for Signal sample {sig_wd}")
    print(f"Non Augmented ROC {auc_loss_list[0]}")
    print(f"Augmented ROC {auc_loss_list[1]}")
    print(f"Non Augmented TPR {tpr_loss_list[0]}")
    print(f"Augmented TPR {tpr_loss_list[1]}")
    
    
if run_augment_scan:
    
    num_samples = {background_name:100000,signal_name:1000}
    num_CV = 10
    print("Running augment scan")
    
    os.makedirs(output_path+"/augment/", exist_ok=True)
    f = open(output_path+"/augment/scan_classical_only.csv",'w')
    f.write(f"model,smear_percent,auc_loss_non_augmented,auc_loss_non_augmented_err,auc_loss_augmented ,auc_loss_augmented_err,tpr_non_augmented,tpr_non_augmented_err,tpr_augmented ,tpr_augmented_err,background_wd,background_wd_err,signal_wd,signal_wd_err,background_mse,background_mse_err,signal_mse,signal_mse_err,background_mae,background_mae_err,signal_mae,signal_mae_err,background_cos,background_cos_err,signal_cos,signal_cos_err,background_euc,background_euc_err,signal_euc,signal_euc_err\n")

    non_augmented_model_results = {background_name: {'CAE':[],'VAE':[],
                                                     'QAE':[],'HW_QAE':[],
                                                     'embeddings':[]},
                                   signal_name: {'CAE':[],'VAE':[],
                                                 'QAE':[],'HW_QAE':[],
                                                 'embeddings':[]}}

    for iCV in range(num_CV):
        start_time = time.time()
        for datasets in [background_name,signal_name]:
                        
                temp_dataset = DataSet.fromH5(input_path+datasets+'/test/')
                temp_dataset.normalise()
                data_test_dataframe = temp_dataset.data_frame
                
                temp_dataset.set_label(test_labels[datasets])
                embeddings = embedding_model.encoder_predict(data_test_dataframe[iCV*num_samples[datasets]:(iCV+1)*num_samples[datasets]],temp_dataset.training_columns) 

                non_augmented_model_results[datasets]['CAE'].append(CAE_model.only_CAE_predict(embeddings))
                non_augmented_model_results[datasets]['VAE'].append(VAE_model.only_VAE_predict(embeddings))
                                    
                use_max_events = True
                if datasets == 'background':
                    use_max_events = False
                    
                non_augmented_model_results[datasets]['QAE'].append(QAE_model.only_QAE_predict(embeddings,use_max_events))
                non_augmented_model_results[datasets]['HW_QAE'].append(HW_QAE_model.only_QAE_predict(embeddings,use_max_events))  
                non_augmented_model_results[datasets]['embeddings'].append(embeddings)
                
                gc.collect()
                
        print(f"Cross validation {iCV} background finishing in {time.time() - start_time} s")

    for smear_percent in [0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.0,5.0,10.0,15.0,20.0]:
        
        augmented_model_results = {background_name:{'CAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                           'augmented_tpr':[],'non_augmented_tpr':[],
                                                           'WD_signal':[],'WD_background':[],
                                                           'MSE_signal':[],'MSE_background':[],
                                                           'MAE_signal':[],'MAE_background':[],
                                                           'cos_signal':[],'cos_background':[],
                                                           'euc_signal':[],'euc_background':[]},
                                                    'VAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                           'augmented_tpr':[],'non_augmented_tpr':[],
                                                           'WD_signal':[],'WD_background':[],
                                                           'MSE_signal':[],'MSE_background':[],
                                                           'MAE_signal':[],'MAE_background':[],
                                                           'cos_signal':[],'cos_background':[],
                                                           'euc_signal':[],'euc_background':[]},
                                                    'QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                           'augmented_tpr':[],'non_augmented_tpr':[],
                                                           'WD_signal':[],'WD_background':[],
                                                           'MSE_signal':[],'MSE_background':[],
                                                           'MAE_signal':[],'MAE_background':[],
                                                           'cos_signal':[],'cos_background':[],
                                                           'euc_signal':[],'euc_background':[]},
                                                    'HW_QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                              'augmented_tpr':[],'non_augmented_tpr':[],
                                                              'WD_signal':[],'WD_background':[],
                                                           'MSE_signal':[],'MSE_background':[],
                                                           'MAE_signal':[],'MAE_background':[],
                                                           'cos_signal':[],'cos_background':[],
                                                           'euc_signal':[],'euc_background':[]},
                                                    'embeddings':{'embeddings':0,'WD':[]}},
                                   signal_name:{'CAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                       'augmented_tpr':[],'non_augmented_tpr':[],
                                                       'WD_signal':[],'WD_background':[],
                                                           'MSE_signal':[],'MSE_background':[],
                                                           'MAE_signal':[],'MAE_background':[],
                                                           'cos_signal':[],'cos_background':[],
                                                           'euc_signal':[],'euc_background':[]},
                                                'VAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                       'augmented_tpr':[],'non_augmented_tpr':[],
                                                       'WD_signal':[],'WD_background':[],
                                                           'MSE_signal':[],'MSE_background':[],
                                                           'MAE_signal':[],'MAE_background':[],
                                                           'cos_signal':[],'cos_background':[],
                                                           'euc_signal':[],'euc_background':[]},
                                                'QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                       'augmented_tpr':[],'non_augmented_tpr':[],
                                                       'WD_signal':[],'WD_background':[],
                                                           'MSE_signal':[],'MSE_background':[],
                                                           'MAE_signal':[],'MAE_background':[],
                                                           'cos_signal':[],'cos_background':[],
                                                           'euc_signal':[],'euc_background':[]},
                                                'HW_QAE':{'predictions':0,'augmented_ROC':[],'non_augmented_ROC':[],
                                                          'augmented_tpr':[],'non_augmented_tpr':[],
                                                          'WD_signal':[],'WD_background':[],
                                                           'MSE_signal':[],'MSE_background':[],
                                                           'MAE_signal':[],'MAE_background':[],
                                                           'cos_signal':[],'cos_background':[],
                                                           'euc_signal':[],'euc_background':[]},
                                                'embeddings':{'embeddings':0,'WD':[]}}}
        
        for iCV in range(num_CV): 
            start_time = time.time()
            for datasets in [background_name,signal_name]:
                temp_dataset =  DataSet.fromH5(input_path+datasets+'/test/')

                #temp_dataset.eta_smear(smear_percent)
                temp_dataset.pt_smear(smear_percent)
                #temp_dataset.phi_smear(smear_percent)

                temp_dataset.normalise()
                temp_dataset.set_label(test_labels[datasets])
                
                data_test_dataframe = temp_dataset.data_frame

                
                print(f"Data set loading {iCV} {datasets} {smear_percent} finishing in {time.time() - start_time} s")
                start_time = time.time()
                augmented_embeddings = embedding_model.encoder_predict(data_test_dataframe[iCV*num_samples[datasets]:(iCV+1)*num_samples[datasets]],temp_dataset.training_columns) 
                print(f"Embedding predict {iCV} {datasets} finishing in {time.time() - start_time} s")
                start_time = time.time()
                print(augmented_embeddings.shape)
                augmented_model_results[datasets]['CAE']['predictions'] = (CAE_model.only_CAE_predict(augmented_embeddings))
                print(f"CAE predict {iCV} {datasets} finishing in {time.time() - start_time} s")
                augmented_model_results[datasets]['VAE']['predictions'] = (VAE_model.only_VAE_predict(augmented_embeddings))
                print(f"VAE predict {iCV} {datasets} finishing in {time.time() - start_time} s")
                start_time = time.time()
                
                use_max_events = True
                if datasets == 'background':
                    use_max_events = False
                
                augmented_model_results[datasets]['QAE']['predictions'] = (QAE_model.only_QAE_predict(augmented_embeddings,use_max_events))
                print(f"QAE predict {iCV} {datasets} finishing in {time.time() - start_time} s")
                start_time = time.time()
                augmented_model_results[datasets]['HW_QAE']['predictions'] = (HW_QAE_model.only_QAE_predict(augmented_embeddings,use_max_events))  
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
                    
                auc_loss_list,tpr_loss_list = ROC_curve([non_augmented_model_results[background_name][model][iCV], augmented_model_results[background_name][model]['predictions']],
                                        [non_augmented_model_results[signal_name][model][iCV],     augmented_model_results[signal_name][model]['predictions']],
                                        [background_name,signal_name],plot=False)
                
                augmented_model_results[background_name][model]['augmented_ROC'].append( auc_loss_list[1])
                augmented_model_results[background_name][model]['non_augmented_ROC'].append( auc_loss_list[0])
                augmented_model_results[background_name][model]['augmented_tpr'].append( tpr_loss_list[1])
                augmented_model_results[background_name][model]['non_augmented_tpr'].append( tpr_loss_list[0])
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
            
            non_augment_tpr = sum(augmented_model_results[background_name][model]['non_augmented_tpr']) /num_CV
            non_augment_tpr_err = np.std(augmented_model_results[background_name][model]['non_augmented_tpr'])
            augment_tpr = sum(augmented_model_results[background_name][model]['augmented_tpr'])/num_CV
            augment_tpr_err = np.std(augmented_model_results[background_name][model]['augmented_tpr']) 
            
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
            
            print("TPR at FPR of 1e-5")
            print(f"Non Augmented TPR {non_augment_tpr} +/- {non_augment_tpr_err}")
            print(f"Augmented TPR {augment_tpr} +/- {augment_tpr_err}")
            
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
                    f"{non_augment_tpr},"
                    f"{non_augment_tpr_err},"
                    f"{augment_tpr} , "
                    f"{augment_tpr_err},"
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
            
        
    
