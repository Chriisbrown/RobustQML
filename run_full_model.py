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

models = ['CAE','IF','QAE','HW_QAE']

######## GLOBAL PARAMETERS for ML Models ############
if Dataset == 'C2V':
    if Model_type == 'MLP':
        embedding_model_path = 'C2V_Contrastive_Embedding/MLP/QCD/ContrastiveEmbedding'
        output_path = 'C2V/MLP/QCD'
    if Model_type == 'transformer':
        embedding_model_path = 'C2V_Contrastive_Embedding/Transformer/QCD/TransformerEmbedding'
        output_path = 'C2V/transformer/QCD'
if Dataset == 'AD': 
    if Model_type == 'MLP':
        embedding_model_path = 'AD_Contrastive_Embedding/MLP/all/ContrastiveEmbedding'
        output_path = 'AD/transformer/all'
    if Model_type == 'transformer':
        embedding_model_path = 'C2V_Contrastive_Embedding/Transformer/QCD/TransformerEmbedding'
        output_path = 'AD/MLP/all'

########## GLOBAL PARAMETERS for Collide 2V ##########

if Dataset == 'C2V':
    train_labels = {"QCD_HT50tobb":2,"QCD_HT50toInf":1,"minbias":0}
    test_labels = {"minbias" : 0, "QCD_HT50toInf" :1, "HH_4b" : 2, 'HH_bbgammagamma':3,'HH_bbtautau':4,'QCD_HT50tobb':5}
    augment_labels = {"minbias" : 0, "HH_4b" : 2, "minbias_augment" : 0, "HH_4b_augment" : 2}


    output_dict = {'CAE' : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}},
                'IF'  : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}},
                'QAE' : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}},
                'HW_QAE' : {"minbias" : {}, "QCD_HT50toInf" :{}, "HH_4b" : {}, "HH_bbgammagamma" : {}, "HH_bbtautau" : {}, "QCD_HT50tobb": {}}}

    augment_output_dict = {'CAE' : {"minbias" : {}, "HH_4b" : {}, "minbias_augment" : {}, "HH_4b_augment" : {} },
                            'IF'  : {"minbias" : {}, "HH_4b" : {}, "minbias_augment" : {}, "HH_4b_augment" : {} },
                            'QAE' : {"minbias" : {}, "HH_4b" : {}, "minbias_augment" : {}, "HH_4b_augment" : {} },
                            'HW_QAE' : {"minbias" : {}, "HH_4b" : {}, "minbias_augment" : {}, "HH_4b_augment" : {} }}

    input_path = '/eos/user/c/cebrown/RobustQML/training_data/'

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
                'IF'  : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}, "blackbox": {}},
                'QAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}, "blackbox": {}},
                'HW_QAE' : {"background" : {}, "ato4l" :{}, "hChToTauNu" : {}, "hToTauTau" : {}, "leptoquark" : {}, "blackbox": {}}}

    augment_output_dict = {'CAE' : {"background" : {}, "ato4l" : {}, "background_augment" : {}, "ato4l_augment" : {} },
                            'IF'  : {"background" : {}, "ato4l" : {}, "background_augment" : {}, "ato4l_augment" : {} },
                            'QAE' : {"background" : {}, "ato4l" : {}, "background_augment" : {}, "ato4l_augment" : {} },
                            'HW_QAE' : {"background" : {}, "ato4l" : {}, "background_augment" : {}, "ato4l_augment" : {} }}

    input_path = '/eos/user/c/cebrown/RobustQML/AD_dataset/processed/'

    background_name = 'background'
    background_augment_name = background_name + '_augment'

    signal_name = 'ato4l'
    signal_augment_name = signal_name + '_augment'

##################


os.makedirs(output_path, exist_ok=True)

setup_gpu_memory_growth()
embedding_model = fromFolder(embedding_model_path)

if rerun_embedding:
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
    CAE_model = fromYaml('model/configs/EmbeddingClassicalAEModel.yaml',output_path+'/models/CAE')
    input_shape = background_embeddings.shape[0]
    input_length = len(background_data_frame)
    CAE_model.build_model(input_shape,embedding_min, embedding_max )
    CAE_model.compile_model(input_length)
    CAE_model.only_CAE_fit(background_embeddings)
    CAE_model.save()
    CAE_model.plot_loss()
    
    IF_model = fromYaml('model/configs/IsolationTreeModel.yaml',output_path+'/models/IF')
    os.makedirs(output_path+'/models/IF/plots', exist_ok=True)
    input_shape = background_embeddings.shape[0]
    input_length = len(background_data_frame)
    IF_model.build_model(input_shape)
    IF_model.compile_model(input_length)
    IF_model.fit_on_embedding(background_embeddings)
    IF_model.save()
    IF_model.plot_loss()
    
    QAE_model = fromYaml('model/configs/EmbeddingPennyLaneQAEModel.yaml',output_path+'/models/QAE')
    input_shape = background_embeddings.shape[0]
    input_length = len(background_data_frame)
    input_length = len(background_data_frame)
    QAE_model.build_model(input_shape,embedding_min, embedding_max )
    QAE_model.compile_model(input_length)
    QAE_model.only_QAE_fit(background_embeddings)
    QAE_model.save()
    QAE_model.plot_loss()
    
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
IF_model = fromFolder(output_path+'/models/IF')
QAE_model = fromFolder(output_path+'/models/QAE')
HW_QAE_model = fromFolder(output_path+'/models/HW_QAE')

if rerun_embedding:
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

    for label in test_labels.keys():
        print("==== Predicting for "+label+" ====")
        indices = np.where(test_data_frame["event_label"] == test_labels[label])
        test_index = np.random.randint(0, len(indices[0]), size=10000)
        embeddings = test_embeddings[indices[0][test_index]]
        data_frame = test_data_frame.iloc[indices[0][test_index]]

        print("Classical Autoencoder Predict")
        output_dict['CAE'][label] = {'predictions' : CAE_model.only_CAE_predict(embeddings)}
        print("Isolation Forest Predict")
        output_dict['IF'][label]  = {'predictions' : IF_model.predict_on_embedding(embeddings)}
        print("Quantum Autoencoder Predict")
        output_dict['QAE'][label] = {'predictions' : QAE_model.only_QAE_predict(embeddings)}
        print("HW Quantum Autoencoder Predict")
        output_dict['HW_QAE'][label] = {'predictions' : HW_QAE_model.only_QAE_predict(embeddings)}    
    
    with open(output_path+'/models/output_dict.pkl', 'wb') as f:
        pickle.dump(output_dict, f)
        
with open(output_path+'/models/output_dict.pkl', 'rb') as f:
    output_dict = pickle.load(f)
    
    
if plot:
    plot_histo([output_dict['CAE'][dataset]['predictions'] for dataset in output_dict['CAE'].keys()], 
               [dataset for dataset in output_dict['CAE'].keys()], 
               '', 
               'AnomalyScore', 
               'a.u.', 
               log = 'linear', 
               x_range=(0, 1), 
               bins = 50)
    os.makedirs(output_path+"/models/CAE/plots/testing/", exist_ok=True)
    plt.savefig(f"{output_path}/models/CAE/plots/testing/outputscores.png", bbox_inches='tight')
    plt.close() 

    plot_histo([output_dict['IF'][dataset]['predictions'] for dataset in output_dict['IF'].keys()], 
                [dataset for dataset in output_dict['IF'].keys()], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
    os.makedirs(output_path+"/models/IF/plots/testing/", exist_ok=True)
    plt.savefig(f"{output_path}/models/IF/plots/testing/outputscores.png", bbox_inches='tight')
    plt.close() 

    plot_histo([output_dict['QAE'][dataset]['predictions'] for dataset in output_dict['QAE'].keys()], 
                [dataset for dataset in output_dict['QAE'].keys()], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
    os.makedirs(output_path+"/models/QAE/plots/testing/", exist_ok=True)
    plt.savefig(f"{output_path}/models/QAE/plots/testing/outputscores.png", bbox_inches='tight')
    plt.close() 

    plot_histo([output_dict['HW_QAE'][dataset]['predictions'] for dataset in output_dict['QAE'].keys()], 
                [dataset for dataset in output_dict['HW_QAE'].keys()], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
    os.makedirs(output_path+"/models/HW_QAE/plots/testing/", exist_ok=True)
    plt.savefig(f"{output_path}/models/HW_QAE/plots/testing/outputscores.png", bbox_inches='tight')
    plt.close() 
    
    os.makedirs(output_path+"/plots/testing/", exist_ok=True)
    for label in test_labels.keys():
        plot_histo([output_dict['CAE'][label]['predictions'],output_dict['IF'][label]['predictions'],output_dict['QAE'][label]['predictions'],output_dict['HW_QAE'][label]['predictions']], 
                ['CAE','IF','QAE','HW_QAE'], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
        plt.savefig(f"{output_path}/plots/testing/{label}_outputscores.png", bbox_inches='tight')
        plt.close()
        
        
    ROC_curve([output_dict['CAE'][background_name]['predictions'] for dataset in output_dict['CAE'].keys()],
            [output_dict['CAE'][dataset]['predictions'] for dataset in output_dict['CAE'].keys()],
            test_labels)
    plt.savefig(f"{output_path}/models/CAE/plots/testing/ROC.png", bbox_inches='tight')
    plt.close() 

    ROC_curve([output_dict['IF'][background_name]['predictions'] for dataset in output_dict['IF'].keys()],
            [output_dict['IF'][dataset]['predictions'] for dataset in output_dict['IF'].keys()],
            test_labels)
    plt.savefig(f"{output_path}/models/IF/plots/testing/ROC.png", bbox_inches='tight')
    plt.close() 

    ROC_curve([output_dict['QAE'][background_name]['predictions'] for dataset in output_dict['QAE'].keys()],
            [output_dict['QAE'][dataset]['predictions'] for dataset in output_dict['QAE'].keys()],
            test_labels)
    plt.savefig(f"{output_path}/models/QAE/plots/testing/ROC.png", bbox_inches='tight')
    plt.close() 

    ROC_curve([output_dict['HW_QAE'][background_name]['predictions'] for dataset in output_dict['HW_QAE'].keys()],
            [output_dict['HW_QAE'][dataset]['predictions'] for dataset in output_dict['HW_QAE'].keys()],
            test_labels)
    plt.savefig(f"{output_path}/models/HW_QAE/plots/testing/ROC.png", bbox_inches='tight')
    plt.close() 
    
    for label in test_labels.keys():
        ROC_curve([output_dict['CAE'][background_name]['predictions'],output_dict['IF'][background_name]['predictions'],output_dict['QAE'][background_name]['predictions'],output_dict['HW_QAE'][background_name]['predictions']],
                [output_dict['CAE'][label]['predictions'],output_dict['IF'][label]['predictions'],output_dict['QAE'][label]['predictions'],output_dict['HW_QAE'][label]['predictions']],
                ['CAE','IF','QAE','HW_QAE'])
        
        plt.savefig(f"{output_path}/plots/testing/{label}_ROC.png", bbox_inches='tight')
        plt.close() 
        
        
if rerun_predictions:
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
        print("Isolation Forest Predict")
        augment_output_dict['IF'][label]  = {'predictions' : IF_model.predict_on_embedding(embeddings)}
        print("Quantum Autoencoder Predict")
        augment_output_dict['QAE'][label] = {'predictions' : QAE_model.only_QAE_predict(embeddings)}
        print("HW Quantum Autoencoder Predict")
        augment_output_dict['HW_QAE'][label] = {'predictions' : HW_QAE_model.only_QAE_predict(embeddings)}  
        
    with open(output_path+'/models/output_augment_dict.pkl', 'wb') as f:
        pickle.dump(augment_output_dict, f)

with open(output_path+'/models/output_augment_dict.pkl', 'rb') as f:
    augment_output_dict = pickle.load(f)
print(augment_output_dict)
    
if plot:
    plot_histo([augment_output_dict['CAE'][dataset]['predictions'] for dataset in augment_output_dict['CAE'].keys()], 
               [dataset for dataset in augment_output_dict['CAE'].keys()], 
               '', 
               'AnomalyScore', 
               'a.u.', 
               log = 'linear', 
               x_range=(0, 1), 
               bins = 50)
    os.makedirs(output_path+"/models/CAE/plots/augment/testing/", exist_ok=True)
    plt.savefig(f"{output_path}/models/CAE/plots/augment/testing/outputscores.png", bbox_inches='tight')
    plt.close() 

    plot_histo([augment_output_dict['IF'][dataset]['predictions'] for dataset in augment_output_dict['IF'].keys()], 
                [dataset for dataset in augment_output_dict['IF'].keys()], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
    os.makedirs(output_path+"/models/IF/plots/augment/testing/", exist_ok=True)
    plt.savefig(f"{output_path}/models/IF/plots/augment/testing/outputscores.png", bbox_inches='tight')
    plt.close() 

    plot_histo([augment_output_dict['QAE'][dataset]['predictions'] for dataset in augment_output_dict['QAE'].keys()], 
                [dataset for dataset in augment_output_dict['QAE'].keys()], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
    os.makedirs(output_path+"/models/QAE/plots/augment/testing/", exist_ok=True)
    plt.savefig(f"{output_path}/models/QAE/plots/augment/testing/outputscores.png", bbox_inches='tight')
    plt.close() 

    plot_histo([augment_output_dict['HW_QAE'][dataset]['predictions'] for dataset in augment_output_dict['QAE'].keys()], 
                [dataset for dataset in augment_output_dict['HW_QAE'].keys()], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
    os.makedirs(output_path+"/models/HW_QAE/plots/augment/testing/", exist_ok=True)
    plt.savefig(f"{output_path}/models/HW_QAE/plots/augment/testing/outputscores.png", bbox_inches='tight')
    plt.close()
    
    os.makedirs(output_path+"/plots/augment/testing/", exist_ok=True)
    for label in augment_labels.keys():
        plot_histo([augment_output_dict['CAE'][label]['predictions'],augment_output_dict['IF'][label]['predictions'],augment_output_dict['QAE'][label]['predictions'],augment_output_dict['HW_QAE'][label]['predictions']], 
                ['CAE','IF','QAE','HW_QAE'], 
                '', 
                'AnomalyScore', 
                'a.u.', 
                log = 'linear', 
                x_range=(0, 1), 
                bins = 50)
        plt.savefig(f"{output_path}/plots/augment/testing/{label}_outputscores.png", bbox_inches='tight')
        plt.close()  
        
        
    ROC_curve([augment_output_dict['CAE'][background_name]['predictions'] for dataset in augment_output_dict['CAE'].keys()],
          [augment_output_dict['CAE'][dataset]['predictions'] for dataset in augment_output_dict['CAE'].keys()],
          augment_labels)
    plt.savefig(f"{output_path}/models/CAE/plots/augment/testing/ROC.png", bbox_inches='tight')
    plt.close() 

    ROC_curve([augment_output_dict['IF'][background_name]['predictions'] for dataset in augment_output_dict['IF'].keys()],
            [augment_output_dict['IF'][dataset]['predictions'] for dataset in augment_output_dict['IF'].keys()],
            augment_labels)
    plt.savefig(f"{output_path}/models/IF/plots/augment/testing/ROC.png", bbox_inches='tight')
    plt.close() 

    ROC_curve([augment_output_dict['QAE'][background_name]['predictions'] for dataset in augment_output_dict['QAE'].keys()],
            [augment_output_dict['QAE'][dataset]['predictions'] for dataset in augment_output_dict['QAE'].keys()],
            augment_labels)
    plt.savefig(f"{output_path}/models/QAE/plots/augment/testing/ROC.png", bbox_inches='tight')
    plt.close() 

    ROC_curve([augment_output_dict['HW_QAE'][background_name]['predictions'] for dataset in augment_output_dict['HW_QAE'].keys()],
            [augment_output_dict['HW_QAE'][dataset]['predictions'] for dataset in augment_output_dict['HW_QAE'].keys()],
            augment_labels)
    plt.savefig(f"{output_path}/models/HW_QAE/plots/augment/testing/ROC.png", bbox_inches='tight')
    plt.close() 
    
    
    for label in augment_labels.keys():
        if 'augment' in label:
            ROC_curve([augment_output_dict['CAE'][background_augment_name]['predictions'],augment_output_dict['IF'][background_augment_name]['predictions'],augment_output_dict['QAE'][background_augment_name]['predictions'],augment_output_dict['HW_QAE'][background_augment_name]['predictions']],
                    [augment_output_dict['CAE'][label]['predictions'],augment_output_dict['IF'][label]['predictions'],augment_output_dict['QAE'][label]['predictions'],augment_output_dict['HW_QAE'][label]['predictions']],
                    ['CAE','IF','QAE','HW_QAE'])
            
        else:
            ROC_curve([augment_output_dict['CAE'][background_name]['predictions'],augment_output_dict['IF'][background_name]['predictions'],augment_output_dict['QAE'][background_name]['predictions'],augment_output_dict['HW_QAE'][background_name]['predictions']],
                    [augment_output_dict['CAE'][label]['predictions'],augment_output_dict['IF'][label]['predictions'],augment_output_dict['QAE'][label]['predictions'],augment_output_dict['HW_QAE'][label]['predictions']],
                    ['CAE','IF','QAE','HW_QAE'])
        
        plt.savefig(f"{output_path}/plots/augment/testing/{label}_ROC.png", bbox_inches='tight')
        plt.close() 
        
        
for model in models:
    background = augment_output_dict[model][background_name]['predictions']
    background_a = augment_output_dict[model][background_augment_name]['predictions']

    sig = augment_output_dict[model][signal_name]['predictions']
    sig_a = augment_output_dict[model][signal_augment_name]['predictions']
    
    background_wd = scipy.stats.wasserstein_distance(background,background_a)
    sig_wd = scipy.stats.wasserstein_distance(sig,sig_a)
    
    print(model)
    print(f"Wasserstein Distance for background samples {background_wd}")
    print(f"Wasserstein Distance for Signal sample {sig_wd}")
    
    
if run_augment_scan:
    os.makedirs(output_path+"/augment/", exist_ok=True)
    f = open(output_path+"/augment/scan.csv",'w')
    f.write(f"model,smear_percent,auc_loss_non_augmented,auc_loss_augmented ,background_wd,signal_wd")
    
    non_augmented_model_results = {background_name:{},signal_name:{}}
    for datasets in [background_name,signal_name]:
            data_test = DataSet.fromH5(input_path+datasets+'/test/')
            data_test_dataframe = data_test.data_frame.sample(n=1000)
            
            data_test.set_label(test_labels[datasets])

            embeddings = embedding_model.encoder_predict(data_test.data_frame,data_test.training_columns) 
                      
            temp_non_augmented_model_results = {'CAE':0,'IF':0,'QAE':0,'HW_QAE':0}
                        
            temp_non_augmented_model_results['CAE'] = CAE_model.only_CAE_predict(embeddings)
            temp_non_augmented_model_results['IF'] = IF_model.predict_on_embedding(embeddings)
            temp_non_augmented_model_results['QAE'] = QAE_model.only_QAE_predict(embeddings)
            temp_non_augmented_model_results['HW_QAE']= HW_QAE_model.only_QAE_predict(embeddings)  
            
            non_augmented_model_results[datasets] = temp_non_augmented_model_results
    
    
    for smear_percent in [0.0001,0.001,0.01,0.1,1.0]:
        augmented_model_results = {background_name:{},signal_name:{}}
        for datasets in [background_name,signal_name]:
            data_test = DataSet.fromH5(input_path+datasets+'/test/')
            data_test_dataframe = data_test.data_frame.sample(n=1000)
            
            #augment_test.drop_a_soft_one('jet')
            data_test.eta_smear(smear_percent)
            data_test.pt_smear(smear_percent)
            data_test.phi_smear(smear_percent)
            #data_test.phi_rotate()
            data_test.normalise()

            data_test.set_label(test_labels[datasets])

            embeddings = embedding_model.encoder_predict(data_test.data_frame,data_test.training_columns) 
                      
            model_results = {'CAE':0,'IF':0,'QAE':0,'HW_QAE':0}
                        
            model_results['CAE'] = CAE_model.only_CAE_predict(embeddings)
            model_results['IF'] = IF_model.predict_on_embedding(embeddings)
            model_results['QAE'] = QAE_model.only_QAE_predict(embeddings)
            model_results['HW_QAE']= HW_QAE_model.only_QAE_predict(embeddings)  
            
            augmented_model_results[datasets] = model_results
            
        for model in model_results.keys():
            
            background_wd = scipy.stats.wasserstein_distance(non_augmented_model_results[background_name][model],model_results[background_name][model])
            sig_wd = scipy.stats.wasserstein_distance(non_augmented_model_results[signal_name][model],model_results[signal_name][model])
            
            auc_loss_list = ROC_curve([non_augmented_model_results[background_name][model], augmented_model_results[background_name][model]],
                                      [non_augmented_model_results[signal_name][model],     augmented_model_results[signal_name][model]],
                                      [background_name,signal_name],plot=False)
            
            
            
            print(model)
            print(smear_percent)
            print(f"Non Augmented ROC {auc_loss_list[0]}")
            print(f"Augmented ROC {auc_loss_list[1]}")
            print("Wasserstein Distance")
            print(f"Background sample {background_wd}")
            print(f"Signal sample {sig_wd}")
            print("============")
            
            f.write(f"{model},{smear_percent},{auc_loss_list[0]},{auc_loss_list[1]} ,{background_wd},{sig_wd}")
    f.close()
            
        
    