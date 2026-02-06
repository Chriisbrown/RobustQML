import os
from argparse import ArgumentParser
from model.common import fromFolder
from data.dataset import DataSet

from plot import style

from basic import error_residual, plot_histo, rates,efficiency, clusters, plot_2d

from sklearn.metrics import roc_curve, auc


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

style.set_style()

if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        '-o', '--output', default='output/autoencoder', help='Output model directory path, also save evaluation plots'
    )

    args = parser.parse_args()
    
    model = fromFolder(args.output)

    plot_dir = os.path.join(model.output_directory, "plots/testing")
    os.makedirs(plot_dir, exist_ok=True)
    
    minbias = DataSet.fromH5('dataset/Minbias')
    #minbias.normalise()
    minbias_test = minbias.data_frame.sample(frac=0.001)
    minbias_outputs = model.predict(minbias_test,minbias.training_columns)
    print(minbias_outputs)
    minbias_rates = rates(type(model).__name__,'minbias',minbias_outputs)
    
    output_dict = {"Minbias" : {}, "VBFHcc" :{}, "ggHbb" : {}, "QCDbb" : {}, "HH4b" : {}, "QCD" : {}}
    
    for datasets in output_dict.keys():
        data_test = DataSet.fromH5('dataset/'+datasets)
        test = data_test.data_frame.sample(frac=0.001)
        #data_test.normalise()
        model_outputs = model.predict(test,data_test.training_columns)
        print(model_outputs)
        efficiency_out = efficiency(type(model).__name__,datasets,model_outputs)
        output_dict[datasets] = {'predictions' : model_outputs,'efficiencies' : efficiency_out,'dataset':data_test}
        
        # plot_2d(data_test.data_frame['jet_multiplicity'], model_outputs, (0,10), (0,1), 'jet multiplicity', 'model predictions', 'jet multiplicity dependence for '+datasets)
        # save_path = os.path.join(plot_dir, "jet_mult_"+datasets)
        # plt.savefig(f"{save_path}.png", bbox_inches='tight')
        # plt.close() 
        # plot_2d(data_test.data_frame['electron_multiplicity'], model_outputs, (0,2), (0,1), 'electron multiplicity', 'model predictions', 'electron multiplicity dependence for '+datasets)
        # save_path = os.path.join(plot_dir, "e_mult_"+datasets)
        # plt.savefig(f"{save_path}.png", bbox_inches='tight')
        # plt.close() 
        # plot_2d(data_test.data_frame['muon_multiplicity'], model_outputs, (0,2), (0,1), 'muon multiplicity', 'model predictions', 'muon multiplicity dependence for '+datasets)
        # save_path = os.path.join(plot_dir, "mu_mult_"+datasets)
        # plt.savefig(f"{save_path}.png", bbox_inches='tight')
        # plt.close() 
        # plot_2d(data_test.data_frame['FullReco_GenMissingET_MET'], model_outputs, (0,100), (0,1), 'Gen Missing ET', 'model predictions', 'Gen Missing ET dependence for '+datasets)
        # save_path = os.path.join(plot_dir, "genEt_"+datasets)
        # plt.savefig(f"{save_path}.png", bbox_inches='tight')
        # plt.close() 
        
    target_background = np.zeros(output_dict['Minbias']['predictions'].shape[0])
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    for i,datasets in enumerate(output_dict.keys()):
        trueVal = np.concatenate((np.ones(output_dict[datasets]['predictions'].shape[0]), target_background)) # anomaly=1, bkg=0
        predVal_loss = np.concatenate((output_dict[datasets]['predictions'], output_dict['Minbias']['predictions']))

        fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

        auc_loss = auc(fpr_loss, tpr_loss)
                
        plt.plot(fpr_loss, tpr_loss, "-", label=datasets+' (auc = %.1f%%)'%(auc_loss*100.), color = style.colours[i])
            
    ax.semilogx()
    ax.semilogy()
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.legend(loc='center right')
    ax.grid(True)
    ax.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    ax.axvline(0.00001, color='green', linestyle='dashed', linewidth=2) # threshold value for measuring anomaly detection efficiency
    save_path = os.path.join(plot_dir, "output_ROC")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close() 
    
    
    if model.encoder_predict(minbias) != None:
    
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        for i,datasets in enumerate(output_dict.keys()):
                
            latent_representations = model.encoder_predict(output_dict[datasets]['dataset'])
            ax.scatter(latent_representations[:, 0], latent_representations[:, 1], 
                    alpha=0.2, c=style.colours[i], edgecolor=style.colours[i], label=datasets)
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        ax.legend()
        save_path = os.path.join(plot_dir, "latent_space")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close() 
        
    if model.var_predict(minbias) != None:
    
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        for i,datasets in enumerate(output_dict.keys()):
                
            mean,logvar = model.var_predict(output_dict[datasets]['dataset'])
            ax.scatter(mean, logvar, 
                    alpha=0.2, c=style.colours[i], edgecolor=style.colours[i], label=datasets)
        ax.set_xlabel("mean")
        ax.set_ylabel("log var")
        ax.legend()
        save_path = os.path.join(plot_dir, "mean_logvar")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close() 
        
    plot_histo([output_dict[dataset]['predictions'] for dataset in output_dict.keys()], 
               [dataset for dataset in output_dict.keys()], 
               '', 
               'AnomalyScore', 
               'a.u.', 
               log = 'linear', 
               x_range=(0, 1), 
               bins = 50)
    
    save_path = os.path.join(plot_dir, "output_scores")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close() 
    
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    for i,sample_name in enumerate(output_dict.keys()):
        ax.plot(minbias_rates,output_dict[sample_name]['efficiencies'], label=sample_name, linewidth=style.LINEWIDTH,color = style.colours[i])
    ax.grid(True)
    ax.set_ylabel('Signal Efficiency')
    ax.set_xlabel('Background Rate')
    ax.legend(loc='upper right')
    ax.set_xlim(0.0001,1)
    ax.set_ylim(0.0001,10)
    ax.set_xscale("log")
    ax.set_yscale("log")

    save_path = os.path.join(plot_dir, "trigger_roc")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    
    labels = {"Minbias" : 0, "VBFHcc" :1, "ggHbb" : 2, "QCDbb" : 3, "HH4b" : 4, "QCD": 5}
    dataset_list = []
    for datasets in output_dict.keys():
        data_test = DataSet.fromH5('dataset/'+datasets)
        #data_test.normalise()
        data_test.set_label(labels[datasets])
        dataset_list.append(data_test)
        
    full_data_frame = pd.concat([dataset.data_frame.sample(n=5000) for dataset in dataset_list])
    full_data_frame = full_data_frame.sample(frac=0.001)
    combined_predictions = model.predict(full_data_frame[data_test.training_columns],data_test.training_columns)
    

    
    plot_2d(full_data_frame['jet_multiplicity'], combined_predictions, (0,10), (0,1), 'jet multiplicity', 'model predictions', 'jet multiplicity dependence')
    save_path = os.path.join(plot_dir, "jet_mult")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close() 
    plot_2d(full_data_frame['electron_multiplicity'], combined_predictions, (0,2), (0,1), 'electron multiplicity', 'model predictions', 'electron multiplicity dependence')
    save_path = os.path.join(plot_dir, "e_mult")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close() 
    plot_2d(full_data_frame['muon_multiplicity'], combined_predictions, (0,2), (0,1), 'muon multiplicity', 'model predictions', 'muon multiplicity dependence')
    save_path = os.path.join(plot_dir, "mu_mult")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close() 
    plot_2d(full_data_frame['FullReco_GenMissingET_MET'], combined_predictions, (0,100), (0,1), 'Gen Missing ET', 'model predictions', 'Gen Missing ET dependence')
    save_path = os.path.join(plot_dir, "genEt")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close() 
    
    
    distances = model.distance(full_data_frame[dataset_list[0].training_columns].iloc[0:250],training_columns)
    clusters(distances,labels=np.array(full_data_frame['event_label']),plot_dir=plot_dir, label_to_names={v: k for k, v in labels.items()})