import os
from argparse import ArgumentParser
from model.common import fromFolder
from data.EOSdataset import DataSet
from model.gpu_utils import setup_gpu_memory_growth

from plot import style

from plot.basic import error_residual, plot_histo, rates,efficiency, clusters, plot_2d

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
    
    parser.add_argument(
        '-n', '--normalise', default='True', help='Normalise the input data?'
    )
    
    parser.add_argument(
        '-e', '--events', default=-1, type=int,help='Number of the test set events to run over'
    )

    args = parser.parse_args()
    
    setup_gpu_memory_growth()
    
    model = fromFolder(args.output)

    plot_dir = os.path.join(model.output_directory, "plots/testing")
    os.makedirs(plot_dir, exist_ok=True)
      
  
    background = DataSet.fromH5('/eos/user/c/cebrown/RobustQML/training_data/minbias/test')
    if args.normalise == 'True':
      background.normalise()
    else:
      background.max_number_of_jets = 10
      background.max_number_of_objects = 4
      background.max_number_of_objects = 4
      background.generate_feature_lists()
    
    training_columns = background.training_columns


    if args.events > 0:
      background = background.data_frame.sample(n=args.events)
    background_outputs = model.predict(background,training_columns)
    
    background_augment = DataSet.fromH5('/eos/user/c/cebrown/RobustQML/training_data/minbias/augment')
    if args.normalise == 'True':
      background_augment.normalise()
    else:
      background_augment.max_number_of_jets = 10
      background_augment.max_number_of_objects = 4
      background_augment.max_number_of_objects = 4
      background_augment.generate_feature_lists()
    
    if args.events > 0:
      background_augment = background_augment.data_frame.sample(n=args.events)
    background_augment_outputs = model.predict(background_augment,training_columns)
    
    signal_augment = DataSet.fromH5('/eos/user/c/cebrown/RobustQML/training_data/HH_4b/augment')
    if args.normalise == 'True':
      signal_augment.normalise()
    else:
      signal_augment.max_number_of_jets = 10
      signal_augment.max_number_of_objects = 4
      signal_augment.max_number_of_objects = 4
      signal_augment.generate_feature_lists()
    if args.events > 0:
      signal_augment = signal_augment.data_frame.sample(n=args.events)
    signal_augment_outputs = model.predict(signal_augment,training_columns)

    signal = DataSet.fromH5('/eos/user/c/cebrown/RobustQML/training_data/HH_4b/test')
    if args.normalise == 'True':
      signal.normalise()
    else:
      signal.max_number_of_jets = 10
      signal.max_number_of_objects = 4
      signal.max_number_of_objects = 4
      signal.generate_feature_lists()
    if args.events > 0:
      signal = signal.data_frame.sample(n=args.events)
    signal_outputs = model.predict(signal,training_columns)   
    
    target_background = np.zeros(background_augment_outputs.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)

    trueVal = np.concatenate((np.ones(signal_augment_outputs.shape[0]), target_background)) # anomaly=1, bkg=0
    predVal_loss = np.concatenate((signal_augment_outputs, background_augment_outputs))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

    auc_loss = auc(fpr_loss, tpr_loss)
                
    plt.plot(fpr_loss, tpr_loss, "-", label='signal augment'+' (auc = %.1f%%)'%(auc_loss*100.), color = style.colours[0])
    
    target_background = np.zeros(background_outputs.shape[0])
    trueVal = np.concatenate((np.ones(signal_outputs.shape[0]), target_background)) # anomaly=1, bkg=0
    predVal_loss = np.concatenate((signal_outputs, background_outputs))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

    auc_loss = auc(fpr_loss, tpr_loss)
                
    plt.plot(fpr_loss, tpr_loss, "-", label='signal'+' (auc = %.1f%%)'%(auc_loss*100.), color = style.colours[1])
            
    ax.semilogx()
    ax.semilogy()
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.legend(loc='center right')
    ax.grid(True)
    ax.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    ax.axvline(0.00001, color='green', linestyle='dashed', linewidth=2) # threshold value for measuring anomaly detection efficiency
    save_path = os.path.join(plot_dir, "output_ROC_augment")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close() 
    
    efficiency_out = efficiency(type(model).__name__,'signal',signal_outputs)        
    minbias_rates = rates(type(model).__name__,'background',background_outputs)
    
    augment_efficiency_out = efficiency(type(model).__name__,'signal_augmented',signal_augment_outputs)        
    augment_minbias_rates = rates(type(model).__name__,'background_augmented',background_augment_outputs)
    
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    ax.plot(minbias_rates,efficiency_out, label='signal', linewidth=style.LINEWIDTH,color=style.colours[0])
    ax.plot(augment_minbias_rates,augment_efficiency_out, label='signal augmented', linewidth=style.LINEWIDTH,color=style.colours[1])
    
    ax.grid(True)
    ax.set_ylabel('Signal Efficiency')
    ax.set_xlabel('Background Rate')
    ax.legend(loc='upper right')
    ax.set_xlim(0.0001,1)
    ax.set_ylim(0.0001,10)
    ax.set_xscale("log")
    ax.set_yscale("log")

    save_path = os.path.join(plot_dir, "trigger_roc_augmented")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    
    
    plot_histo([background_outputs,background_augment_outputs,signal_outputs,signal_augment_outputs], 
               ['background','Augmented background','signal','Augmented signal'], 
               '', 
               'AnomalyScore', 
               'a.u.', 
               log = 'linear', 
               x_range=(0, 1), 
               bins = 50)
    
    save_path = os.path.join(plot_dir, "augmented_output_scores")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close() 
    
        
    