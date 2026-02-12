import os
from argparse import ArgumentParser
from model.common import fromFolder
from data.dataset import DataSet

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
    
    model = fromFolder(args.output)

    plot_dir = os.path.join(model.output_directory, "plots/testing")
    os.makedirs(plot_dir, exist_ok=True)
      
  
    background = DataSet.fromH5('dataset/background_test')
    training_columns = background.training_columns
    if args.normalise == 'True':
      background.normalise()
    if args.events > 0:
      background = background.data_frame.sample(n=args.events)
    background_outputs = model.predict(background,training_columns)
    
    background_augment = DataSet.fromH5('dataset/background_augment_test')
    if args.normalise == 'True':
      background_augment.normalise()
    if args.events > 0:
      minbias_augment = background_augment.data_frame.sample(n=args.events)
    background_augment_outputs = model.predict(background_augment,training_columns)
    
    ato4l_augment = DataSet.fromH5('dataset/ato4l_augmented')
    if args.normalise == 'True':
      ato4l_augment.normalise()
    if args.events > 0:
      ato4l_augment = ato4l_augment.data_frame.sample(n=args.events)
    ato4l_augment_outputs = model.predict(ato4l_augment,training_columns)

    ato4l = DataSet.fromH5('dataset/ato4l')
    if args.normalise == 'True':
      ato4l.normalise()
    if args.events > 0:
      ato4l = ato4l.data_frame.sample(n=args.events)
    ato4l_outputs = model.predict(ato4l,training_columns)   
    
    target_background = np.zeros(background_outputs.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)

    trueVal = np.concatenate((np.ones(ato4l_augment_outputs.shape[0]), target_background)) # anomaly=1, bkg=0
    predVal_loss = np.concatenate((ato4l_augment_outputs, background_augment_outputs))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

    auc_loss = auc(fpr_loss, tpr_loss)
                
    plt.plot(fpr_loss, tpr_loss, "-", label='ato4l augment'+' (auc = %.1f%%)'%(auc_loss*100.), color = style.colours[0])
    
    
    trueVal = np.concatenate((np.ones(ato4l_outputs.shape[0]), target_background)) # anomaly=1, bkg=0
    predVal_loss = np.concatenate((ato4l_outputs, backgrouns_outputs))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

    auc_loss = auc(fpr_loss, tpr_loss)
                
    plt.plot(fpr_loss, tpr_loss, "-", label='ato4l'+' (auc = %.1f%%)'%(auc_loss*100.), color = style.colours[1])
            
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
    
    efficiency_out = efficiency(type(model).__name__,'ato4l',ato4l_outputs)        
    minbias_rates = rates(type(model).__name__,'background',background_outputs)
    
    augment_efficiency_out = efficiency(type(model).__name__,'ato4l_augmented',ato4l_augment_outputs)        
    augment_minbias_rates = rates(type(model).__name__,'background_augmented',background_augment_outputs)
    
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    ax.plot(minbias_rates,efficiency_out, label='ato4l', linewidth=style.LINEWIDTH,color=style.colours[0])
    ax.plot(augment_minbias_rates,augment_efficiency_out, label='ato4l augmented', linewidth=style.LINEWIDTH,color=style.colours[1])
    
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
    
    
    plot_histo([background_outputs,background_augment_outputs,ato4l_outputs,ato4l_augment_outputs], 
               ['background','Augmented background','ato4l','Augmented ato4l'], 
               '', 
               'AnomalyScore', 
               'a.u.', 
               log = 'linear', 
               x_range=(0, 1), 
               bins = 50)
    
    save_path = os.path.join(plot_dir, "augmented_output_scores")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close() 
    
        
    