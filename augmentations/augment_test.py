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

    args = parser.parse_args()
    
    model = fromFolder(args.output)

    plot_dir = os.path.join(model.output_directory, "plots/testing")
    os.makedirs(plot_dir, exist_ok=True)
    
    minbias = DataSet.fromH5('dataset/Minbias')
    minbias.normalise()
    minbias_outputs = model.predict(minbias)
    
    minbias_augment = DataSet.fromH5('dataset/Minbias_augmented')
    # minbias_augment.drop_a_soft_one('jet')
    # minbias_augment.eta_smear()
    # minbias_augment.pt_smear()
    # minbias_augment.phi_smear()
    minbias_augment.normalise()
    minbias_augment_outputs = model.predict(minbias_augment)
    
    HH4b_augment = DataSet.fromH5('dataset/HH4b_augmented')
    # HH4b_augment.drop_a_soft_one('jet')
    # HH4b_augment.eta_smear()
    # HH4b_augment.pt_smear()
    # HH4b_augment.phi_smear()
    # HH4b_augment.normalise()
    HH4b_augment_outputs = model.predict(HH4b_augment)

    HH4b = DataSet.fromH5('dataset/HH4b')
    HH4b.normalise()
    HH4b_outputs = model.predict(HH4b)   
    
    target_background = np.zeros(minbias_outputs.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)

    trueVal = np.concatenate((np.ones(HH4b_augment_outputs.shape[0]), target_background)) # anomaly=1, bkg=0
    predVal_loss = np.concatenate((HH4b_augment_outputs, minbias_augment_outputs))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

    auc_loss = auc(fpr_loss, tpr_loss)
                
    plt.plot(fpr_loss, tpr_loss, "-", label='hh4b augment'+' (auc = %.1f%%)'%(auc_loss*100.), color = style.colours[0])
    
    
    trueVal = np.concatenate((np.ones(HH4b_outputs.shape[0]), target_background)) # anomaly=1, bkg=0
    predVal_loss = np.concatenate((HH4b_outputs, minbias_outputs))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)

    auc_loss = auc(fpr_loss, tpr_loss)
                
    plt.plot(fpr_loss, tpr_loss, "-", label='hh4b'+' (auc = %.1f%%)'%(auc_loss*100.), color = style.colours[1])
            
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
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close() 
    
    
    
    efficiency_out = efficiency(type(model).__name__,'HHbb',HH4b_outputs)        
    minbias_rates = rates(type(model).__name__,'minbias',minbias_outputs)
    
    augment_efficiency_out = efficiency(type(model).__name__,'HHbb_augmented',HH4b_augment_outputs)        
    augment_minbias_rates = rates(type(model).__name__,'minbias_augmented',minbias_augment_outputs)
    
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    ax.plot(minbias_rates,efficiency_out, label='HHbbbb', linewidth=style.LINEWIDTH,color=style.colours[0])
    ax.plot(augment_minbias_rates,augment_efficiency_out, label='HHbbbb augmented', linewidth=style.LINEWIDTH,color=style.colours[1])
    
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
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    
        
    