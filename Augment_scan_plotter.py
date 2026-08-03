import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from plot.style import set_style

set_style()

smear = True
drop = False

FIGURE_SIZE = (15, 12)
plt.rcParams.update({'font.size': 16})
line_width = 4
marker_size = 8

input_dir = sys.argv[1]
file_name = sys.argv[2]

# Load the data
df = pd.read_csv(input_dir+file_name, on_bad_lines='warn')

# Clean the data: remove rows with N/A values and strip whitespace from column names
df = df.dropna()
df.columns = df.columns.str.strip()

print(df.head())


colours = ['cornflowerblue','mediumvioletred','darkorange','orange']
colours = ['#FF2C55','#731963','#348AA7','#FE6100']
models = ['CAE','VAE','QAE','HW_QAE']
#models = ['CEC','QEC','LEC']#

model_names = {'CAE' : 'Classical Autoencoder',
               'VAE' : 'Variational Autoencoder',
               'QAE' : 'Quantum Autoencoder',
               'HW_QAE': 'Hardware Optimised Quantum Autoencoder'}

metric_names = {'wd': 'Wasserstein Distance',
                'mse': 'Mean Squared Error',
                'mae': 'Mean Absolute Error',
                'cos': 'Cosine Similarity',
                'euc': 'Euclidean Distance'}

if smear:


    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for colour,model in zip(colours,models):
        model_df =  df[df['model'] == model]
        print(model_df.head())
        print(model_df['smear_percent'])
        print(model_df['auc_loss_augmented'])
        
        # ax.errorbar(model_df['smear_percent'],model_df['auc_loss_augmented'],yerr=model_df['auc_loss_augmented_err'],
        #             label=model_names[model],color=colour,markersize=marker_size,marker='s',linewidth=line_width,capsize=cap_size,
        #             )
        
        lower = np.maximum(model_df['auc_loss_augmented'] - model_df['auc_loss_augmented'+'_err'], 0)   # clip at 0
        upper = model_df['auc_loss_augmented'] + model_df['auc_loss_augmented'+'_err']
        
        
        ax.plot(model_df['smear_percent'],model_df['auc_loss_augmented'],
                        label=model_names[model],color=colour,markersize=marker_size,marker='s',linewidth=line_width,)
        ax.fill_between(model_df['smear_percent'],
                            lower, 
                            upper, 
                            color=colour, alpha=0.2, linewidth=0)
        
        
        #ax.plot(model_df['smear_percent'],model_df['auc_loss_non_augmented'],label=model+' non augmented',linestyle='-',alpha=0.5,color=colour,linewidth=line_width)
        #ax.plot(model_df['smear_percent'],model_df['auc_loss_non_augmented']+model_df['auc_loss_non_augmented_err'],linestyle='--',alpha=0.1,color=colour,linewidth=line_width)
        #ax.plot(model_df['smear_percent'],model_df['auc_loss_non_augmented']-model_df['auc_loss_non_augmented_err'],linestyle='--',alpha=0.1,color=colour,linewidth=line_width)

    ax.set_xscale('log')
    ax.set_ylabel("ROC AUC",loc='top')
    ax.set_xlabel("Smear Percentage",loc='right')
    ax.legend(loc='lower left')
    ax.grid(True)
    ax.set_ylim(0.4,0.9)
    save_path = os.path.join(input_dir, "ROC_vs_smear")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()
    
    for distance in ['wd','mse','mae','cos','euc']:
        
        max_y = 0

        fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
        for colour,model in zip(colours,models):
            model_df =  df[df['model'] == model]
            
            current_max_y = np.max(model_df['background_'+distance]) 
            max_y = current_max_y if current_max_y > max_y else max_y
            
            
            lower = np.maximum(model_df['background_'+distance] - model_df['background_'+distance+'_err'], 0)   # clip at 0
            upper = model_df['background_'+distance] + model_df['background_'+distance+'_err']
            
            
            #ax.errorbar(model_df['smear_percent'],model_df['background_'+distance],yerr=model_df['background_'+distance+'_err'],
            #            label=model_names[model],color=colour,markersize=marker_size,marker='s',linewidth=line_width,capsize=cap_size)
            ax.plot(model_df['smear_percent'],model_df['background_'+distance],
                        label=model_names[model],color=colour,markersize=marker_size,marker='s',linewidth=line_width,)
            ax.fill_between(model_df['smear_percent'],
                            lower, 
                            upper, 
                            color=colour, alpha=0.2, linewidth=0)

            #ax.plot(model_df['smear_percent'],model_df['background_'+distance],label=model+' non augmented',linestyle='-',alpha=0.5,color=colour,linewidth=line_width)

        #ax.set_yscale('log')  
        ax.set_xscale('log')
        ax.set_ylabel("Background "+metric_names[distance],loc='top')
        ax.set_xlabel("Smear Percentage",loc='right')
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_ylim(-0.0001,max_y*1.5)
        save_path = os.path.join(input_dir, "background_"+distance+"_vs_smear")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
        for colour,model in zip(colours,models):
            model_df =  df[df['model'] == model]
            
            current_max_y = np.max(model_df['signal_'+distance]) 
            max_y = current_max_y if current_max_y > max_y else max_y
            
            # ax.errorbar(model_df['smear_percent'],model_df['signal_'+distance],yerr=model_df['signal_'+distance+'_err'],
            #             label=model,color=colour,markersize=marker_size,marker='s',linewidth=line_width,capsize=cap_size)
            lower = np.maximum(model_df['signal_'+distance] - model_df['signal_'+distance+'_err'], 0)   # clip at 0
            upper = model_df['signal_'+distance] + model_df['signal_'+distance+'_err']
            
            ax.plot(model_df['smear_percent'],model_df['signal_'+distance],
                        label=model_names[model],color=colour,markersize=marker_size,marker='s',linewidth=line_width,)
            ax.fill_between(model_df['smear_percent'],
                            lower, 
                            upper, 
                            color=colour, alpha=0.2, linewidth=0)
          
        #ax.set_yscale('log')  
        ax.set_xscale('log')
        ax.set_ylabel("Signal "+metric_names[distance],loc='top')
        ax.set_xlabel("Smear Percentage",loc='right')
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_ylim(-0.0001,max_y*1.5)
        save_path = os.path.join(input_dir, "signal_"+distance+"_vs_smear")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close()
