import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

smear = True
drop = False

FIGURE_SIZE = (17, 17)

input_dir = sys.argv[1]
file_name = sys.argv[2]

# Load the data
df = pd.read_csv(input_dir+file_name, on_bad_lines='warn')

# Clean the data: remove rows with N/A values and strip whitespace from column names
df = df.dropna()
df.columns = df.columns.str.strip()

print(df.head())

colours = ['r','g','b','orange']
#models = ['CAE','QAE','HW_QAE']
models = ['CEC','QEC']#

if smear:


    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for colour,model in zip(colours,models):
        model_df =  df[df['model'] == model]
        print(model_df.head())
        print(model_df['smear_percent'])
        print(model_df['auc_loss_augmented'])
        
        ax.errorbar(model_df['smear_percent'],model_df['auc_loss_augmented'],yerr=model_df['auc_loss_augmented_err'],label=model,color=colour,markersize=4,marker='s')
        ax.plot(model_df['smear_percent'],model_df['auc_loss_non_augmented'],label=model+' non augmented',linestyle='-',alpha=0.5,color=colour)
        ax.plot(model_df['smear_percent'],model_df['auc_loss_non_augmented']+model_df['auc_loss_non_augmented_err'],linestyle='--',alpha=0.1,color=colour)
        ax.plot(model_df['smear_percent'],model_df['auc_loss_non_augmented']-model_df['auc_loss_non_augmented_err'],linestyle='--',alpha=0.1,color=colour)

    ax.set_xscale('log')
    ax.set_ylabel("AUC")
    ax.set_xlabel("Smear")
    ax.legend(loc='lower left')
    ax.grid(True)
    save_path = os.path.join(input_dir, "ROC_vs_smear")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()


    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for colour,model in zip(colours,models):
        model_df =  df[df['model'] == model]
        ax.errorbar(model_df['smear_percent'],model_df['background_wd'],yerr=model_df['background_wd_err'],label=model,color=colour,markersize=4,marker='s')
        ax.plot(model_df['smear_percent'],model_df['background_wd'],label=model+' non augmented',linestyle='-',alpha=0.5,color=colour)

    ax.set_xscale('log')
    ax.set_ylabel("Background WD")
    ax.set_xlabel("Smear")
    ax.legend(loc='center right')
    ax.grid(True)
    save_path = os.path.join(input_dir, "background_wd_vs_smear")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()


    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    for colour,model in zip(colours,models):
        model_df =  df[df['model'] == model]
        ax.errorbar(model_df['smear_percent'],model_df['signal_wd'],yerr=model_df['signal_wd_err'],label=model,color=colour,markersize=4,marker='s')
    ax.set_xscale('log')
    ax.set_ylabel("Signal WD")
    ax.set_xlabel("Smear")
    ax.legend(loc='center right')
    ax.grid(True)
    save_path = os.path.join(input_dir, "signal_wd_vs_smear")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

if drop:
    
    for drop_type in ['smear_percent','pt_threshold_e','pt_threshold_mu']:
        fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
        for colour,model in zip(colours,models):
            model_df =  df[df['model'] == model]
            print(model_df.head())
            print(model_df[drop_type])
            print(model_df['auc_loss_augmented'])
            
            ax.errorbar(model_df[drop_type],model_df['auc_loss_augmented'],yerr=model_df['auc_loss_augmented_err'],label=model,color=colour,markersize=4,marker='s')
            ax.plot(model_df[drop_type],model_df['auc_loss_non_augmented'],label=model+' non augmented',linestyle='-',alpha=0.5,color=colour)
            ax.plot(model_df[drop_type],model_df['auc_loss_non_augmented']+model_df['auc_loss_non_augmented_err'],linestyle='--',alpha=0.1,color=colour)
            ax.plot(model_df[drop_type],model_df['auc_loss_non_augmented']-model_df['auc_loss_non_augmented_err'],linestyle='--',alpha=0.1,color=colour)

        ax.set_xscale('log')
        ax.set_ylabel("AUC")
        ax.set_xlabel("Smear")
        ax.legend(loc='lower left')
        ax.grid(True)
        save_path = os.path.join(input_dir, "ROC_vs_"+drop_type)
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
        for colour,model in zip(colours,models):
            model_df =  df[df['model'] == model]
            ax.errorbar(model_df[drop_type],model_df['background_wd'],yerr=model_df['background_wd_err'],label=model,color=colour,markersize=4,marker='s')
            ax.plot(model_df[drop_type],model_df['background_wd'],label=model+' non augmented',linestyle='-',alpha=0.5,color=colour)

        ax.set_xscale('log')
        ax.set_ylabel("Background WD")
        ax.set_xlabel("Smear")
        ax.legend(loc='center right')
        ax.grid(True)
        save_path = os.path.join(input_dir, "background_wd_vs_"+drop_type)
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
        for colour,model in zip(colours,models):
            model_df =  df[df['model'] == model]
            ax.errorbar(model_df[drop_type],model_df['signal_wd'],yerr=model_df['signal_wd_err'],label=model,color=colour,markersize=4,marker='s')
        ax.set_xscale('log')
        ax.set_ylabel("Signal WD")
        ax.set_xlabel("Smear")
        ax.legend(loc='center right')
        ax.grid(True)
        save_path = os.path.join(input_dir, "signal_wd_vs_"+drop_type)
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.close()