import os
from argparse import ArgumentParser
from model.common import fromFolder
from data.dataset import DataSet

from plot import style

from basic import error_residual, plot_histo, rates,efficiency, clusters

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
        
    minbias_rates = rates(type(model).__name__,'minbias',minbias_outputs)
    
    output_dict = {"Minbias" : {}, "VBFHcc" :{}, "ggHbb" : {}, "QCDbb" : {}, "HH4b" : {}, "QCD" : {}}
    
    for datasets in output_dict.keys():
        data_test = DataSet.fromH5('dataset/'+datasets)
        data_test.normalise()
        model_outputs = model.predict(data_test)
        efficiency_out = efficiency(type(model).__name__,datasets,model_outputs)
        output_dict[datasets] = {'predictions' : model_outputs,'efficiencies' : efficiency_out}
        
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
    for sample_name in output_dict.keys():
        ax.plot(minbias_rates,output_dict[sample_name]['efficiencies'], label=sample_name, linewidth=style.LINEWIDTH)
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
        data_test.normalise()
        data_test.set_label(labels[datasets])
        dataset_list.append(data_test)
        
    full_data_frame = pd.concat([dataset.data_frame.sample(n=5000) for dataset in dataset_list])
    full_data_frame = full_data_frame.sample(frac=1)
    distances = model.distance(full_data_frame[dataset_list[0].training_columns].iloc[0:25000])
    clusters(distances,labels=np.array(full_data_frame['event_label']),plot_dir=plot_dir, label_to_names={v: k for k, v in labels.items()})