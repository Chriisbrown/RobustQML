import os
from argparse import ArgumentParser
from model.common import fromFolder
from data.dataset import DataSet

from plot import style

from basic import error_residual,plot_histo

import matplotlib.pyplot as plt
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
    
    output_dict = {"Minbias" : {}, "VBFHcc" :{}, "ggHbb" : {}, "QCDbb" : {}, "HH4b" : {}}
    
    for datasets in output_dict.keys():
        data_test = DataSet.fromH5('dataset/'+datasets)
        data_test.normalise()
        model_outputs = model.predict(data_test.data_frame)
        output_dict[datasets] = {'predictions' : model_outputs}
        
    
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