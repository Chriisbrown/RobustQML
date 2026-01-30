import os
from argparse import ArgumentParser

# Third parties
import numpy as np
import pandas as pd
# Import from other modules
from model.common import fromYaml
from data.dataset import DataSet

def train(model):

    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    
    labels = {"QCD": 0, 'QCDbb':1, 'DY':5}
    dataset_list = []
    for datasets in labels.keys():
        data_test = DataSet.fromH5('dataset/'+datasets)
        data_test.normalise()
        data_test.set_label(labels[datasets])
        dataset_list.append(data_test)
        training_columns = data_test.training_columns
        
    full_data_frame = pd.concat([dataset.data_frame for dataset in dataset_list])
    full_data_frame = full_data_frame.sample(frac=1)
    
    input_shape = len(training_columns)
    model.build_model(input_shape)
    model.compile_model()
    model.fit(full_data_frame,training_columns)
    model.save()
    model.plot_loss()

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        '-o', '--output', default='output/autoencoder', help='Output model directory path, also save evaluation plots'
    )
    parser.add_argument(
        '-y', '--yaml_config', default='model/configs/AutoEncoder.yaml', help='YAML config for model'
    )

    args = parser.parse_args()

    model = fromYaml(args.yaml_config,args.output)
    train(model)