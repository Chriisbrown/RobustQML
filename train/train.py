import os
from argparse import ArgumentParser

# Third parties
import numpy as np
import pandas as pd
# Import from other modules
from model.common import fromYaml
from data.EOSdataset import DataSet
from model.gpu_utils import setup_gpu_memory_growth

def train(model ,normalise):

    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    
    #labels = {"QCD": 0, 'QCDbb':1}
    labels = {"QCD_HT50toInf":1}
    dataset_list = []
    for datasets in labels.keys():
        data_test = DataSet.fromH5('/eos/user/c/cebrown/RobustQML/training_data/'+datasets+'/train/')
        if normalise == "True":
            data_test.normalise()
        else:
            data_test.max_number_of_jets = 8
            data_test.max_number_of_electrons = 3
            data_test.max_number_of_muons = 3
            data_test.generate_feature_lists()
        data_test.set_label(labels[datasets])
        dataset_list.append(data_test)
        training_columns = data_test.training_columns
        
    full_data_frame = pd.concat([dataset.data_frame for dataset in dataset_list])
    full_data_frame = full_data_frame.sample(frac=1)
    
    input_shape = len(training_columns)
    input_length = len(full_data_frame)
    model.build_model(input_shape)
    model.compile_model(input_length)
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
    
    parser.add_argument(
        '-n', '--normalise', default='True', help='Normalise the input data?'
    )

    args = parser.parse_args()
    
    setup_gpu_memory_growth()
    
    model = fromYaml(args.yaml_config,args.output)
    train(model, args.normalise)
