import os
from argparse import ArgumentParser

# Third parties
import numpy as np

# Import from other modules
from model.common import fromYaml
from data.dataset import DataSet

def train(model):

    # Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train = DataSet.fromH5('QCD_HT50toInf')

    # Get input shape
    input_shape = len(data_train.data_frame.columns)

    model.build_model(input_shape)
    model.compile_model()
    model.fit(data_train)
    model.save()

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

    model = fromYaml(args.yaml_config, args.output)
    train(model, args.output)