
import os
from argparse import ArgumentParser

# Third parties
import numpy as np
import pandas as pd
# Import from other modules
from model.common import fromYaml
from data.EOSdataset import DataSet



if ad_dataset:
        labels = {"background" : 0, "ato4l" :1, "hChToTauNu" : 2, 'hToTauTau':3,'leptoquark':4,'blackbox':5}
        inpath = '/eos/user/c/cebrown/RobustQML/AD_dataset/processed/'
        outpath = '/eos/user/c/cebrown/RobustQML/AD_dataset/processed/embedding'
        
    else:
        if scenario == 'minbias':
            labels = {"minbias":0}
        if scenario == 'QCD':
            labels = {"QCD_HT50tobb":2,"QCD_HT50toInf":1,"minbias":0}
        if scenario == 'all':
            labels = {"minbias" : 0, "QCD_HT50toInf" :1, "HH_4b" : 2, 'HH_bbgammagamma':3,'HH_bbtautau':4,'QCD_HT50tobb':5}
        if scenario == 'qcd_but':
            labels = {"QCD_HT50toInf" :1,'QCD_HT50tobb':5}
            
        path = '/eos/user/c/cebrown/RobustQML/training_data/'

dataset_list = []
    for datasets in labels.keys():
        data_test = DataSet.fromH5(path+datasets+'/train/')
        data_test.normalise()

        data_test.set_label(labels[datasets])
        dataset_list.append(data_test)
        training_columns = data_test.training_columns
        
    full_data_frame = pd.concat([dataset.data_frame for dataset in dataset_list])
    full_data_frame = full_data_frame.sample(frac=1)