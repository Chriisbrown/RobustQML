import numpy as np
import pandas as pd
import awkward as ak
import time
import datetime
from pathlib import Path
import json
import os
from plot.basic import plot_histo
import matplotlib.pyplot as plt
import multiprocessing
import math


from sklearn.preprocessing import MinMaxScaler


def remove_feature(array,feature_list):    
    for feature in feature_list:
        array[feature] = 0
    return array


def normalise(array, columns):
    for column in columns:
        array[column]=  (array[column]-array[column].mean())/array[column].std()
        array[column] = array[column].fillna(0)

class DataSet:
    def __init__(self, name, orig=None):
        self.name = name
        self.pretty_name = name
        
        self.data_frame = pd.DataFrame
        
        self.columns = ['event_label','lepton_1_pT','lepton_1_eta','lepton_1_phi',
                                   'lepton_2_pT','lepton_2_eta','lepton_2_phi',
                                   'ET_miss','ET_miss_phi',
                                   'MET_rel','MET_axial','M_R',
                                   'M_TR_2', 'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']
        self.training_columns =  ['lepton_1_pT','lepton_1_eta','lepton_1_phi',
                                   'lepton_2_pT','lepton_2_eta','lepton_2_phi',
                                   'ET_miss','ET_miss_phi',
                                   'MET_rel','MET_axial','M_R',
                                   'M_TR_2', 'R', 'MT2', 'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']
        self.random_state = 4
        self.verbose = 1
        
        self.config_dict = {'name':self.name}
        
    def __add__(self, others : list ):
        frames = [other.data_frame for other in others]
        frames.append(self.data_frame)
        self.data_frame = pd.concat(frames)
        
        if orig is not None:
            self.copy_constructor(orig)

    def copy_constructor(self, orig):
        self.data_frame = orig.data_frame
    
    @classmethod
    def fromH5(cls, filepath):
        h5class = cls("From H5")
        h5class.load_h5(filepath=filepath)
        return h5class
    
    @classmethod
    def fromCSV(cls, filepath):
        csvclass = cls('from CSV')
        csvclass.load_data_from_csv(filepath=filepath)
        return csvclass
    
    def get_training_dataset(self):
        return self.data_frame[self.training_columns]
    
    def load_data_from_csv(self, filepath: str):
                
        self.data_frame = pd.read_csv(filepath,names=self.columns)
                
        self.data_frame.reset_index(inplace=True)
        
        self.config_dict["CSVLoaded"] = datetime.datetime.now().strftime(
            "%H:%M %d/%m/%y")
        self.config_dict["CSVfilepath"] = filepath
        self.config_dict["NumEvents"] = len(self.data_frame)

    def save_h5(self, filepath):
        Path(filepath).mkdir(parents=True, exist_ok=True)

        store = pd.HDFStore(filepath+'/full_Dataset.h5')
        store['df'] = self.data_frame  # save it
        store.close()

        self.config_dict["h5filepath"] = filepath
        self.config_dict["save_timestamp"] = datetime.datetime.now().strftime(
            "%H:%M %d/%m/%y")
        with open(filepath+'/config_dict.json', 'w') as f:
            json.dump(self.config_dict, f, indent=4)
        if self.verbose == 1:
            print("===Full Data Saved===")

    def load_h5(self, filepath):
        my_file = Path(filepath+'/full_Dataset.h5')
        if my_file.is_file():

            store = pd.HDFStore(filepath+'/full_Dataset.h5')
            self.data_frame = store['df']
            store.close()
        else:
            print("No Full dataset")

        my_file2 = Path(filepath+'/config_dict.json')
        if my_file2.is_file():
            with open(filepath+'/config_dict.json', 'r') as f:
                self.config_dict = json.load(f)
            self.config_dict["loaded_timestamp"] = datetime.datetime.now().strftime(
                "%H:%M %d/%m/%y")
            self.name = self.config_dict["name"]
        else:
            print("No Config Dict")

    def plot_inputs(self, filepath):
        plot_dir = os.path.join(filepath, "plots/")
        os.makedirs(plot_dir, exist_ok=True)
        
        print('plot features')
        for feature in self.training_columns:
            plot_histo(
                [self.data_frame[feature]],
                [feature],
                self.pretty_name,
                feature,
                'a.u',
                log = 'log',
                x_range=(np.min(self.data_frame[feature]), np.max(self.data_frame[feature])),
            )
            save_path = os.path.join(plot_dir, feature)
            plt.savefig(f"{save_path}.png", bbox_inches='tight')
            plt.close()

    def normalise(self,minmax=True):
        if minmax:
            for column in self.training_columns:
                self.data_frame[column]=(self.data_frame[column]-self.data_frame[column].min())/(self.data_frame[column].max() - self.data_frame[column].min())
        else:
            for column in self.training_columns:
                self.data_frame[column]=(self.data_frame[column]-self.data_frame[column].mean())/self.data_frame[column].std()
        self.data_frame = self.data_frame.fillna(0)
        
    def set_label(self, label):
        return    
        
    def phi_smear(self,scale=0.01):
        return
    
    def eta_smear(self,scale=0.01):
        return
        
    def pt_smear(self,scale=0.01):

        pt= self.data_frame[self.training_columns].to_numpy()

        non_zeros = pt != 0

        pt_smear = np.random.uniform(-1, 1, (pt.shape))*scale
        
        pt_matrix = pt_smear * non_zeros
        
        event_pt_smear = np.sum(pt_matrix,axis=1)
        
        pt = pt[:, :]+pt_matrix
        
        
        self.data_frame[self.training_columns] = pt