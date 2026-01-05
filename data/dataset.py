import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset
import awkward as ak
import time
import datetime
from pathlib import Path
import json
import dask.dataframe as dd
import os
from plot.basic import plot_histo
import matplotlib.pyplot as plt
import multiprocessing
import math

from sklearn.preprocessing import MinMaxScaler


def add_multiplicities(array):        
    array['jet_multiplicity'] = ak.count_nonzero(array['L1T_JetPuppiAK4_PT'])
    array['muon_multiplicity'] = ak.count_nonzero(array['L1T_MuonTight_PT'])
    array['electron_multiplicity'] = ak.count_nonzero(array['L1T_Electron_PT'])
    # array['pfcand_multiplicity'] = ak.count_nonzero(array['L1T_PFCand_PT'])
    # array['puppicand_multiplicity'] = ak.count_nonzero(array['L1T_PUPPIPart_PT'])
    return array

def pad_jets(array,jet_feature_list,max_number_of_jets):
    for jet_feature in jet_feature_list:
        padded_jets = ak.pad_none(array[jet_feature],max_number_of_jets,axis=0,clip=True)
        padded_jets = ak.fill_none(padded_jets, 0)
        padded_jets = ak.values_astype(padded_jets, np.float64)
        for j in range(max_number_of_jets):
            array[jet_feature+str(j)] = (padded_jets[j])
    return array

def pad_objects(array,object_feature_list,max_number_of_objects):    
    for object_feature in object_feature_list:
        padded_objects = ak.pad_none(array[object_feature],max_number_of_objects,axis=0,clip=True)
        padded_objects = ak.fill_none(padded_objects, 0)
        padded_objects = ak.values_astype(padded_objects, np.float64)
        for j in range(max_number_of_objects):
            array[object_feature+str(j)] = (padded_objects[j])
    return array

def process_objects(array,feature_list):    
    for feature in feature_list:
        padded = ak.pad_none(array[feature],1,axis=0,clip=True)
        padded = ak.fill_none(padded, 0)
        padded = ak.values_astype(padded, np.float64)
        for j in range(1):
            array[feature] = (padded[j])
    return array


def normalise(array, columns):
    for column in columns:
            array[column]=(array[column]-array[column].mean())/array[column].std()
            array[column] = array[column].fillna(0)

class DataSet:
    def __init__(self, name, orig=None):
        self.name = name
        self.pretty_name = name
        
        self.data_frame = pd.DataFrame
        
        self.max_number_of_jets = 5
        self.max_number_of_objects = 4
        self.max_number_of_constituents = 20
        
        self.jet_feature_list = ['L1T_JetPuppiAK4_PT','L1T_JetPuppiAK4_Eta','L1T_JetPuppiAK4_Phi']
        self.object_feature_list = ['L1T_MuonTight_PT','L1T_MuonTight_Eta','L1T_MuonTight_Phi',
                                    'L1T_Electron_PT','L1T_Electron_Eta','L1T_Electron_Phi']
        self.met_feature_list = ['L1T_PUPPIMET_MET','L1T_PUPPIMET_Phi','L1T_PUPPIMET_Eta']
        self.bonus_columns = ['L1T_PFCand_PT','L1T_PUPPIPart_PT']
        self.gen_feature_list = ['FullReco_GenMissingET_MET']
        self.multiplicity_feature_list = ['jet_multiplicity','muon_multiplicity','electron_multiplicity']
        
        top_x_jets = [feature + str(i) for feature in self.jet_feature_list for i in range(self.max_number_of_jets)]
        top_x_objects = [feature + str(i) for feature in self.object_feature_list for i in range(self.max_number_of_objects)]
        self.all_features = self.met_feature_list + self.gen_feature_list+ self.multiplicity_feature_list + top_x_jets + top_x_objects
        
        self.training_columns =  top_x_jets + top_x_objects + self.met_feature_list

        self.random_state = 4
        self.verbose = 1
        
        self.config_dict = {'name':self.name}
        
        if orig is not None:
            self.copy_constructor(orig)

    def copy_constructor(self, orig):
        self.data_frame = orig.data_frame
    
    @classmethod
    def fromHF(cls, filepath,max_number_of_events=-1):
        hfclass = cls("From Hugging Face")
        hfclass.load_data_from_HF(filepath=filepath,max_number_of_events=max_number_of_events)
        return hfclass
    
    @classmethod
    def fromH5(cls, filepath):
        h5class = cls("From H5")
        h5class.load_h5(filepath=filepath)
        return h5class
    
    def get_training_dataset(self):
        return self.data_frame[self.training_columns]
    
    def stream_data_from_HF(self, filepath: str):
        dataset = load_dataset("fastmachinelearning/collide-1m",
                       data_dir=filepath,
                       on_bad_files='warn',
                       streaming=True,
                       columns = self.met_feature_list + self.jet_feature_list + self.object_feature_list + self.gen_feature_list + self.bonus_columns,
        )
        dataset = dataset.map(add_multiplicities,remove_columns=self.bonus_columns)
        dataset = dataset.map(pad_jets,fn_kwargs = {'jet_feature_list' : self.jet_feature_list,'max_number_of_jets' :self.max_number_of_jets},remove_columns=self.jet_feature_list)
        dataset = dataset.map(pad_objects,fn_kwargs = {'object_feature_list' : self.object_feature_list,'max_number_of_objects' : self.max_number_of_objects}, remove_columns=self.object_feature_list)        
        dataset = dataset.map(process_objects,fn_kwargs = {'feature_list' : self.gen_feature_list + self.met_feature_list})   
        dataset = dataset.map(normalise,fn_kwargs = {'columns' : self.training_columns})
        return dataset

    def load_data_from_HF(self, filepath: str, max_number_of_events : int = 2000):
        starting_time = time.time()
        proc_num = multiprocessing.cpu_count()

        dataset = load_dataset("fastmachinelearning/collide-1m",
                       data_dir=filepath,
                       on_bad_files='warn',
                       columns = self.met_feature_list + self.jet_feature_list + self.object_feature_list + self.gen_feature_list + self.bonus_columns,
                       num_proc = 16)
        
        print(f"cpu count: {proc_num}")
        print("Add multiplicities")
        dataset = dataset.map(add_multiplicities,num_proc=proc_num,remove_columns=self.bonus_columns)
        print("Pad Jets")
        dataset = dataset.map(pad_jets,fn_kwargs = {'jet_feature_list' : self.jet_feature_list,'max_number_of_jets' :self.max_number_of_jets}, num_proc=proc_num,remove_columns=self.jet_feature_list)
        print("Pad Objects")
        dataset = dataset.map(pad_objects,fn_kwargs = {'object_feature_list' : self.object_feature_list,'max_number_of_objects' : self.max_number_of_objects}, num_proc=proc_num,remove_columns=self.object_feature_list)        
        print("Process other columns")
        dataset = dataset.map(process_objects,fn_kwargs = {'feature_list' : self.gen_feature_list + self.met_feature_list}, num_proc=proc_num)   
        
        self.data_frame = dataset['train'].to_pandas()
        print(self.data_frame.describe())
        
        self.config_dict["HFLoaded"] = datetime.datetime.now().strftime(
            "%H:%M %d/%m/%y")
        self.config_dict["HFfilepath"] = filepath
        self.config_dict["NumEvents"] = len(self.data_frame)
        if self.verbose == 1:
            print("Event Reading Complete, read: ",
                  len(self.data_frame), " events in ", time.time() - starting_time, " seconds")

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
        
        print('plot jet features')
        for jet_feature in self.jet_feature_list:
            plot_histo(
                [self.data_frame[jet_feature +  str(i)] for i in range(self.max_number_of_jets)],
                [jet_feature +  str(i) for i in range(self.max_number_of_jets)],
                self.pretty_name,
                jet_feature,
                'a.u',
                log = 'log',
                x_range=(np.min(self.data_frame[jet_feature + "0"]), np.max(self.data_frame[jet_feature + "0"])),
            )
            save_path = os.path.join(plot_dir, jet_feature)
            plt.savefig(f"{save_path}.png", bbox_inches='tight')
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
            plt.close()
            
        print('plot object features')
        for obj_feature in self.object_feature_list:
            plot_histo(
                [self.data_frame[obj_feature +  str(i)] for i in range(self.max_number_of_objects)],
                [obj_feature +  str(i) for i in range(self.max_number_of_objects)],
                self.pretty_name,
                obj_feature,
                'a.u',
                log = 'log',
                x_range=(np.min(self.data_frame[obj_feature + "0"]), np.max(self.data_frame[obj_feature + "0"])),
            )
            save_path = os.path.join(plot_dir, obj_feature)
            plt.savefig(f"{save_path}.png", bbox_inches='tight')
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
            plt.close()
        
        print('plot met features')
        for met_feature in self.met_feature_list:
            plot_histo(
                [self.data_frame[met_feature] ],
                [met_feature],
                self.pretty_name,
                met_feature,
                'a.u',
                log = 'log',
                x_range=(np.min(self.data_frame[met_feature]), np.max(self.data_frame[met_feature])),
            )
            save_path = os.path.join(plot_dir, met_feature)
            plt.savefig(f"{save_path}.png", bbox_inches='tight')
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
            plt.close()
        
        print('plot gen features')
        for gen_feature in self.gen_feature_list:
            plot_histo(
                [self.data_frame[gen_feature] ],
                [gen_feature],
                self.pretty_name,
                gen_feature,
                'a.u',
                log = 'log',
                x_range=(np.min(self.data_frame[gen_feature]), np.max(self.data_frame[gen_feature])),
            )
            save_path = os.path.join(plot_dir, gen_feature)
            plt.savefig(f"{save_path}.png", bbox_inches='tight')
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
            plt.close()
        
        print('plot multiplicity features')
        for multiplicity_feature in self.multiplicity_feature_list:
            plot_histo(
                [self.data_frame[multiplicity_feature] ],
                [multiplicity_feature],
                self.pretty_name,
                multiplicity_feature,
                'a.u',
                log = 'log',
                x_range=(np.min(self.data_frame[multiplicity_feature]), np.max(self.data_frame[multiplicity_feature])),
            )
            save_path = os.path.join(plot_dir, multiplicity_feature)
            plt.savefig(f"{save_path}.png", bbox_inches='tight')
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
            plt.close()
        
    def normalise(self,minmax=True):
        if minmax:
            for column in self.training_columns:
                self.data_frame[column]=(self.data_frame[column]-self.data_frame[column].min())/(self.data_frame[column].max() - self.data_frame[column].min())
        else:
            for column in self.training_columns:
                self.data_frame[column]=(self.data_frame[column]-self.data_frame[column].mean())/self.data_frame[column].std()
        self.data_frame = self.data_frame.fillna(0)
