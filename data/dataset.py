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
            
def split_dataframe(df, num_chunks = 24, max_events = -1): 
    chunks = list()
    chunk_size = math.ceil(max_events / num_chunks)
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

class DataSet:
    def __init__(self, name, orig=None):
        self.name = name
        self.pretty_name = name
        
        self.data_frame = pd.DataFrame
        
        self.max_number_of_jets = 4
        self.max_number_of_objects = 4
        self.max_number_of_constituents = 20
        
        self.jet_feature_list = ['L1T_JetPuppiAK4_PT','L1T_JetPuppiAK4_Eta','L1T_JetPuppiAK4_Phi']
        self.object_feature_list = ['L1T_MuonTight_PT','L1T_MuonTight_Eta','L1T_MuonTight_Phi',
                                    'L1T_Electron_PT','L1T_Electron_Eta','L1T_Electron_Phi']
        self.met_feature_list = ['L1T_PUPPIMET_MET','L1T_PUPPIMET_Phi','L1T_PUPPIMET_Eta']
        
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
    
    
    def process_chunk(self,chunk):
        dict_of_lists = {i:[] for i in self.all_features}
        for i,array in enumerate(chunk):
            for jet_feature in self.jet_feature_list:
                padded_jets = ak.pad_none(array[jet_feature],self.max_number_of_jets,axis=0,clip=True)
                padded_jets = ak.fill_none(padded_jets, 0)
                for j in range(self.max_number_of_jets):
                    dict_of_lists[jet_feature+str(j)].append(padded_jets[j])
                            
            for object_feature in self.object_feature_list:
                padded_objects = ak.pad_none(array[object_feature],self.max_number_of_objects,axis=0,clip=True)
                padded_objects = ak.fill_none(padded_objects, 0)
                for j in range(self.max_number_of_objects):
                    dict_of_lists[object_feature+str(j)].append(padded_objects[j])
                            
            for met_feature in self.met_feature_list:
                dict_of_lists[met_feature].append(array[met_feature][0]) 
                
            if i % 5000 == 0:
                print(f"{i} out of {len(chunk)}")

        return pd.DataFrame(dict_of_lists)
    
    def load_data_from_HF(self, filepath: str, max_number_of_events : int = 2000):
        starting_time = time.time()
        proc_num = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=proc_num)

        top_x_jets = [feature + str(i) for feature in self.jet_feature_list for i in range(self.max_number_of_jets)]
        top_x_objects = [feature + str(i) for feature in self.object_feature_list for i in range(self.max_number_of_objects)]
        self.all_features = self.met_feature_list + top_x_jets + top_x_objects
        
        all_features_dict = datasets.Features.from_dict({feature : {}  for feature in self.all_features})
        dataset = load_dataset("fastmachinelearning/collide-1m",
                       data_dir=filepath,
                       on_bad_files='warn',
                       columns = self.met_feature_list + self.jet_feature_list + self.object_feature_list,
                       num_proc = 4)
        
        print(f"cpu count: {proc_num}")
        
        
        if max_number_of_events == -1:
            max_number_of_events = len(dataset['train'])
        
        print(f"chunk size: {max_number_of_events/proc_num}")
        print(f"Total events: {max_number_of_events}")
            
        chunk = np.array_split(dataset['train'].select(range(0,max_number_of_events)), proc_num)
        results = pool.map(self.process_chunk, chunk)
        pool.close()
        pool.join()
        
        self.data_frame = pd.concat(results, axis=0, ignore_index=True)
        self.data_frame.reset_index(inplace=True)
        self.data_frame.dropna(inplace=True)

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
        
    def normalise(self):
        for column in self.data_frame.columns:
            self.data_frame[column]=(self.data_frame[column]-self.data_frame[column].mean())/self.data_frame[column].std()
        self.data_frame = self.data_frame.fillna(0)
