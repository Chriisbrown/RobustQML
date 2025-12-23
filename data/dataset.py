import numpy as np
import pandas as pd
from datasets import load_dataset
import awkward as ak
import time
import datetime
from pathlib import Path
import json

class DataSet:
    def __init__(self, name, orig=None):
        self.name = name
        
        self.data_frame = pd.DataFrame
        
        self.max_number_of_jets = 4
        self.max_number_of_objects = 4
        self.max_number_of_constituents = 20
        
        self.jet_feature_list = ['L1T_JetPuppiAK4_PT','L1T_JetPuppiAK4_Eta','L1T_JetPuppiAK4_Phi']
        self.object_feature_list = ['L1T_MuonTight_PT','L1T_MuonTight_Eta','L1T_MuonTight_Phi',
                                    'L1T_Electron_PT','L1T_Electron_Eta','L1T_Electron_Phi']
        self.met_feature_list = ['L1T_PUPPIMET_MET','L1T_PUPPIMET_Phi','L1T_PUPPIMET_Eta',]
        
        self.random_state = 4
        self.verbose = 1
        
        self.config_dict = {'name':self.name}
        
        if orig is not None:
            self.copy_constructor(orig)

    def copy_constructor(self, orig):
        self.data_frame = orig.data_frame
        

    @classmethod
    def fromHF(cls, filepath,max_number_of_events):
        hfclass = cls("From Hugging Face")
        hfclass.load_data_from_HF(filepath=filepath,max_number_of_events=max_number_of_events)
        return hfclass
    
    @classmethod
    def fromH5(cls, filepath):
        h5class = cls("From H5")
        h5class.load_h5(filepath=filepath)
        return h5class
    
    def load_data_from_HF(self, filepath: str, max_number_of_events : int = 2000):
        starting_time = time.time()
        dataset = load_dataset("fastmachinelearning/collide-1m",
                       data_dir=filepath,
                       streaming=True)
        
        top_x_jets = [feature + str(i) for feature in self.jet_feature_list for i in range(self.max_number_of_jets)]
        top_x_objects = [feature + str(i) for feature in self.object_feature_list for i in range(self.max_number_of_objects)]
        all_features = self.met_feature_list + top_x_jets + top_x_objects
        
        dict_of_lists = {i:[] for i in all_features}
        EventDF = pd.DataFrame()
        for i,array in enumerate(dataset["train"]):
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
                    
            if i % 1000 == 0:
                print(i)
                tempdf = pd.DataFrame(dict_of_lists)
                EventDF = pd.concat([EventDF,tempdf],ignore_index=False)
                print(EventDF.describe())
                del [tempdf, dict_of_lists]
                
                dict_of_lists = {i:[] for i in all_features}
            if i > max_number_of_events:
                break

        EventDF.reset_index(inplace=True)
        EventDF.dropna(inplace=True)

        self.data_frame = EventDF
        
        print(self.data_frame.head())
        
        del [EventDF, dict_of_lists]
        
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
        self.data_frame = None
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
