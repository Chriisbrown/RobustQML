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


def add_multiplicities(array):        
    array['jet_multiplicity'] = np.count_nonzero(array['L1T_JetPuppiAK4_PT'])
    array['muon_multiplicity'] = np.count_nonzero(array['L1T_MuonTight_PT'])
    array['electron_multiplicity'] = np.count_nonzero(array['L1T_Electron_PT'])
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

def remove_feature(array,feature_list):    
    for feature in feature_list:
        array[feature] = 0
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
        
        self.max_number_of_jets = 10
        self.max_number_of_electrons = 4
        self.max_number_of_muons = 4
        
        self.jet_pt_cut = 20
        self.electron_pt_cut = 4 
        self.muon_pt_cut = 2
        
        self.jet_feature_list = ['L1T_JetPuppiAK4_PT','L1T_JetPuppiAK4_Eta','L1T_JetPuppiAK4_Phi']
        self.muon_feature_list = ['L1T_MuonTight_PT','L1T_MuonTight_Eta','L1T_MuonTight_Phi']
        self.electron_feature_list = ['L1T_Electron_PT','L1T_Electron_Eta','L1T_Electron_Phi']
        self.met_feature_list = ['L1T_PUPPIMET_MET','L1T_PUPPIMET_Eta','L1T_PUPPIMET_Phi']
        self.gen_feature_list = ['Gen_MissingET_MET']
        self.multiplicity_feature_list = ['jet_multiplicity','muon_multiplicity','electron_multiplicity']
        
        top_x_jets = [feature + str(i) for i in range(self.max_number_of_jets) for feature in self.jet_feature_list]
        top_x_muons = [feature + str(i) for i in range(self.max_number_of_muons) for feature in self.muon_feature_list ]
        top_x_electrons = [feature + str(i) for i in range(self.max_number_of_electrons) for feature in self.electron_feature_list ]
        self.all_features = self.met_feature_list + self.gen_feature_list+ self.multiplicity_feature_list + top_x_jets + top_x_muons + top_x_electrons
        
        self.training_columns =  top_x_jets + top_x_electrons + top_x_muons +  self.met_feature_list
        self.non_met_columns = [column for column in self.training_columns if "PT" in column ]
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
    def fromEOS(cls, filepath,):
        eosclass = cls("From EOS")
        eosclass.load_data_from_EOS(filepath=filepath)
        return eosclass
    
    @classmethod
    def fromH5(cls, filepath):
        h5class = cls("From H5")
        h5class.load_h5(filepath=filepath)
        return h5class
    
    def get_training_dataset(self):
        return self.data_frame[self.training_columns]
    
    def generate_feature_lists(self):
        
        self.jet_feature_list = ['L1T_JetPuppiAK4_PT','L1T_JetPuppiAK4_Eta','L1T_JetPuppiAK4_Phi']
        self.muon_feature_list = ['L1T_MuonTight_PT','L1T_MuonTight_Eta','L1T_MuonTight_Phi']
        self.electron_feature_list = ['L1T_Electron_PT','L1T_Electron_Eta','L1T_Electron_Phi']
        self.met_feature_list = ['L1T_PUPPIMET_MET','L1T_PUPPIMET_Eta','L1T_PUPPIMET_Phi']
        self.bonus_columns = ['L1T_PFCand_PT','L1T_PUPPIPart_PT']
        self.multiplicity_feature_list = ['jet_multiplicity','muon_multiplicity','electron_multiplicity']
        
        top_x_jets = [feature + str(i) for i in range(self.max_number_of_jets) for feature in self.jet_feature_list]
        top_x_muons = [feature + str(i) for i in range(self.max_number_of_objects) for feature in self.muon_feature_list ]
        top_x_electrons = [feature + str(i) for i in range(self.max_number_of_objects) for feature in self.electron_feature_list ]
        self.all_features = self.met_feature_list +  self.multiplicity_feature_list + top_x_jets + top_x_muons + top_x_electrons
        
        self.training_columns =   self.met_feature_list + top_x_electrons  + top_x_muons + top_x_jets 
        self.non_met_columns = [column for column in self.training_columns if "PT" in column ]
    
    def load_data_from_EOS(self, filepath: str):
        starting_time = time.time()
        
        columns = self.jet_feature_list + self.muon_feature_list + self.electron_feature_list + self.met_feature_list + self.gen_feature_list
        
        dataset = pd.read_parquet(filepath,columns=columns)
        
        feature_array = []

        for ievent in range(len(dataset['L1T_JetPuppiAK4_PT'])):
            feature_vector = np.zeros([self.max_number_of_jets * 3 + self.max_number_of_muons * 3 + self.max_number_of_electrons * 3 + len(self.met_feature_list) + len(self.gen_feature_list) + len(self.multiplicity_feature_list) ])
            offset = 0
            for ijet in range(self.max_number_of_jets):
                try:
                    feature_vector[offset + ijet*3 + 0] = dataset['L1T_JetPuppiAK4_PT'][ievent][ijet]
                    feature_vector[offset + ijet*3 + 1] = dataset['L1T_JetPuppiAK4_Eta'][ievent][ijet]
                    feature_vector[offset + ijet*3 + 2] = dataset['L1T_JetPuppiAK4_Phi'][ievent][ijet]
                except IndexError:
                    feature_vector[offset + ijet*3 + 0] = 0
                    feature_vector[offset + ijet*3 + 1] = 0
                    feature_vector[offset + ijet*3 + 2] = 0
            offset = self.max_number_of_jets*3
            for ielectron in range(self.max_number_of_electrons):
                try:
                    feature_vector[offset + ielectron*3 + 0] = dataset['L1T_Electron_PT'][ievent][ielectron]
                    feature_vector[offset + ielectron*3 + 1] = dataset['L1T_Electron_Eta'][ievent][ielectron]
                    feature_vector[offset + ielectron*3 + 2] = dataset['L1T_Electron_Phi'][ievent][ielectron]  
                except IndexError:
                    feature_vector[offset + ielectron*3 + 0] = 0
                    feature_vector[offset + ielectron*3 + 1] = 0
                    feature_vector[offset + ielectron*3 + 2] = 0
            offset = self.max_number_of_jets*3 + self.max_number_of_electrons*3  
            for imuon in range(self.max_number_of_muons):
                try:
                    feature_vector[offset + imuon*3 + 0] = dataset['L1T_MuonTight_PT'][ievent][imuon]
                    feature_vector[offset + imuon*3 + 1] = dataset['L1T_MuonTight_Eta'][ievent][imuon]
                    feature_vector[offset + imuon*3 + 2] = dataset['L1T_MuonTight_Phi'][ievent][imuon]
                except IndexError:    
                    feature_vector[offset + imuon*3 + 0] = 0
                    feature_vector[offset + imuon*3 + 1] = 0
                    feature_vector[offset + imuon*3 + 2] = 0

            offset = self.max_number_of_jets*3 + self.max_number_of_electrons*3 + self.max_number_of_muons*3
            feature_vector[offset + 0] = dataset['L1T_PUPPIMET_MET'][ievent]
            feature_vector[offset + 1] = 0
            feature_vector[offset + 2] = dataset['L1T_PUPPIMET_Phi'][ievent]
            feature_vector[offset + 3] = dataset['Gen_MissingET_MET'][ievent]
            
            feature_vector[offset + 4] = len(dataset['L1T_JetPuppiAK4_PT'][ievent])
            feature_vector[offset + 5] = len(dataset['L1T_Electron_PT'][ievent])
            feature_vector[offset + 6] = len(dataset['L1T_Electron_Phi'][ievent])
            
            

            feature_array.append(feature_vector)
        
        new_feature_array = np.stack( feature_array)
        
        self.data_frame = pd.DataFrame(new_feature_array, columns=self.training_columns + self.gen_feature_list + self.multiplicity_feature_list)
        
        for i in range(self.max_number_of_jets):
            mask = np.where((self.data_frame['L1T_JetPuppiAK4_PT'+str(i)] > self.jet_pt_cut),1,0)
            self.data_frame['L1T_JetPuppiAK4_PT'+str(i)] =  self.data_frame['L1T_JetPuppiAK4_PT'+str(i)] * mask
            self.data_frame['L1T_JetPuppiAK4_Eta'+str(i)] = self.data_frame['L1T_JetPuppiAK4_Eta'+str(i)] * mask
            self.data_frame['L1T_JetPuppiAK4_Phi'+str(i)] = self.data_frame['L1T_JetPuppiAK4_Phi'+str(i)]  * mask
            
        for i in range(self.max_number_of_muons):
            mask  = np.where((self.data_frame['L1T_MuonTight_PT'+str(i)] > self.muon_pt_cut),1,0)
            self.data_frame['L1T_MuonTight_PT'+str(i)] =  self.data_frame['L1T_MuonTight_PT'+str(i)] * mask
            self.data_frame['L1T_MuonTight_Eta'+str(i)] = self.data_frame['L1T_MuonTight_Eta'+str(i)] * mask
            self.data_frame['L1T_MuonTight_Phi'+str(i)] = self.data_frame['L1T_MuonTight_Phi'+str(i)]  * mask
            
        for i in range(self.max_number_of_electrons):
            mask = np.where((self.data_frame['L1T_Electron_PT'+str(i)] > self.electron_pt_cut),1,0)
            self.data_frame['L1T_Electron_PT'+str(i)] =  self.data_frame['L1T_Electron_PT'+str(i)] * mask
            self.data_frame['L1T_Electron_Eta'+str(i)] = self.data_frame['L1T_Electron_Eta'+str(i)] * mask
            self.data_frame['L1T_Electron_Phi'+str(i)] = self.data_frame['L1T_Electron_Phi'+str(i)]  * mask

        #print(self.data_frame.describe())
        
        self.data_frame = self.data_frame[self.data_frame[self.non_met_columns].sum(axis=1) != 0]
        self.data_frame.reset_index(inplace=True)
        
        self.config_dict["EOSLoaded"] = datetime.datetime.now().strftime(
            "%H:%M %d/%m/%y")
        self.config_dict["EOSfilepath"] = filepath
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
            plt.close()
            
        print('plot muon features')
        for muon_feature in self.muon_feature_list:
            plot_histo(
                [self.data_frame[muon_feature +  str(i)] for i in range(self.max_number_of_muons)],
                [muon_feature +  str(i) for i in range(self.max_number_of_muons)],
                self.pretty_name,
                muon_feature,
                'a.u',
                log = 'log',
                x_range=(np.min(self.data_frame[muon_feature + "0"]), np.max(self.data_frame[muon_feature + "0"])),
            )
            save_path = os.path.join(plot_dir, muon_feature)
            plt.savefig(f"{save_path}.png", bbox_inches='tight')
            plt.close()
            
        print('plot electron features')
        for electron_feature in self.electron_feature_list:
            plot_histo(
                [self.data_frame[electron_feature +  str(i)] for i in range(self.max_number_of_electrons)],
                [electron_feature +  str(i) for i in range(self.max_number_of_electrons)],
                self.pretty_name,
                electron_feature,
                'a.u',
                log = 'log',
                x_range=(np.min(self.data_frame[electron_feature + "0"]), np.max(self.data_frame[electron_feature + "0"])),
            )
            save_path = os.path.join(plot_dir, electron_feature)
            plt.savefig(f"{save_path}.png", bbox_inches='tight')
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
        self.data_frame['event_label'] = (np.ones(len(self.data_frame)) * label).astype(int)
        
        
    def uniformity(self,label, n_bins=100, samples_per_bin=100):
        self.data_frame['bin'] = pd.cut(self.data_frame[label], bins=n_bins)

        # Sample equally from each bin
        sampled_df = self.data_frame.groupby('bin', observed=True).sample(
            n=samples_per_bin, 
            replace=True  # set to True if you need more samples than available
        )

        # Remove the bin column if you don't need it
        self.data_frame = sampled_df.drop('bin', axis=1)
        self.data_frame.reset_index(inplace=True)

                
        
    def phi_rotate(self):
        # Get the phi features
        phi_columns = []
        for column in self.training_columns:
            if "Phi" in column:
                phi_columns.append(column)
        # Get the pT features, only do the phi rotation if the object has non zero pT
        phi= self.data_frame[phi_columns].to_numpy()
        
        
        pt_columns = []
        for column in self.training_columns:
            if (("PT" in column) or (column == "L1T_PUPPIMET_MET")):
                pt_columns.append(column)

        pt = self.data_frame[pt_columns].to_numpy()
        
        non_zeros = pt != 0
        
        rot_angle = np.random.rand(len(phi)) *2*np.pi
        rot_angle = rot_angle - np.pi
        
        ones  = np.ones_like(phi)
        angle_matrix = np.einsum('ij,i->ij', ones, rot_angle)
        angle_matrix = angle_matrix * non_zeros
        phi = phi[:, :]+angle_matrix
        
        phi = np.where(phi > np.pi, phi - 2*np.pi, phi)
        phi = np.where(phi < -np.pi, phi+2*np.pi, phi)
        
        self.data_frame[phi_columns] = phi
        
        
    def phi_smear(self,scale=0.01):
        # Get the phi features
        phi_columns = []
        for column in self.training_columns:
            if "Phi" in column:
                if column != 'L1T_PUPPIMET_Phi':
                    phi_columns.append(column)
        # Get the pT features, only do the phi rotation if the object has non zero pT
        phi= self.data_frame[phi_columns].to_numpy()
        
        
        pt_columns = []
        for column in self.training_columns:
            if (("PT" in column)):
                pt_columns.append(column)

        pt = self.data_frame[pt_columns].to_numpy()
        
        non_zeros = pt != 0
        rot_angle = np.random.uniform(-1, 1, (phi.shape))*scale*2*np.pi

        
        angle_matrix = rot_angle * non_zeros
        event_phi_change = np.sum(angle_matrix,axis=1)
        phi = phi[:, :]+angle_matrix


        phi = np.where(phi>np.pi, phi - 2*np.pi,phi)
        phi = np.where(phi<-np.pi, phi+2*np.pi, phi)

        self.data_frame[phi_columns] = phi
        
        self.data_frame['L1T_PUPPIMET_Phi'] = self.data_frame['L1T_PUPPIMET_Phi'] + event_phi_change
        self.data_frame['L1T_PUPPIMET_Phi'] = np.where(self.data_frame['L1T_PUPPIMET_Phi']>np.pi,  self.data_frame['L1T_PUPPIMET_Phi'] - 2*np.pi,self.data_frame['L1T_PUPPIMET_Phi'])        
        self.data_frame['L1T_PUPPIMET_Phi'] = np.where(self.data_frame['L1T_PUPPIMET_Phi']<-np.pi, self.data_frame['L1T_PUPPIMET_Phi']+2*np.pi,self.data_frame['L1T_PUPPIMET_Phi'])
        
        print(self.data_frame.describe())
        
    def pt_smear(self,scale=0.01):
        # Get the phi features
        pt_columns = []
        for column in self.training_columns:
            if "PT" in column:
                pt_columns.append(column)
        # Get the pT features, only do the phi rotation if the object has non zero pT
        pt= self.data_frame[pt_columns].to_numpy()

        
        non_zeros = pt != 0

        pt_smear = np.random.uniform(-1, 1, (pt.shape))*scale
        
        pt_matrix = pt_smear * non_zeros
        
        event_pt_smear = np.sum(pt_matrix,axis=1)
        
        pt = pt[:, :]+pt_matrix
        

        pt = np.where(pt<0, 0 , pt)
        pt = np.where(pt>2000, 2000, pt)
    
        
        self.data_frame[pt_columns] = pt
        
        self.data_frame['L1T_PUPPIMET_MET'] = self.data_frame['L1T_PUPPIMET_MET'] + event_pt_smear
        self.data_frame['L1T_PUPPIMET_MET'] = np.where(self.data_frame['L1T_PUPPIMET_MET']>2000, 2000, self.data_frame['L1T_PUPPIMET_MET'])        
        self.data_frame['L1T_PUPPIMET_MET'] = np.where(self.data_frame['L1T_PUPPIMET_MET']<0, 0, self.data_frame['L1T_PUPPIMET_MET'])
        
    def eta_smear(self,scale=0.01):
        # Get the phi features
        eta_columns = []
        
        eta_jet_columns = []
        eta_object_columns = []
              
        for column in self.training_columns:
            if "Eta" in column:
                if column != 'L1T_PUPPIMET_Eta':
                    eta_columns.append(column)
                if "Jet" in column:
                    eta_jet_columns.append(column)
                elif "Electron" or "Muon" in column:
                    eta_object_columns.append(column)
        # Get the pT features, only do the phi rotation if the object has non zero pT
        eta= self.data_frame[eta_columns].to_numpy()
        
        pt_columns = []
        for column in self.training_columns:
            if (("PT" in column)):
                pt_columns.append(column)

        pt = self.data_frame[pt_columns].to_numpy()
        
        non_zeros = pt != 0
        eta_smear = np.random.uniform(-1, 1, (eta.shape))*scale
                
        eta_matrix = eta_smear * non_zeros
        
        eta = eta[:, :]+eta_matrix
        
        self.data_frame[eta_columns] = eta
        
        self.data_frame[eta_jet_columns] = np.where(self.data_frame[eta_jet_columns]>4.5, 4.5, self.data_frame[eta_jet_columns])        
        self.data_frame[eta_jet_columns] = np.where(self.data_frame[eta_jet_columns]<-4.5, -4.5, self.data_frame[eta_jet_columns])
        
        self.data_frame[eta_object_columns] = np.where(self.data_frame[eta_object_columns]>2.4, 2.4, self.data_frame[eta_object_columns])        
        self.data_frame[eta_object_columns] = np.where(self.data_frame[eta_object_columns]<-2.4, -2.4, self.data_frame[eta_object_columns])

    def eta_flip(self):
        # Get the phi features
        eta_columns = []
        for column in self.training_columns:
            if "Eta" in column:
                eta_columns.append(column)
        # Get the pT features, only do the phi rotation if the object has non zero pT
        self.data_frame[eta_columns] = self.data_frame[eta_columns]*-1
        
        
    def drop_a_soft_one(self,object_type):
        
        match object_type:
            case 'jet':
                pt_column = 'L1T_JetPuppiAK4_PT'
                phi_column = 'L1T_JetPuppiAK4_Phi'
            case 'e':
                pt_column = 'L1T_Electron_PT'
                phi_column = 'L1T_Electron_Phi'
            case 'mu':
                pt_column = 'L1T_MuonTight_PT'
                phi_column = 'L1T_MuonTight_Eta'
                
        pt_columns = []
        for column in self.training_columns:
            if (("PT" in column) and ("Jet" in column)):
                pt_columns.append(column)
        
        pt = self.data_frame[pt_columns].to_numpy()
        
        for i,event in enumerate(pt):
            event = np.trim_zeros(event)
            if len(event) > 1:
                delta_pt = self.data_frame[pt_column+str(np.argmin(event))][i]
                delta_phi = self.data_frame[phi_column+str(np.argmin(event))][i]
                
                if delta_pt < 20:
                
                    et = self.data_frame['L1T_PUPPIMET_MET'][i]
                    phi = self.data_frame['L1T_PUPPIMET_Phi'][i]
                    
                    #print(et)
                    #print(phi)
                    
                    #print(delta_pt)
                    #print(delta_phi)
                                
                    sumpx = np.sqrt(et**2 / (1 + np.tan(phi)**2))
                    sumpy = np.tan(phi) * sumpx

                    deltapx = delta_pt * np.cos(delta_phi);
                    deltapy = delta_pt * np.sin(delta_phi);
                
                    newet = np.sqrt((sumpx - deltapx)**2 + (sumpy - deltapy)**2)
                    etphi = np.arctan2(sumpy - deltapx, sumpx - deltapy)
                    
                    #print(newet)
                    #print(etphi)
                    
                    self.data_frame.loc[i, 'L1T_PUPPIMET_MET'] = newet
                    self.data_frame.loc[i, 'L1T_PUPPIMET_Phi'] = etphi
                    
                    self.data_frame['L1T_PUPPIMET_Phi'] = np.where(self.data_frame['L1T_PUPPIMET_Phi']>np.pi,  self.data_frame['L1T_PUPPIMET_Phi'] - 2*np.pi,self.data_frame['L1T_PUPPIMET_Phi'])        
                    self.data_frame['L1T_PUPPIMET_Phi'] = np.where(self.data_frame['L1T_PUPPIMET_Phi']<-np.pi, self.data_frame['L1T_PUPPIMET_Phi']+2*np.pi,self.data_frame['L1T_PUPPIMET_Phi'])
                    
                    self.data_frame.loc[i, pt_column+str(np.argmin(event))] = 0
                    self.data_frame.loc[i, phi_column+str(np.argmin(event))] = 0
                
