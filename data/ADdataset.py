import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset
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
import h5py


from sklearn.preprocessing import MinMaxScaler


class DataSet:
    def __init__(self, name, orig=None):
        self.name = name
        self.pretty_name = name
        
        self.data_frame = pd.DataFrame
        
        self.max_number_of_jets = 10
        self.max_number_of_objects = 4
        
        self.jet_pt_cut = 0
        self.electron_pt_cut = 0
        self.muon_pt_cut = 0

            
        self.generate_feature_lists()
        self.random_state = 4
        self.verbose = 1
        
        self.config_dict = {'name':self.name}
        
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
    def fromOrginalH5(cls, filepath, with_labels=False):
        h5class = cls("From H5")
        h5class.load_original_h5(filepath=filepath,with_labels=with_labels)
        return h5class
    
    def get_training_dataset(self):
        return self.data_frame[self.training_columns]
    
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
            try:
                self.data_frame.reset_index(inplace=True)
            except ValueError:
                print('already reset')
                
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
            
    def load_original_h5(self, filepath,with_labels = False):
        my_file = Path(filepath+'.h5')
        with h5py.File(my_file, 'r') as file:
            full_data = file['Particles'][:,:,:-1]
            p = np.random.permutation(len(full_data))
            if with_labels:
                labels = file['EvtId'][:]
                labels = labels[p]
            full_data = full_data[p]
        columns = self.training_columns
        flattened = np.reshape(full_data, (full_data.shape[0], full_data.shape[1]*full_data.shape[2]))
        if with_labels:
            columns.append('event_label')
            flattened = np.c_[flattened,labels]
        self.data_frame = pd.DataFrame(flattened, columns=columns)
        self.data_frame.reset_index(inplace=True)
            

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
                [self.data_frame[muon_feature +  str(i)] for i in range(self.max_number_of_objects)],
                [muon_feature +  str(i) for i in range(self.max_number_of_objects)],
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
                [self.data_frame[electron_feature +  str(i)] for i in range(self.max_number_of_objects)],
                [electron_feature +  str(i) for i in range(self.max_number_of_objects)],
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
        
        eta_electron_columns = []
        eta_muon_columns = []
              
        for column in self.training_columns:
            if "Eta" in column:
                if column != 'L1T_PUPPIMET_Eta':
                    eta_columns.append(column)
                if "Jet" in column:
                    eta_jet_columns.append(column)
                elif "Electron" in column:
                    eta_electron_columns.append(column)
                elif "Muon" in column:
                    eta_muon_columns.append(column)
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
        
        self.data_frame[eta_electron_columns] = np.where(self.data_frame[eta_electron_columns]>3, 3, self.data_frame[eta_electron_columns])        
        self.data_frame[eta_electron_columns] = np.where(self.data_frame[eta_electron_columns]<-3, -3, self.data_frame[eta_electron_columns])

        self.data_frame[eta_muon_columns] = np.where(self.data_frame[eta_muon_columns]>2.4, 2.4, self.data_frame[eta_muon_columns])        
        self.data_frame[eta_muon_columns] = np.where(self.data_frame[eta_muon_columns]<-2.4, -2.4, self.data_frame[eta_muon_columns])

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
                    
                    deltapx = delta_pt * np.cos(delta_phi)
                    deltapy = delta_pt * np.sin(delta_phi)
                    
                    etpx = et * np.cos(phi)
                    etpy = et * np.sin(phi)
                    
                    sumpx = etpx - deltapx
                    sumpy = etpy - deltapy
                    
                    newet = np.sqrt(sumpx**2 + sumpy**2) 
                    etphi = np.atan2(sumpy , sumpx ) 
                    
                    self.data_frame.loc[i, 'L1T_PUPPIMET_MET'] = newet
                    self.data_frame.loc[i, 'L1T_PUPPIMET_Phi'] = etphi
                                        
                    self.data_frame.loc[i, pt_column+str(np.argmin(event))] = 0
                    self.data_frame.loc[i, phi_column+str(np.argmin(event))] = 0
                
