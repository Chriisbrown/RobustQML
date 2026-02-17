import os

import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel
from data.ADdataset import DataSet
import pandas as pd

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import keras
from keras.models import load_model
from keras.layers import Dense

import tensorflow as tf

import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane import numpy as np

from plot.basic import loss_history, plot_2d


import matplotlib.pyplot as plt
from plot import style

def uniform_sample(array,n_bins,samples_per_bin):
    # Create bins and assign each value to a bin
    bin_edges = np.linspace(np.min(array[np.nonzero(array)]), array.max(), n_bins + 1)
    bin_indices = np.digitize(array, bin_edges[:-1]) - 1
    # Sample equally from each bin
    sampled_indices = []
    for bin_num in range(n_bins):
        bin_mask = bin_indices == bin_num
        bin_positions = np.where(bin_mask)[0]
            
        if len(bin_positions) > 0:
            n_samples = min(samples_per_bin, len(bin_positions))
            sampled = np.random.choice(bin_positions, size=n_samples, replace=False)
            sampled_indices.append(sampled)

    sampled_indices = np.concatenate(sampled_indices)
    return(sampled_indices)

# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('PennyLaneQAEModel')
class PennyLaneQAEModel(ADModel):

    """PennyLaneQAEModel class

    Args:
        PennyLaneQAEModel (_type_): Base class of a PennyLaneQAEModel
    """
    
    def __init__(self,output_dir):
        super().__init__(output_dir)
    
        self.number_of_jets = 10
        self.number_of_muons = 4
        self.number_of_electrons = 4
        
        self.jet_indices = 0,self.number_of_jets
        self.muon_indices = self.jet_indices[1] ,self.jet_indices[1] + self.number_of_muons
        self.electron_indices = self.muon_indices[1] ,self.muon_indices[1] + self.number_of_electrons
        self.sum_indices = self.electron_indices[1] 

    def build_model(self, inputs_shape: tuple):
        """build model override

        Args:
            inputs_shape (tuple): Shape of the input
        """
        total_object = int(inputs_shape / 3) - 1
        
        nq = total_object
        nt = int(nq - self.model_config['latent_dim'])        
        self.nq = nq
        
        self.q_wires = [f"q{i}" for i in range(nq)]
        self.t_wires = [f"t{i}" for i in range(nt)]
        self.a_wire = "a"
                
        self.dev = qml.device("lightning.qubit", wires=self.q_wires + self.t_wires + [self.a_wire])
        @qml.set_shots(shots=None)
        @qml.qnode(self.dev)
        def circuit(weights , x1, x2 ,return_projector=False):
            """
            featsy: shape (nq,)
            featsx: shape (nq,)
            theta:  shape (4*nq,)
            """
            
            nq = len(self.q_wires)
            nt = len(self.t_wires)

                # Encoding layer
            for i in range(nq):
                qml.RY(x1[i], wires=self.q_wires[i])
                qml.RX(x2[i], wires=self.q_wires[i])

                # CNOT pattern: for i<j, CX(q[i], q[j])
            for i in range(nq):
                for j in range(i + 1, nq):
                    qml.CNOT(wires=[self.q_wires[i], self.q_wires[j]])

                # Rotation layer
            for i in range(nq):
                qml.RZ(weights[i],           wires=self.q_wires[i])
                qml.RY(weights[i + nq],      wires=self.q_wires[i])
                qml.RZ(weights[i + 2 * nq],  wires=self.q_wires[i])
                qml.RX(weights[i + 3 * nq],  wires=self.q_wires[i])

                # Ancilla H
            qml.Hadamard(wires=self.a_wire)

                # Controlled-SWAPs
            for i in reversed(range(nt)):
                qml.CSWAP(wires=[self.a_wire, self.q_wires[-(i + 1)], self.t_wires[-(i + 1)]])

                # Final H
            qml.Hadamard(wires=self.a_wire)

                # "Measure ancilla"
            if return_projector:
                # returns samples of projector onto |1>, i.e. 1 if ancilla is |1>, else 0
                #return qml.sample(qml.Projector([1], wires=self.a_wire))
                return qml.expval(qml.Projector([1], wires=self.a_wire))
                #return qml.Projector([1], wires=self.a_wire)
            else:
                # returns samples of PauliZ: +1 corresponds to |0>, -1 to |1>
                return qml.sample(qml.PauliZ(wires=self.a_wire))

        self.circuit = circuit

    def autoencoder(self,circuit_weights, normalisation_weight, x1,x2):
        #theta = np.zeros_like(x1)
        #phi = np.zeros_like(x2)
        # normalisation weight feature specific, 0-4 jets, 5-8 muons, 9-12 electrons, 13 sums
        
        #f_jets = 1 + 2*np.pi / (1 + np.exp(-normalisation_weight[0]))    
        # f_muons = 1 + 2*np.pi / (1 + np.exp(-normalisation_weight[1]))    
        # f_electrons = 1 + 2*np.pi / (1 + np.exp(-normalisation_weight[2]))    
        #f_sums = 1 + 2*np.pi / (1 + np.exp(-normalisation_weight[3]))    
        
        # theta_jets = f_jets * x1[self.jet_indices[0]:self.jet_indices[1]]
        # phi_jets = f_jets * x2[self.jet_indices[0]:self.jet_indices[1]]
        
        # theta_muons = f_muons * x1[self.muon_indices[0]:self.muon_indices[1]]
        # phi_muons = f_muons * x2[self.muon_indices[0]:self.muon_indices[1]]
        
        # theta_electons = f_electrons * x1[self.electron_indices[0]:self.electron_indices[1]]
        # phi_electons = f_electrons * x2[self.electron_indices[0]:self.electron_indices[1]]
        
        #theta_sums = np.expand_dims(f_sums * x1[self.sum_indices],axis=0)
        #phi_sums = np.expand_dims(f_sums * x2[self.sum_indices],axis=0)
         
        #theta = np.concatenate((theta_jets,theta_muons,theta_electons))
        #phi = np.concatenate((phi_jets,phi_muons,phi_electons))
        
        f = 1 + 2*np.pi / (1 + np.exp(-normalisation_weight))    
        
        theta = f * x1
        phi = f * x2
        
        return self.circuit(circuit_weights, theta,phi,return_projector=True)

    def cost(self,circuit_weights, normalisation_weight, x1,x2 ):
        predictions = 0
        for ievent in range(len(x1)):
            single_value_predict = (self.autoencoder(circuit_weights, normalisation_weight, x1[ievent],x2[ievent]))
            predictions += qml.math.mean(single_value_predict)
        return predictions / len(x1)

    def compile_model(self, n_samples,**kwargs):
        #self.opt = NesterovMomentumOptimizer(0.5)
        #self.opt = qml.QNGOptimizer(0.01, approx="block-diag")
        self.opt = qml.AdamOptimizer(self.training_config['learning_rate'])
        self.batch_size = self.training_config['batch_size']
        

    def fit(
        self,
        X_train: DataSet,
        training_columns: list,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        self.history = {}
        
        pt_columns = []
        for column in training_columns:
            if ("PT" in column) :
                pt_columns.append(column)
        pt = X_train[pt_columns].to_numpy()
        
        print(pt.shape)
        pt_tot = np.sum(pt,axis=1) 

        pt_jet_tot = np.sum(pt[:,self.jet_indices[0]:self.jet_indices[1]],axis=1) 
        pt_muon_tot = np.sum(pt[:,self.muon_indices[0]:self.muon_indices[1]],axis=1) 
        pt_electron_tot = np.sum(pt[:,self.electron_indices[0]:self.electron_indices[1]],axis=1) 

        pt_tot = np.sum(pt,axis=1) 
        #pt_columns.append('L1T_PUPPIMET_MET')
        all_pt = X_train[pt_columns].to_numpy()
        
        eta_columns = []
        for column in training_columns:
            if (("Eta" in column) and ('PUPPIMET' not in column)):
                eta_columns.append(column)
        print(eta_columns)
        eta = X_train[eta_columns].to_numpy()

        phi_columns = []
        for column in training_columns:
            if (("Phi" in column) and ('PUPPIMET' not in column)):
                phi_columns.append(column)
        
        phi = X_train[phi_columns].to_numpy()

        
        uniform_sample_pt = uniform_sample(pt_tot,100,100)
        os.makedirs(os.path.join(self.output_directory, 'plots/training'), exist_ok=True)

        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        ax.hist(
                    eta.flatten(),
                    bins=100,
                    range=(-4.5,4.5),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    label = "$\\eta$",
                    density=True,
                    color = 'r'
                    )
        ax.hist(
                    phi.flatten(),
                    bins=100,
                    range=(-4.5,4.5),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    label = "$\\phi$",
                    density=True,
                    color = 'b'
                    )
        ax.grid(True)
        ax.set_xlabel('Input variable', ha="right", x=1)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.output_directory+'/plots/training/phiandeta.png')
        
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        ax.hist(
                    all_pt/pt_tot[:,None],
                    bins=100,
                    range=(0,1),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    label = "pt/pt_tot",
                    density=True,
                    )
        ax.grid(True)
        ax.set_xlabel('Input variable', ha="right", x=1)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.output_directory+'/plots/training/pttotpt.png')
        
        plot_2d((all_pt/pt_tot[:,None]).flatten(), phi.flatten(), (0,1), (-np.pi,np.pi), 'allpt / pt', 'phi', '')
        plt.savefig(self.output_directory+'/plots/training/ptvsphi.png')
        
        plot_2d((all_pt/pt_tot[:,None]).flatten(), eta.flatten(), (0,1), (-4.5,4.5), 'allpt / pt', 'eta', '')
        plt.savefig(self.output_directory+'/plots/training/ptvseta.png')
        
        plot_2d(phi.flatten(), eta.flatten(), (-np.pi,np.pi), (-4.5,4.5), 'phi', 'eta', '')
        plt.savefig(self.output_directory+'/plots/training/phivseta.png')
        
        old_x1 = all_pt/pt_tot[:,None] * eta
        old_x2 = all_pt/pt_tot[:,None] * phi
        
        old_x1 = (((old_x1 - np.min(old_x1)) / (np.max(old_x1) - np.min(old_x1))) * 2*np.pi) - np.pi
        old_x2 = (((old_x2 - np.min(old_x2)) / (np.max(old_x2) - np.min(old_x2))) * 2*np.pi) - np.pi
        
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        ax.hist(
                    old_x1.flatten(),
                    bins=100,
                    range=(-np.pi,np.pi),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    label = "$\\eta$",
                    density=True,
                    color = 'r'
                    )
        ax.hist(
                    old_x2.flatten(),
                    bins=100,
                    range=(-np.pi,np.pi),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    label = "$\\phi$",
                    density=True,
                    color = 'b'
                    )
        ax.grid(True)
        ax.set_xlabel('Input variable', ha="right", x=1)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.output_directory+'/plots/training/old_rescaled_inputs.png')
        
        x1_jets = np.nan_to_num((pt[:,self.jet_indices[0]:self.jet_indices[1]]/pt_jet_tot[:,None])  * eta[:,self.jet_indices[0]:self.jet_indices[1]])
        x2_jets = np.nan_to_num((pt[:,self.jet_indices[0]:self.jet_indices[1]]/pt_jet_tot[:,None])  * phi[:,self.jet_indices[0]:self.jet_indices[1]])
        
        #x1_jets = (((x1_jets - np.min(x1_jets)) / (np.max(x1_jets) - np.min(x1_jets))) * 2*np.pi) - np.pi
        #x2_jets = (((x2_jets - np.min(x2_jets)) / (np.max(x2_jets) - np.min(x2_jets))) * 2*np.pi) - np.pi
        
        x1_muon = np.nan_to_num((pt[:,self.muon_indices[0]:self.muon_indices[1]]/pt_muon_tot[:,None])  * eta[:,self.muon_indices[0]:self.muon_indices[1]])
        x2_muon = np.nan_to_num((pt[:,self.muon_indices[0]:self.muon_indices[1]]/pt_muon_tot[:,None])  * phi[:,self.muon_indices[0]:self.muon_indices[1]])
        
        #x1_muon = (((x1_muon - np.min(x1_muon)) / (np.max(x1_muon) - np.min(x1_muon))) * 2*np.pi) - np.pi
        #x2_muon = (((x2_muon - np.min(x2_muon)) / (np.max(x2_muon) - np.min(x2_muon))) * 2*np.pi) - np.pi
        
        x1_electron = np.nan_to_num((pt[:,self.electron_indices[0]:self.electron_indices[1]]/pt_electron_tot[:,None])  * eta[:,self.electron_indices[0]:self.electron_indices[1]])
        x2_electron = np.nan_to_num((pt[:,self.electron_indices[0]:self.electron_indices[1]]/pt_electron_tot[:,None])  * phi[:,self.electron_indices[0]:self.electron_indices[1]])
        
        #x1_electron = (((x1_electron - np.min(x1_electron)) / (np.max(x1_electron) - np.min(x1_electron))) * 2*np.pi) - np.pi
        #x2_electron = (((x2_electron - np.min(x2_electron)) / (np.max(x2_electron) - np.min(x2_electron))) * 2*np.pi) - np.pi
        
        #x1_sum =  np.nan_to_num(np.expand_dims((all_pt/pt_tot[:,None])[:,self.sum_indices]  *eta[:,self.sum_indices],axis=1))
        #x2_sum = np.nan_to_num(np.expand_dims((all_pt/pt_tot[:,None])[:,self.sum_indices]   *phi[:,self.sum_indices],axis=1))
        
        #x1_sum = (((x1_sum - np.min(x1_sum)) / (np.max(x1_sum) - np.min(x1_sum))) * 2*np.pi) - np.pi
        #x2_sum = (((x2_sum - np.min(x2_sum)) / (np.max(x2_sum) - np.min(x2_sum))) * 2*np.pi) - np.pi
         
        x1 = np.concatenate((x1_jets,x1_muon,x1_electron),axis=1)
        x2 = np.concatenate((x2_jets,x2_muon,x2_electron),axis=1) 
        
        for one_dim in range(x1.shape[1]):
            x1[:,one_dim] = (((x1[:,one_dim] - np.min(x1[:,one_dim])) / (np.max(x1[:,one_dim]) - np.min(x1[:,one_dim]))) * 2*np.pi) - np.pi
            x2[:,one_dim] = (((x2[:,one_dim] - np.min(x2[:,one_dim])) / (np.max(x2[:,one_dim]) - np.min(x2[:,one_dim]))) * 2*np.pi) - np.pi
                    
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        ax.hist(
                    x1.flatten(),
                    bins=100,
                    range=(-np.pi,np.pi),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    label = "$\\eta$",
                    density=True,
                    color = 'r'
                    )
        ax.hist(
                    x2.flatten(),
                    bins=100,
                    range=(-np.pi,np.pi),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    label = "$\\phi$",
                    density=True,
                    color = 'b'
                    )
        ax.grid(True)
        ax.set_xlabel('Input variable', ha="right", x=1)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.output_directory+'/plots/training/rescaled_inputs.png')
        
        plot_2d(x1.flatten(), x2.flatten(), (-np.pi,np.pi), (-np.pi,np.pi), 'x1', 'x2', '')
        plt.savefig(self.output_directory+'/plots/training/x1vsx2.png')
        
        for i in range(x1.shape[1]):
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
            ax.hist(
                        x1[:,i].flatten(),
                        bins=100,
                        range=(-np.pi,np.pi),
                        histtype="step",
                        stacked=False,
                        linewidth=style.LINEWIDTH - 1.5,
                        label = "x1_" + str(i),
                        density=True,
                        color = 'r'
                        )
            ax.hist(
                        x2[:,i].flatten(),
                        bins=100,
                        range=(-np.pi,np.pi),
                        histtype="step",
                        stacked=False,
                        linewidth=style.LINEWIDTH - 1.5,
                        label = "x2_" + str(i),
                        density=True,
                        color = 'b'
                        )
            ax.grid(True)
            ax.set_xlabel('Input variable', ha="right", x=1)
            ax.set_yscale('log')
            ax.legend()
            plt.savefig(self.output_directory+'/plots/training/x1x2_'+str(i)+'.png')
        
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        ax.hist(
                    pt_tot[uniform_sample_pt],
                    bins=100,
                    range=(0,1000),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    label = "sampled pT",
                    density=True,
                    color = 'r'
                    )
        ax.hist(
                    pt_tot,
                    bins=100,
                    range=(0,1000),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    label = "orginal pT",
                    density=True,
                    color = 'b'
                    )
        ax.grid(True)
        ax.set_xlabel('Total Event pT', ha="right", x=1)
        ax.legend()
        plt.savefig(self.output_directory+'/plots/training/pt_tot.png')

        circuit_weights_init = 0.01 * np.random.randn(self.nq*4, requires_grad=True)
        normalisation_weight_init = np.zeros(1, requires_grad=True)
                
        self.circuit_weights = circuit_weights_init
        self.normalisation_weight = normalisation_weight_init
        
        print("circuit_weights:", circuit_weights_init)
        print("normalisation_weight: ", normalisation_weight_init)
        cost = []
        val_cost = []
        val_batch_index = np.random.choice(uniform_sample_pt, self.training_config['batch_size'])
        
        for it in range(self.training_config['epochs']):

            # Update the weights by one optimizer step, using only a limited batch of data
            batch_index = np.random.choice(uniform_sample_pt, (self.batch_size,))
            x1_batch = old_x1[batch_index]
            x2_batch = old_x2[batch_index]
            
            self.circuit_weights, self.normalisation_weight = self.opt.step(self.cost, self.circuit_weights, self.normalisation_weight, x1=x1_batch, x2=x2_batch)

            current_cost = self.cost(self.circuit_weights, self.normalisation_weight, x1_batch,x2_batch )
            cost.append(current_cost)
            
            x1_val_batch = old_x1[val_batch_index]
            x2_val_batch = old_x2[val_batch_index]
            
            # Compute accuracy            
            validation_cost = self.cost(self.circuit_weights, self.normalisation_weight, x1_val_batch,x2_val_batch )
            val_cost.append(validation_cost)
            print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Validation Cost: {validation_cost:0.7f}")
            
            
            # plt.clf()
            # fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
            # ax.hist(
            #         self.circuit_weights,
            #         bins=10,
            #         range=(-0.5,0.5),
            #         histtype="step",
            #         stacked=False,
            #         linewidth=style.LINEWIDTH - 1.5,
            #         density=True,
            #         )
            # ax.grid(True)
            # ax.set_ylim(0,6)
            # ax.set_xlabel('circuit parameters', ha="right", x=1)
            # plt.savefig(self.output_directory+'/plots/training/weights_'+str(it)+'.png')
           
        print("circuit_weights at at iteration", it," : ", self.circuit_weights)
        print("normalisation_weight at iteration", it," : ", self.normalisation_weight) 
        self.history['loss'] = cost
        self.history['val_loss'] = val_cost
        
        # print(qml.draw(self.circuit)(self.circuit_weights,x1_batch[0],x2_batch[0]))
        
        # self.dev._circuit.draw(output="mpl")
        # plt.savefig(self.output_directory+'/plots/training/circuit.png')

    def predict(self, X_test, training_columns, return_score = True) -> npt.NDArray[np.float64]:
        
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test.to_numpy()
        else:
            test = X_test
        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """
        
        self.build_model(len(training_columns))
        pt_columns = []
        for column in training_columns:
            if ("PT" in column) :
                pt_columns.append(column)
        
        pt = X_test[pt_columns].to_numpy()
        pt_tot = np.sum(pt,axis=1) 
        #pt_columns.append('L1T_PUPPIMET_MET')
        all_pt = X_test[pt_columns].to_numpy()
        
        eta_columns = []
        for column in training_columns:
            if (("Eta" in column) and ('PUPPIMET' not in column)):
                eta_columns.append(column)
        
        eta = X_test[eta_columns].to_numpy()
        
        phi_columns = []
        for column in training_columns:
            if (("Phi" in column) and ('PUPPIMET' not in column)):
                phi_columns.append(column)
        
        phi = X_test[phi_columns].to_numpy()
        
        x1 = (all_pt/pt_tot[:,None]) * eta
        x2 = (all_pt/pt_tot[:,None]) * phi
        
        x1 = (((x1 - np.min(x1)) / (np.max(x1) - np.min(x1))) * 2*np.pi) - np.pi
        x2 = (((x2 - np.min(x2)) / (np.max(x2) - np.min(x2))) * 2*np.pi) - np.pi
        print(x1.shape)
        print(self.circuit_weights.shape)
        predictions = []
        
        for ievent in range(len(x1)):
            predictions.append(qml.math.mean(self.autoencoder(self.circuit_weights, self.normalisation_weight, x1[ievent], x2[ievent] )))
        ad_scores = np.array(predictions ) 
        ad_scores = ad_scores / 0.5
        return ad_scores
    
    
    def plot_loss(self):
        """Plot the loss of the model to the output directory"""
        out_dir = self.output_directory
        # Produce some basic plots with the training for diagnostics
        plot_path = os.path.join(out_dir, "plots/training")
        # Plot history
        loss_history(plot_path, ['loss'], self.history)
    # Decorated with save decorator for added functionality
    @ADModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        np.save(out_dir+"/model/circuit_weights.npy", self.circuit_weights)
        np.save(out_dir+"/model/normalisation_weight.npy", self.normalisation_weight)

    @ADModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.circuit_weights=np.load(out_dir+"/model/circuit_weights.npy")
        self.normalisation_weight=np.load(out_dir+"/model/normalisation_weight.npy")