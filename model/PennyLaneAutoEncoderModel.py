import os

import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel
from data.dataset import DataSet
import pandas as pd

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import keras
from keras.models import load_model
from keras.layers import Dense

import tensorflow as tf

import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane import numpy as np

from plot.basic import loss_history

import matplotlib.pyplot as plt
from plot import style

# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('PennyLaneQAEModel')
class PennyLaneQAEModel(ADModel):

    """PennyLaneQAEModel class

    Args:
        PennyLaneQAEModel (_type_): Base class of a PennyLaneQAEModel
    """

    def build_model(self, inputs_shape: tuple):
        """build model override

        Args:
            inputs_shape (tuple): Shape of the input
        """
        total_object = int(inputs_shape / 3)
        
        nq = total_object
        nt = int(nq/2)        
        self.nq = nq
        
        self.q_wires = [f"q{i}" for i in range(nq)]
        self.t_wires = [f"t{i}" for i in range(nt)]
        self.a_wire = "a"
                
        self.dev = qml.device("lightning.qubit", wires=self.q_wires + self.t_wires + [self.a_wire])
        @qml.set_shots(shots=5000)
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
                return qml.expval(qml.Projector([1], wires=self.a_wire))
            else:
                # returns samples of PauliZ: +1 corresponds to |0>, -1 to |1>
                return qml.expval(qml.PauliZ(wires=self.a_wire))

        self.circuit = circuit

    def autoencoder(self,circuit_weights, normalisation_weight, x1,x2):
        f = 1 + 2*np.pi / (1 + np.exp(-normalisation_weight))    
        theta = f * x1
        phi = f * x2
        return self.circuit(circuit_weights, theta,phi,return_projector=True)

    def cost(self,circuit_weights, normalisation_weight, x1,x2 ):
        predictions = 0
        for ievent in range(len(x1)):
            single_value_predict = -(self.autoencoder(circuit_weights, normalisation_weight, x1[ievent],x2[ievent]))
            #print(qml.math.stack(single_value_predict))
            predictions += single_value_predict
        return predictions / len(x1)

    def compile_model(self, **kwargs):
        #self.opt = NesterovMomentumOptimizer(0.5)
        #self.opt = qml.QNGOptimizer(0.01, approx="block-diag")
        self.opt = qml.AdamOptimizer(0.1)
        self.batch_size = 5
        

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
        pt_tot = np.sum(pt,axis=1) 
        pt_columns.append('L1T_PUPPIMET_MET')
        all_pt = X_train[pt_columns].to_numpy()
        
        eta_columns = []
        for column in training_columns:
            if (("Eta" in column)):
                eta_columns.append(column)
        
        eta = X_train[eta_columns].to_numpy()
        
        phi_columns = []
        for column in training_columns:
            if (("Phi" in column)):
                phi_columns.append(column)
        
        phi = X_train[phi_columns].to_numpy()
        
        x1 = (all_pt/pt_tot[:,None]) * eta
        x2 = (all_pt/pt_tot[:,None]) * phi
        
        x1 = (((x1 - np.min(x1)) / (np.max(x1) - np.min(x1))) * 2*np.pi) - np.pi
        x2 = (((x2 - np.min(x2)) / (np.max(x2) - np.min(x2))) * 2*np.pi) - np.pi
        
        os.makedirs(os.path.join(self.output_directory, 'plots/training'), exist_ok=True)
        
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

        circuit_weights_init = 0.01 * np.random.randn(self.nq*4, requires_grad=True)
        normalisation_weight_init = np.array(0.0, requires_grad=True)
                
        self.circuit_weights = circuit_weights_init
        self.normalisation_weight = normalisation_weight_init
        
        print("circuit_weights:", circuit_weights_init)
        print("normalisation_weight: ", normalisation_weight_init)
        cost = []
        val_cost = []
        val_batch_index = np.random.randint(0, len(x1), (50))
        
        for it in range(20):

            # Update the weights by one optimizer step, using only a limited batch of data
            batch_index = np.random.randint(0, len(x1), (self.batch_size,))
            x1_batch = x1[batch_index]
            x2_batch = x2[batch_index]
            
            self.circuit_weights, self.normalisation_weight = self.opt.step(self.cost, self.circuit_weights, self.normalisation_weight, x1=x1_batch, x2=x2_batch)

            current_cost = self.cost(self.circuit_weights, self.normalisation_weight, x1_batch,x2_batch )
            cost.append(current_cost)
            
            x1_val_batch = x1[val_batch_index]
            x2_val_batch = x2[val_batch_index]
            
            # Compute accuracy            
            validation_cost = self.cost(self.circuit_weights, self.normalisation_weight, x1_val_batch,x2_val_batch )
            val_cost.append(validation_cost)
            print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Validation Cost: {validation_cost:0.7f}")
            
            
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
            ax.hist(
                    self.circuit_weights,
                    bins=10,
                    range=(-0.5,0.5),
                    histtype="step",
                    stacked=False,
                    linewidth=style.LINEWIDTH - 1.5,
                    density=True,
                    )
            ax.grid(True)
            ax.set_ylim(0,6)
            ax.set_xlabel('circuit parameters', ha="right", x=1)
            plt.savefig(self.output_directory+'/plots/training/weights_'+str(it)+'.png')
           
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
        pt_columns.append('L1T_PUPPIMET_MET')
        all_pt = X_test[pt_columns].to_numpy()
        
        eta_columns = []
        for column in training_columns:
            if (("Eta" in column)):
                eta_columns.append(column)
        
        eta = X_test[eta_columns].to_numpy()
        
        phi_columns = []
        for column in training_columns:
            if (("Phi" in column)):
                phi_columns.append(column)
        
        phi = X_test[phi_columns].to_numpy()
        
        x1 = (all_pt/pt_tot[:,None]) * eta
        x2 = (all_pt/pt_tot[:,None]) * phi
        
        predictions = []
        
        for ievent in range(len(x1)):
            predictions.append(self.autoencoder(self.circuit_weights, self.normalisation_weight, x1[ievent], x2[ievent] ))
        return np.array(predictions)
    
    
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