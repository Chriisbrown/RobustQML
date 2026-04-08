import os

import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel
from data.EOSdataset import DataSet
import pandas as pd

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import keras
from keras.models import load_model
from keras.layers import Dense

from sklearn.model_selection import train_test_split

import tensorflow as tf

import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane import numpy as np

from plot.basic import loss_history, plot_2d

from model.common import fromFolder


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
@ADModelFactory.register('EmbeddingPennyLaneQAEModel')
class EmbeddingPennyLaneQAEModel(ADModel):

    """EmbeddingPennyLaneQAEModel class

    Args:
        EmbeddingPennyLaneQAEModel (_type_): Base class of a PennyLaneQAEModel
    """
    
    def __init__(self,output_dir):
        super().__init__(output_dir)
        
    def load_embedding_model(self,model_folder):
        self.embedding_model = fromFolder(model_folder)

    def build_model(self, inputs_shape: tuple):
        """build model override

        Args:
            inputs_shape (tuple): Shape of the input
        """
        total_object = int(self.model_config['embedding_dim']) 
        
        nq = total_object
        nt = int(nq - self.model_config['latent_dim'])        
        self.nq = nq
        
        self.q_wires = [f"q{i}" for i in range(nq)]
        self.t_wires = [f"t{i}" for i in range(nt)]
        self.a_wire = "a"
                
        self.dev = qml.device("lightning.gpu", wires=self.q_wires + self.t_wires + [self.a_wire])
        @qml.set_shots(shots=None)
        @qml.qnode(self.dev)
        def circuit(weights , x1, return_projector=False):
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

    def autoencoder(self,circuit_weights, x):        
        return self.circuit(circuit_weights, x,return_projector=True)

    def cost(self,circuit_weights, x):
        predictions = 0
        for ievent in range(len(x)):
            single_value_predict = (self.autoencoder(circuit_weights, x[ievent]))
            predictions += qml.math.mean(single_value_predict)
        return predictions / len(x)

    def compile_model(self, n_samples,**kwargs):
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
        
        os.makedirs(os.path.join(self.output_directory, 'plots/training'), exist_ok=True)
        
        circuit_weights_init = 0.01 * np.random.randn(self.nq*4, requires_grad=True)
                
        self.circuit_weights = circuit_weights_init
        
        print("circuit_weights:", circuit_weights_init)
        cost = []
        val_cost = []
                
        val_batch_index = np.random.randint(0, len(X_train) // 2, size=self.training_config['batch_size'])
        print(val_batch_index)
        val_embeddings = self.embedding_model.encoder_predict(X_train[training_columns].to_numpy()[val_batch_index],training_columns)         
        
        x_val = np.zeros_like(val_embeddings)
        
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        for i_embedding in range(val_embeddings.shape[1]):
            x_val[:,i_embedding] = (((val_embeddings[:,i_embedding] - np.min(val_embeddings[:,i_embedding])) / (np.max(val_embeddings[:,i_embedding]) - np.min(val_embeddings[:,i_embedding]))) * 2*np.pi) - np.pi
            ax.hist(
                        x_val[:,i_embedding],
                        bins=100,
                        range=(-np.pi,np.pi),
                        histtype="step",
                        stacked=False,
                        linewidth=style.LINEWIDTH - 1.5,
                        label = "val dim: "+str(i_embedding),
                        density=True,
                        )
        ax.grid(True)
        ax.set_xlabel('Input variable', ha="right", x=1)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.output_directory+'/plots/training/rescaled_inputs.png')
        
        print(qml.specs(self.circuit)(circuit_weights_init, x_val[0],return_projector=True ))
        
        batch_index = np.random.randint(len(X_train)//2, len(X_train), size=self.training_config['batch_size'])
        train_embeddings = self.embedding_model.encoder_predict(X_train[training_columns].to_numpy()[batch_index],training_columns) 
        x_train = np.zeros_like(train_embeddings)
        for i_embedding in range(train_embeddings.shape[1]):
            x_train[:,i_embedding] = (((train_embeddings[:,i_embedding] - np.min(train_embeddings[:,i_embedding])) / (np.max(train_embeddings[:,i_embedding]) - np.min(train_embeddings[:,i_embedding]))) * 2*np.pi) - np.pi
                  
                  
        for it in range(self.training_config['epochs']):
            # Update the weights by one optimizer step, using only a limited batch of data
                  
            self.circuit_weights= self.opt.step(self.cost, self.circuit_weights, x=x_train)

            current_cost = self.cost(self.circuit_weights, x_train)
            cost.append(current_cost)
                        
            # Compute accuracy            
            validation_cost = self.cost(self.circuit_weights, x_val )
            val_cost.append(validation_cost)
            print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Validation Cost: {validation_cost:0.7f}")

        print("circuit_weights at at iteration", it," : ", self.circuit_weights)
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

        embeddings = self.embedding_model.encoder_predict(X_test[training_columns],training_columns) 
        x = np.zeros_like(embeddings)
        for i_embedding in range(embeddings.shape[1]):
            x[:,i_embedding] = (((embeddings[:,i_embedding] - np.min(embeddings[:,i_embedding])) / (np.max(embeddings[:,i_embedding]) - np.min(embeddings[:,i_embedding]))) * 2*np.pi) - np.pi

        predictions = []
        
        for ievent in range(len(x)):
            predictions.append(qml.math.mean(self.autoencoder(self.circuit_weights,  x[ievent] )))
        ad_scores = np.array(predictions ) 
        ad_scores = 1 - ad_scores/0.5
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

    @ADModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.circuit_weights=np.load(out_dir+"/model/circuit_weights.npy")
