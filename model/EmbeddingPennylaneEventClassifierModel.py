import os

import numpy.typing as npt

from model.EventClassifierModel import ECModelFactory, ECModel
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

from model.common import ECfromFolder, fromFolder


import matplotlib.pyplot as plt
from plot import style

import numpy as np
import pennylane.numpy as pnp

# Register the model in the factory with the string name corresponding to what is in the yaml config
@ECModelFactory.register('EmbeddingCECModel')
class EmbeddingCECModel(ECModel):

    """EmbeddingCECModel class

    Args:
        EmbeddingCECModel (_type_): Base class of a Event Classifier Model
    """
    
    def __init__(self,output_dir):
        super().__init__(output_dir)
        
    def load_embedding_model(self,model_folder):
        self.embedding_model = fromFolder(model_folder)

    def build_model(self, inputs_shape: tuple,output_shape: int, xmin,xmax):
        """build model override

        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        total_object = int(self.model_config['embedding_dim']) 
        
        self.xmin = xmin
        self.xmax = xmax
        
        ### Circuit params
        self.nq = 8
        self.n_layers = self.model_config['n_layers']
        
        
        def reupload_block(x, params,n_qubits):
            """
                One reuploading layer: weighted data encoding + entangling cascade.

                params has shape (n_qubits, 4): (w_y, b_y, w_z, b_z) per qubit.
                iff optional variational layer is used, params has shape (n_qubits, 7): (w_y, b_y, w_z, b_z, w_rx, w_ry, w_rz) per qubit.
            """
            for i in range(n_qubits):
                qml.RY(params[i, 0] * x[i] + params[i, 1], wires=i)
                qml.RZ(params[i, 2] * x[i] + params[i, 3], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
                
            ### optional variational layer after entanglement
            for i in range(n_qubits):
                qml.RX(params[i, 4], wires=i)
                qml.RY(params[i, 5], wires=i)
                qml.RZ(params[i, 6], wires=i)
                
        self.dev = qml.device("lightning.gpu", wires=self.n_qubuits)
        
        
        @qml.set_shots(shots=None)
        @qml.qnode(dev, interface="autograd")
        def circuit(x, weights):
            # weights shape: (n_layers, n_qubits, 4)
            for layer in range(self.n_layers):
                reupload_block(x, weights[layer],self.nq)
            return qml.expval(qml.PauliZ(0))      # not sure what best to measure
            # return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            # return (
            #     [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            #     +
            #     [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i+1) % n_qubits))
            #     for i in range(n_qubits)]
            # )
        
        self.circuit = circuit
        
    def cost(self, weights, x, y):
        predictions = pnp.stack([self.circuit(weights, x[i]) for i in range(len(y))])
        predictions = pnp.clip(predictions, 1e-7, 1-1e-7)
        bce = y * pnp.log(predictions) + (1 - y) * pnp.log(1 - predictions)
        return -pnp.mean(bce)

    def compile_model(self, n_samples,**kwargs):
        self.opt = qml.AdamOptimizer(self.training_config['learning_rate'])
        self.batch_size = self.training_config['batch_size']

        

    def fit(
        self,
        X_train: DataSet,
        y_train,
        training_columns: list,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        self.history = {}
        
        circuit_weights_init = 0.01 * np.random.randn((self.n_layers,self.nq,7), requires_grad=True)
                
        self.circuit_weights = circuit_weights_init
        
        print("circuit_weights:", circuit_weights_init)
        cost = []
        val_cost = []
                
        val_batch_index = np.random.randint(0, len(X_train) // 2, size=self.training_config['batch_size'])
        val_embeddings = self.embedding_model.encoder_predict(X_train[training_columns].to_numpy()[val_batch_index],training_columns)         
        
        
        val_embeddings = np.clip(val_embeddings, self.xmin, self.xmax)
        x_val = np.array((((val_embeddings - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
        y_val = y_train[val_batch_index]
        
        batch_index = np.random.randint(len(X_train)//2, len(X_train), size=self.training_config['batch_size'])
        train_embeddings = self.embedding_model.encoder_predict(X_train[training_columns].to_numpy()[batch_index],training_columns) 
        train_embeddings = np.clip(train_embeddings, self.xmin, self.xmax)
        x_train = np.array((((train_embeddings - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
        y_train = y_train[batch_index]       
                  
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
        
    def only_QEC_fit(
        self,
        X_train: DataSet,
        y_train,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        self.history = {}
        
        circuit_weights_init = 0.01 * np.random.randn((self.n_layers,self.nq,7), requires_grad=True)
                
        self.circuit_weights = circuit_weights_init
        
        print("circuit_weights:", circuit_weights_init)
        cost = []
        val_cost = []
                
        val_batch_index = np.random.randint(0, len(X_train) // 2, size=self.training_config['batch_size'])
        val_embeddings = X_train[val_batch_index]         
        
        
        val_embeddings = np.clip(val_embeddings, self.xmin, self.xmax)
        x_val = np.array((((val_embeddings - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
        y_val = y_train[val_batch_index]
        
        batch_index = np.random.randint(len(X_train)//2, len(X_train), size=self.training_config['batch_size'])
        train_embeddings = X_train[batch_index] 
        train_embeddings = np.clip(train_embeddings, self.xmin, self.xmax)
        x_train = np.array((((train_embeddings - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
        y_train = y_train[batch_index]       
                  
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
        

    def predict(self, X_test, training_columns,n_out) -> npt.NDArray[np.float64]:
        
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test[training_columns].to_numpy()
        else:
            test = X_test
        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """
        
        self.build_model(len(training_columns),n_out,self.xmin,self.xmax)
        
        embeddings = self.embedding_model.encoder_predict(test,training_columns) 
        embeddings = np.clip(embeddings, self.xmin, self.xmax)
        x = (((embeddings - self.xmin) / (self.xmax- self.xmin)) * 2*np.pi) - np.pi
        
        predictions = []
        
        for ievent in range(len(x)):
            predictions.append(qml.math.mean(self.circuit(self.circuit_weights,  x[ievent] )))
        scores = np.array(predictions ) 
        return scores        
        
    def only_QEC_predict(self, X_test, n_out) -> npt.NDArray[np.float64]:

        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """
        
        self.build_model(X_test.shape[1],n_out,self.xmin,self.xmax)

        X_test = np.clip(X_test, self.xmin, self.xmax)
        x = (((X_test - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi
        predictions = []
        
        for ievent in range(len(x)):
            predictions.append(qml.math.mean(self.circuit(self.circuit_weights,  x[ievent] )))
        scores = np.array(predictions ) 
        return scores 
    
    
    
    @ECModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        p.save(out_dir+"/model/circuit_weights.npy", self.circuit_weights)
        
        with open(out_dir+"/model/min.txt", "a") as f:
            f.write(str(self.xmin))
        with open(out_dir+"/model/max.txt", "a") as f:
            f.write(str(self.xmax))
            
        print(f"Model saved to {export_path}")

    @ECModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.circuit_weights=np.load(out_dir+"/model/circuit_weights.npy")
        
        with open(f"{out_dir}/model/min.txt") as f:
            self.xmin = float(f.read())
        with open(f"{out_dir}/model/max.txt") as f:
            self.xmax = float(f.read())

