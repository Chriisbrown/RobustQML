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
@ECModelFactory.register('EmbeddingQECModel')
class EmbeddingQECModel(ECModel):

    """EmbeddingCQECModel class

    Args:
        EmbeddingQECModel (_type_): Base class of a Event Classifier Model
    """
    
    def __init__(self,output_dir):
        super().__init__(output_dir)
        
    def load_embedding_model(self,model_folder):
        self.embedding_model = fromFolder(model_folder)
        
    def cosine_annealing(self,initial_lr, min_lr, total_steps, step):
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * step / total_steps))


    def build_model(self, inputs_shape: tuple,output_shape: int, xmin,xmax):
        """build model override

        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        total_object = int(self.model_config['embedding_dim']) 
        
        self.xmin = xmin
        self.xmax = xmax
        
        ### Circuit params
        self.nq = self.model_config['embedding_dim']
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
                
        self.dev = qml.device("lightning.gpu", wires=self.nq)
        
        
        @qml.set_shots(shots=None)
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights,x):
            # weights shape: (n_layers, n_qubits, 4)
            for layer in range(self.n_layers):
                reupload_block(x, weights[layer],self.nq)
            return qml.expval(qml.PauliZ(0))      # not sure what best to measure
            #return pnp.sum([qml.expval(qml.PauliZ(i)) for i in range(self.nq)])/self.nq
            #return qml.expval(sum(qml.PauliZ(i) for i in range(self.nq)) / self.nq)
            #return (
            #    pnp.sum( [qml.expval(qml.PauliZ(i)) for i in range(self.nq)]
            #     +
            #     [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i+1) % self.nq))
            #     for i in range(self.nq)]) / self.nq
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
        
        circuit_weights_init = 0.01 * pnp.random.randn(self.n_layers,self.nq,7, requires_grad=True)
                
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
            
            self.opt.stepsize = self.cosine_annealing(self.training_config['learning_rate'], 1e-5, self.training_config['epochs'], it)
            
            # Update the weights by one optimizer step, using only a limited batch of data
                  
            self.circuit_weights,current_cost = self.opt.step_and_cost(self.cost, self.circuit_weights, x=x_train, y=y_train)

            cost.append(current_cost)
                        
            # Compute accuracy            
            validation_cost = self.cost(self.circuit_weights, x_val, y_val )
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
        
        circuit_weights_init = 0.01 * pnp.random.randn(self.n_layers,self.nq,7, requires_grad=True)
                
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
            self.opt.stepsize = self.cosine_annealing(self.training_config['learning_rate'], 1e-5, self.training_config['epochs'], it)
                  
            self.circuit_weights, current_cost= self.opt.step_and_cost(self.cost, self.circuit_weights, x=x_train, y=y_train)
            cost.append(current_cost)
                        
            # Compute accuracy            
            validation_cost = self.cost(self.circuit_weights, x_val, y_val )
            val_cost.append(validation_cost)
            print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Validation Cost: {validation_cost:0.7f}" )

        print("circuit_weights at at iteration", it," : ", self.circuit_weights)
        self.history['loss'] = cost
        self.history['val_loss'] = val_cost
        

    def predict(self, X_test,training_columns,n_out) -> npt.NDArray[np.float64]:
        
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
        return 1 -scores        
        
    def only_QEC_predict(self, X_test,n_out) -> npt.NDArray[np.float64]:

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
        return 1 - scores 
    
    
    
    @ECModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        np.save(out_dir+"/model/circuit_weights.npy", self.circuit_weights)
        
        with open(out_dir+"/model/min.txt", "a") as f:
            f.write(str(self.xmin))
        with open(out_dir+"/model/max.txt", "a") as f:
            f.write(str(self.xmax))
            

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


@ECModelFactory.register('EmbeddingQECPLModel')
class EmbeddingQECPLModel(EmbeddingQECModel):
    def __init__(self,output_dir):
        super().__init__(output_dir)
        label_0 = [[1], [0]]
        label_1 = [[0], [1]]
        self.state_labels = pnp.array([label_0, label_1], requires_grad=False)
    
    # Define output labels as quantum state vectors
    def density_matrix(self,state):
        """Calculates the density matrix representation of a state.

        Args:
            state (array[complex]): array representing a quantum state vector

        Returns:
            dm: (array[complex]): array representing the density matrix
        """
        return state * pnp.conj(state).T

    def build_model(self, inputs_shape: tuple,output_shape: int, xmin,xmax):
        """build model override

        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        total_object = int(self.model_config['embedding_dim']) 
        
        self.xmin = xmin
        self.xmax = xmax
        
        ### Circuit params
        self.nq = self.model_config['embedding_dim']
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
                
        self.dev = qml.device("lightning.gpu", wires=self.nq)
        
        
        @qml.set_shots(shots=None)
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights,x,y):
            # weights shape: (n_layers, n_qubits, 4)
            for layer in range(self.n_layers):
                reupload_block(x, weights[layer],self.nq)
            return qml.expval(qml.Hermitian(y, wires=[0]))
        
        self.circuit = circuit
    
    def cost(self,weights, x, y, state_labels=None):
        """Cost function to be minimized.

        Args:
            params (array[float]): array of parameters
            x (array[float]): 2-d array of input vectors
            y (array[float]): 1-d array of targets
            state_labels (array[float]): array of state representations for labels

        Returns:
            float: loss value to be minimized
        """
        # Compute prediction for each input in data batch
        loss = 0.0
        dm_labels = [self.density_matrix(s) for s in state_labels]
        for i in range(len(x)):
            f = self.circuit(weights, x[i], dm_labels[y[i]])
            loss = loss + (1 - f) ** 2
        return loss / len(x)

    def compile_model(self, n_samples,**kwargs):
        self.opt = qml.AdamOptimizer(self.training_config['learning_rate'])
        self.batch_size = self.training_config['batch_size']
        
        
    def test(self,weights, x, y, state_labels=None):
        """
        Tests on a given set of data.

        Args:
            params (array[float]): array of parameters
            x (array[float]): 2-d array of input vectors
            y (array[float]): 1-d array of targets
            state_labels (array[float]): 1-d array of state representations for labels

        Returns:
            predicted (array([int]): predicted labels for test data
            output_states (array[float]): output quantum states from the circuit
        """
        fidelity_values = []
        dm_labels = [self.density_matrix(s) for s in state_labels]
        predicted = []

        for i in range(len(x)):
            fidel_function = lambda y: self.circuit(weights, x[i], y)
            fidelities = [fidel_function(dm) for dm in dm_labels]
            best_fidel = np.argmax(fidelities)

            predicted.append(best_fidel)
            fidelity_values.append(fidelities)

        return np.array(predicted), np.array(fidelity_values)
    
    def accuracy_score(self,y_true, y_pred):
        """Accuracy score.

        Args:
            y_true (array[float]): 1-d array of targets
            y_predicted (array[float]): 1-d array of predictions
            state_labels (array[float]): 1-d array of state representations for labels

        Returns:
            score (float): the fraction of correctly classified samples
        """
        score = y_true == y_pred
        return score.sum() / len(y_true)
    
    def iterate_minibatches(self,inputs, targets, batch_size):
        """
        A generator for batches of the input data

        Args:
            inputs (array[float]): input data
            targets (array[float]): targets

        Returns:
            inputs (array[float]): one batch of input data of length `batch_size`
            targets (array[float]): one batch of targets of length `batch_size`
        """
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]

            

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
        
        self.history = {'loss':[],'val_loss':[]}
        
        circuit_weights_init = 0.01 * pnp.random.randn(self.n_layers,self.nq,7, requires_grad=True)
                
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
        
        
        predicted_train, fidel_train = self.test(self.circuit_weights, X_train, y_train, self.state_labels)
        accuracy_train = self.accuracy_score(y_train, predicted_train)

        predicted_test, fidel_test = self.test(self.circuit_weights, X_val, y_val, self.state_labels)
        accuracy_test = self.accuracy_score(y_val, predicted_test)

        # save predictions with random weights for comparison
        initial_predictions = predicted_test

        loss = self.cost(params, X_test, y_test, self.state_labels)  
        
        print(
                "Epoch: {:2d} | Cost: {:3f} | Train accuracy: {:3f} | Test Accuracy: {:3f}".format(
                    0, loss, accuracy_train, accuracy_test
                )
            )
  
                  
        for it in range(self.training_config['epochs']):
            # Update the weights by one optimizer step, using only a limited batch of data
            for Xbatch, ybatch in self.iterate_minibatches(X_train, y_train, batch_size=100):
                self.circuit_weights, _, _, _ = self.opt.step(self.cost, self.circuit_weights, Xbatch, ybatch, state_labels)

            predicted_train, fidel_train = self.test(self.circuit_weights, X_train, y_train, state_labels)
            accuracy_train = self.accuracy_score(y_train, predicted_train)
            loss = self.cost(self.circuit_weights, X_train, y_train, state_labels)

            predicted_test, fidel_test = self.test(self.circuit_weights, X_test, y_test, state_labels)
            accuracy_test = self.accuracy_score(y_test, predicted_test)
            loss_test = self.cost(self.circuit_weights, X_test, y_test, state_labels)
            res = [it + 1, loss, accuracy_train, accuracy_test]
            print(
                "Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Test accuracy: {:3f}".format(
                    *res
                )
            )
            
            self.history['loss'].append(loss)
            self.history['val_loss'].append(loss_test)

        print("circuit_weights at at iteration", it," : ", self.circuit_weights)
        
        
    def only_QEC_fit(
        self,
        X_train: DataSet,
        y_train,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        self.history = {'loss':[],'val_loss':[]}
        
        circuit_weights_init = 0.01 * pnp.random.randn(self.n_layers,self.nq,7, requires_grad=True)
                
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
                  
        predicted_train, fidel_train = self.test(self.circuit_weights, x_train, y_train, self.state_labels)
        accuracy_train = self.accuracy_score(y_train, predicted_train)

        predicted_test, fidel_test = self.test(self.circuit_weights, x_val, y_val, self.state_labels)
        accuracy_test = self.accuracy_score(y_val, predicted_test)

        # save predictions with random weights for comparison
        initial_predictions = predicted_test

        loss = self.cost(self.circuit_weights, x_val, y_val, self.state_labels)  
        
        print(
                "Epoch: {:2d} | Cost: {:3f} | Train accuracy: {:3f} | Test Accuracy: {:3f}".format(
                    0, loss, accuracy_train, accuracy_test
                )
            )
  
                  
        for it in range(self.training_config['epochs']):
            # Update the weights by one optimizer step, using only a limited batch of data
            for Xbatch, ybatch in self.iterate_minibatches(x_train, y_train, batch_size=100):
                self.circuit_weights, _, _, _ = self.opt.step(self.cost, self.circuit_weights, Xbatch, ybatch, self.state_labels)

            predicted_train, fidel_train = self.test(self.circuit_weights, x_train, y_train, self.state_labels)
            accuracy_train = self.accuracy_score(y_train, predicted_train)
            loss = self.cost(self.circuit_weights, x_train, y_train, self.state_labels)

            predicted_test, fidel_test = self.test(self.circuit_weights, x_val, y_val, self.state_labels)
            accuracy_test = self.accuracy_score(y_val, predicted_test)
            loss_test = self.cost(self.circuit_weights, x_val, y_val, self.state_labels)
            res = [it + 1, loss, accuracy_train, accuracy_test]
            print(
                "Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Test accuracy: {:3f}".format(
                    *res
                )
            )
            
            self.history['loss'].append(loss)
            self.history['val_loss'].append(loss_test)

        print("circuit_weights at at iteration", it," : ", self.circuit_weights)

    def predict(self, X_test,y_test, training_columns,n_out) -> npt.NDArray[np.float64]:
        
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
                        
        predicted, fidel = self.test(self.circuit_weights, x, y_test, self.state_labels)
        
        scores = np.array(predicted ) 
        return scores        
        
    def only_QEC_predict(self, X_test,y_test, n_out) -> npt.NDArray[np.float64]:

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
        
        predicted, fidel = self.test(self.circuit_weights, x, y_test, self.state_labels)
        
        scores = np.array(predicted ) 
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
        np.save(out_dir+"/model/circuit_weights.npy", self.circuit_weights)
        
        with open(out_dir+"/model/min.txt", "a") as f:
            f.write(str(self.xmin))
        with open(out_dir+"/model/max.txt", "a") as f:
            f.write(str(self.xmax))
            

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
