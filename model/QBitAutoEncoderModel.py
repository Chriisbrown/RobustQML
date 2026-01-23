import os

import numpy as np
import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel
from data.dataset import DataSet
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import keras
from keras.models import load_model
from keras.layers import Dense

import tensorflow as tf

from sklearn.metrics import mean_absolute_error
# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('AutoEncoderModel')
class AutoEncoderModel(ADModel):

    """AutoEncoderModel class

    Args:
        AutoEncoderModel (_type_): Base class of a AutoEncoderModel
    """

    def build_model(self, inputs_shape: tuple):
        """build model override

        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        total_object = int(inputs_shape[0] / 3)
        
        nt = total_object
        
        # Registers (match the labels in your diagram)
        q = QuantumRegister(total_object, "q")     
        t = QuantumRegister(nt, "t")    
        a = QuantumRegister(1, "a") 
        m = ClassicalRegister(1, "m")  

        qc = QuantumCircuit(q, t, a, m)

        # Parameters θ0..θnq
        x = [Parameter(f"x{i}") for i in range(total_object)]
        y = [Parameter(f"y{i}") for i in range(total_object)]
        z = [Parameter(f"z{i}") for i in range(total_object)]

        # Ry layer on q0..q3
        for i in range(total_object):
            qc.rx(x[i], q[i])
            qc.ry(y[i], q[i])
            qc.rz(z[i], q[i])

        # CNOT pattern
        for i in range(total_object):
            for j in range(i+1, total_object):
                qc.cx(q[i], q[j])

        # Ancilla H
        qc.h(a[0])

        # Controlled-SWAPs
        for i in reversed(range(nt)):
            qc.cswap(a[0], q[-(i+1)], t[-(i+1)])

        # Final H + measure ancilla
        qc.h(a[0])
        qc.measure(a[0], m[0])
        
        self.AD_model = qc
                
        

    def compile_model(self):
        """compile the model generating callbacks and loss function
        Args:
        """

    def fit(
        self,
        X_train: DataSet,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        # Train the model using hyperparameters in yaml config
        keras.config.disable_traceback_filtering()
        train = X_train.get_training_dataset()
        history = self.AD_model.fit(
            train.to_numpy(),
            train.to_numpy(),
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            shuffle=True,
            callbacks=self.callbacks,
        )
        
        self.history = history.history
        
    def predict(self, X_test, return_score = True) -> npt.NDArray[np.float64]:
        
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
        model_outputs = self.AD_model.predict(test)
        ad_scores = tf.keras.losses.mse(model_outputs, test)
        ad_scores = ad_scores._numpy()
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        if return_score:
            return ad_scores
        else:
            return model_outputs

    # Decorated with save decorator for added functionality
    @ADModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.AD_model.save(export_path)
        print(f"Model saved to {export_path}")

    @ADModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.AD_model = load_model(f"{out_dir}/model/saved_model.keras")