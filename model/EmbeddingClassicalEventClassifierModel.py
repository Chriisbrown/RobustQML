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


from plot.basic import loss_history, plot_2d

from model.common import ECfromFolder, fromFolder


import matplotlib.pyplot as plt
from plot import style

import numpy as np

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
        
        inputs = keras.layers.Input(shape=(self.model_config['latent_dim'],), name='model_input')
        bn_inputs = keras.layers.BatchNormalization()(inputs)
        for ienc, depthenc in enumerate(self.model_config['encoder_layers']):
            if ienc == 0:
                encoder = Dense(depthenc, activation='relu')(bn_inputs)
            else:
                encoder = Dense(depthenc, activation='relu')(encoder)
        encoder = Dense(self.model_config['latent_dim'],activation='relu')(encoder)

        output = Dense(output_shape,activation='sigmoid',name='model_output')(encoder)
                
        self.EC_model = keras.Model(inputs=inputs, outputs=output)
        
        self.xmin = xmin
        self.xmax = xmax
        
        
    def compile_model(self,num_samples):
        """compile the model generating callbacks and loss function
        Args:
        """
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience']),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
        ]

        # compile the tensorflow model setting the loss and metrics
        self.EC_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics= ['accuracy'],
        )

        

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
        
        
        
        
        train_embeddings = self.embedding_model.encoder_predict(X_train[training_columns].to_numpy(),training_columns) 
        train_embeddings = np.clip(train_embeddings, self.xmin, self.xmax)
        x_train = np.array((((train_embeddings - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
                  
                  
        keras.config.disable_traceback_filtering()
        history = self.EC_model.fit(
            x_train,
            y_train,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            shuffle=True,
            callbacks=self.callbacks,
        )
        
        self.history = history.history
        
    def only_CEC_fit(
        self,
        X_train: DataSet,
        y_train,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        
        X_train = np.clip(X_train, self.xmin, self.xmax)
        x_train = np.array((((X_train - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
                  
                  
        keras.config.disable_traceback_filtering()
        history = self.EC_model.fit(
            x_train,
            y_train,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            shuffle=True,
            callbacks=self.callbacks,
        )
        
        self.history = history.history
        

    def predict(self, X_test, training_columns) -> npt.NDArray[np.float64]:
        
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
        
        embeddings = self.embedding_model.encoder_predict(test,training_columns) 
        embeddings = np.clip(embeddings, self.xmin, self.xmax)
        x = (((embeddings - self.xmin) / (self.xmax- self.xmin)) * 2*np.pi) - np.pi
        
        model_outputs = self.EC_model.predict(x)
        return model_outputs
        
        
    def only_CEC_predict(self, X_test) -> npt.NDArray[np.float64]:

        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """

        X_test = np.clip(X_test, self.xmin, self.xmax)
        x = (((X_test - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi
        model_outputs = np.array(self.EC_model(x))
        return 1 - model_outputs[:,0]
    
    
    
    @ECModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        # Use keras save format !NOT .h5! due to depreciation
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.EC_model.save(export_path)
        
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
        self.EC_model = load_model(f"{out_dir}/model/saved_model.keras")
        
        with open(f"{out_dir}/model/min.txt") as f:
            self.xmin = float(f.read())
        with open(f"{out_dir}/model/max.txt") as f:
            self.xmax = float(f.read())

