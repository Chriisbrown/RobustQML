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


from plot.basic import loss_history, plot_2d

from model.common import fromFolder


import matplotlib.pyplot as plt
from plot import style

import numpy as np

# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('EmbeddingCAEModel')
class EmbeddingCAEModel(ADModel):

    """EmbeddingCAEModel class

    Args:
        EmbeddingCAEModel (_type_): Base class of a PennyLaneQAEModel
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
        
        inputs = keras.layers.Input(shape=(self.model_config['embedding_dim'],), name='model_input')
        for ienc, depthenc in enumerate(self.model_config['encoder_layers']):
            if ienc == 0:
                encoder = Dense(depthenc, activation='relu')(inputs)
            else:
                encoder = Dense(depthenc, activation='relu')(encoder)

        encoder = Dense(self.model_config['latent_dim'],activation='relu')(encoder)
                
        for idec, depthdec in enumerate(self.model_config['decoder_layers']):
            if idec == 0:
                decoder = Dense(depthdec, activation='relu')(encoder)
            else:
                decoder = Dense(depthdec, activation='relu')(decoder)
        
        decoder = Dense(self.model_config['embedding_dim'],activation='sigmoid',name='model_output')(decoder)
                
        self.AD_model = keras.Model(inputs=inputs, outputs=decoder)
        
        
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
        self.AD_model.compile(
            optimizer='adam',
            loss='mae',
            metrics= ['mae', 'mean_squared_error'],
        )

        

    def fit(
        self,
        X_train: DataSet,
        training_columns: list,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        
        
        
        train_embeddings = self.embedding_model.encoder_predict(X_train[training_columns].to_numpy(),training_columns) 
        x_train = np.array((((train_embeddings - np.min(train_embeddings)) / (np.max(train_embeddings) - np.min(train_embeddings))) * 2*np.pi) - np.pi)
                  
                  
        keras.config.disable_traceback_filtering()
        history = self.AD_model.fit(
            x_train,
            x_train,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            shuffle=True,
            callbacks=self.callbacks,
        )
        
        self.history = history.history
        
    def only_CAE_fit(
        self,
        X_train: DataSet,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        
        
        x_train = np.array((((X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))) * 2*np.pi) - np.pi)
                  
                  
        keras.config.disable_traceback_filtering()
        history = self.AD_model.fit(
            x_train,
            x_train,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            shuffle=True,
            callbacks=self.callbacks,
        )
        
        self.history = history.history
        

    def predict(self, X_test, training_columns,return_score = True) -> npt.NDArray[np.float64]:
        
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
        #x = np.zeros_like(embeddings)
        #for i_embedding in range(embeddings.shape[1]):
        x = (((embeddings - np.min(embeddings)) / (np.max(embeddings) - np.min(embeddings))) * 2*np.pi) - np.pi
        
        model_outputs = self.AD_model.predict(x)
        ad_scores = tf.keras.losses.mse(model_outputs, x)
        ad_scores = ad_scores._numpy()
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        if return_score:
            return ad_scores
        else:
            return model_outputs
        
        
    def only_CAE_predict(self, X_test,return_score = True) -> npt.NDArray[np.float64]:

        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """
        
        #x = np.zeros_like(embeddings)
        #for i_embedding in range(embeddings.shape[1]):
        x = (((X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))) * 2*np.pi) - np.pi
        
        model_outputs = self.AD_model.predict(x)
        ad_scores = tf.keras.losses.mse(model_outputs, x)
        ad_scores = ad_scores._numpy()
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        if return_score:
            return ad_scores
        else:
            return model_outputs
    
    
    
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
