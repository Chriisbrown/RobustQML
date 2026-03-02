"""Improved AutoEncoder model with regularization techniques

Based on AutoEncoderModel.py with the following improvements:
- Dropout layers to prevent overfitting
- L2 regularization on Dense layers
- Batch Normalization for training stability
- MSE loss instead of MAE for smoother gradients

References:
- Dropout: Srivastava et al. (2014) "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- Batch Normalization: Ioffe & Szegedy (2015) "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
"""

import os

import numpy as np
import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel
from data.EOSdataset import DataSet
import pandas as pd

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import keras
from keras.models import load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l2

import tensorflow as tf

from sklearn.metrics import mean_absolute_error

@ADModelFactory.register('ImprovedAutoEncoderModel')
class ImprovedAutoEncoderModel(ADModel):

    """ImprovedAutoEncoderModel class with regularization

    Args:
        ImprovedAutoEncoderModel (_type_): Base class of an Improved AutoEncoderModel
    """

    def build_model(self, inputs_shape: tuple):
        """build model override, makes the model layer by layer with improvements

        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        dropout_rate = self.model_config.get('dropout_rate', 0.2)
        l2_reg = self.model_config.get('l2_reg', 1e-4)
        use_batch_norm = self.model_config.get('use_batch_norm', True)
        
        inputs = keras.layers.Input(shape=(inputs_shape,), name='model_input')
        
        x = inputs
        for ienc, depthenc in enumerate(self.model_config['encoder_layers']):
            x = Dense(depthenc, 
                      activation='relu',
                      kernel_regularizer=l2(l2_reg))(x)
            if use_batch_norm:
                x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

        encoder_output = Dense(self.model_config['latent_dim'],
                              activation='relu',
                              kernel_regularizer=l2(l2_reg))(x)
        
        if use_batch_norm:
            encoder_output = BatchNormalization()(encoder_output)
        
        x = encoder_output
        for idec, depthdec in enumerate(self.model_config['decoder_layers']):
            x = Dense(depthdec, 
                      activation='relu',
                      kernel_regularizer=l2(l2_reg))(x)
            if use_batch_norm:
                x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        decoder = Dense(inputs_shape, activation='sigmoid', name='model_output')(x)
                
        self.AD_model = keras.Model(inputs=inputs, outputs=decoder)
        
        self.encoder_model = keras.Model(inputs=inputs, outputs=encoder_output)
        
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

        self.AD_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.training_config.get('learning_rate', 0.001)
            ),
            loss='mse',
            metrics= ['mae', 'mean_squared_error'],
        )

    def fit(
        self,
        X_train: DataSet,
        training_features : list,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        keras.config.disable_traceback_filtering()
        train = X_train[training_features]
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
        model_outputs = self.AD_model.predict(test)
        ad_scores = tf.keras.losses.mse(model_outputs, test)
        ad_scores = ad_scores._numpy()
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        if return_score:
            return ad_scores
        else:
            return model_outputs

    def encoder_predict(self, X_test, training_columns) -> npt.NDArray[np.float64]:
        """Get encoder latent representation
        
        Args:
            X_test: Input data
            training_columns: Column names
            
        Returns:
            Latent representation
        """
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test[training_columns].to_numpy()
        else:
            test = X_test
            
        return self.encoder_model.predict(test)

    @ADModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.AD_model.save(export_path)
        
        encoder_path = os.path.join(out_dir, "model/encoder_model.keras")
        self.encoder_model.save(encoder_path)
        print(f"Model saved to {export_path}")

    @ADModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        self.AD_model = load_model(f"{out_dir}/model/saved_model.keras")
        self.encoder_model = load_model(f"{out_dir}/model/encoder_model.keras")
