"""AutoEncoder model child class

Written 23/12/2025 cebrown@cern.ch
"""

import os

import numpy as np
import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel
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
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        inputs = keras.layers.Input(shape=(inputs_shape,), name='model_input')
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
        
        decoder = Dense(inputs_shape,activation='sigmoid',name='model_output')(decoder)
                
        self.AD_model = keras.Model(inputs=inputs, outputs=decoder)

    def compile_model(self):
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
        train: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        # Train the model using hyperparameters in yaml config
        keras.config.disable_traceback_filtering()
        
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
        
        
    def predict(self, X_test) -> float:
        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """
        model_outputs = self.AD_model.predict(X_test)
        ad_scores = tf.keras.losses.mae(model_outputs, X_test)
        ad_scores = ad_scores._numpy()
        return ad_scores

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