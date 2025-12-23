"""AutoEncoder model child class

Written 23/12/2025 cebrown@cern.ch
"""

import json
import os

import numpy as np
import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel

import keras
from keras.models import load_model
from keras.layers import Dense

from sklearn.metrics import mean_absolute_error
# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('AutoEncoderModel')
class AutoEncoderModel(ADModel):

    """AutoEncoderModel class

    Args:
        AutoEncoderModel (_type_): Base class of a AutoEncoderModel
    """

    def build_model(self, inputs_shape: tuple, outputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        inputs = keras.layers.Input(shape=inputs_shape, name='model_input')
        encoder = Dense(32,activation='relu')(inputs)
        encoder = Dense(16,activation='relu')(encoder)
        encoder = Dense(8,activation='relu')(encoder)
        
        self.encoder = keras.Model(inputs=inputs, outputs=encoder)
        
        decoder = Dense(16,activation='relu')(encoder)
        decoder = Dense(32,activation='relu')(decoder)
        decoder = Dense(inputs_shape,activation='sigmoid',name='model_output')(decoder)
                
        self.AD_model = keras.Model(inputs=inputs, outputs=decoder)

    def compile_model(self):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """

        # compile the tensorflow model setting the loss and metrics
        self.AD_model.compile(
            optimizer='adam',
            loss='mae'
        )

    def fit(
        self,
        train: npt.NDArray[np.float64],
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_target_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """
        # Train the model using hyperparameters in yaml config
        
        self.AD_model.fit(train)
        
        self.history = self.AD_model.fit(
            {'model_input': train},
            {'model_output':train},
            epochs=self.training_config['finetuning_epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            shuffle=True,
        )
        
        
    def predict(self, X_test: npt.NDArray[np.float64]) -> tuple:
        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            tuple: (class_predictions , pt_ratio_predictions)
        """
        model_outputs = self.AD_model.predict(X_test)
        ad_score = mean_absolute_error(X_test,model_outputs)
        return ad_score

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