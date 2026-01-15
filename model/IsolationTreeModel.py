"""AutoEncoder model child class

Written 23/12/2025 cebrown@cern.ch
"""

import os
from typing import overload

import numpy as np
import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel
from data.dataset import DataSet

from plot.basic import clusters

import ydf 

# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('IsolationTreeModel')
class IsolationTreeModel(ADModel):

    """IsolationTreeModel class

    Args:
        IsolationTreeModel (_type_): Base class of a IsolationTreeModel
    """

    def build_model(self, inputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        pass

    def compile_model(self):
        """compile the model generating callbacks and loss function
        Args:
        """
        pass

    def fit(
        self,
        X_train: DataSet,
        ):
        """Fit the model to the training dataset

        Args:
            train (DataSet): train dataset
        """
        self.AD_model = ydf.IsolationForestLearner(features=X_train.training_columns).train(X_train.data_frame)
        

    def predict(self, test: DataSet) -> npt.NDArray[np.float64]:
        """Predict method for model

        Args:
            test (DataSet): Input X test

        Returns:
            float: model prediction
        """
        ad_scores = self.AD_model.predict(test.data_frame)
        return ad_scores
    
    
    
    def plot_loss(self):
        print("Not implemented for tree based methods")
        
    def distance(self,test):
        return self.AD_model.distance(test)
    # Decorated with save decorator for added functionality
    @ADModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file

        Args:
            out_dir (str, optional): Where to save it if not in the output_directory. Defaults to "None".
        """
        # Export the model
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        export_path = os.path.join(out_dir, "model/saved_model/")
        self.AD_model.save(export_path)
        print(f"Model saved to {export_path}")

    @ADModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.AD_model = ydf.load_model(f"{out_dir}/model/saved_model/")