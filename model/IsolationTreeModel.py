"""AutoEncoder model child class

Written 23/12/2025 cebrown@cern.ch
"""

import os
from typing import overload

import numpy as np
import numpy.typing as npt

from model.AnomalyDetectionModel import ADModelFactory, ADModel
from data.EOSdataset import DataSet

import pandas as pd

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

    def compile_model(self,num_samples):
        """compile the model generating callbacks and loss function
        Args:
        """
        pass

    def fit(
        self,
        X_train: DataSet,
        training_features : list,
        ):
        """Fit the model to the training dataset

        Args:
            train (DataSet): train dataset
        """
        
        num_trees = self.model_config.get('n_estimators', 50)
        max_depth = self.model_config.get('max_depth', -1)
        min_examples = self.model_config.get('min_examples', 5)
        split_axis = self.model_config.get('split_axis', 'SPARSE_OBLIQUE')
        random_seed = self.model_config.get('random_seed', 123456)
        
        if isinstance(X_train, DataSet):
            train_df = X_train.data_frame
        else:
            train_df = X_train
        
        learner_kwargs = {
            'features': training_features,
            'num_trees': num_trees,
            'max_depth': max_depth,
            'min_examples': min_examples,
            'split_axis': split_axis,
            'random_seed': random_seed
        }
        
        if split_axis == 'SPARSE_OBLIQUE':
            learner_kwargs['sparse_oblique_weights'] = self.model_config.get('sparse_oblique_weights', 'CONTINUOUS')
            learner_kwargs['sparse_oblique_projection_density_factor'] = self.model_config.get('sparse_oblique_projection_density_factor', 5.0)
        
        self.AD_model = ydf.IsolationForestLearner(**learner_kwargs).train(train_df)
        
    def fit_on_embedding(
        self,
        X_train,
        ):
        """Fit the model to the training dataset

        Args:
            train (DataSet): train dataset
        """
        
        num_trees = self.model_config.get('n_estimators', 50)
        max_depth = self.model_config.get('max_depth', -1)
        min_examples = self.model_config.get('min_examples', 5)
        split_axis = self.model_config.get('split_axis', 'SPARSE_OBLIQUE')
        random_seed = self.model_config.get('random_seed', 123456)

        train_dict = {'feature_'+str(i) : X_train[:,i] for i in range(X_train.shape[1])}
        learner_kwargs = {
            'num_trees': num_trees,
            'max_depth': max_depth,
            'min_examples': min_examples,
            'split_axis': split_axis,
            'random_seed': random_seed
        }
        
        if split_axis == 'SPARSE_OBLIQUE':
            learner_kwargs['sparse_oblique_weights'] = self.model_config.get('sparse_oblique_weights', 'CONTINUOUS')
            learner_kwargs['sparse_oblique_projection_density_factor'] = self.model_config.get('sparse_oblique_projection_density_factor', 5.0)
        
        self.AD_model = ydf.IsolationForestLearner(**learner_kwargs).train(train_dict)
        
        
    def predict(self, X_test: DataSet, training_columns) -> npt.NDArray[np.float64]:
        
        if isinstance(X_test, DataSet):
            test = X_test.data_frame
        elif isinstance(X_test, pd.DataFrame):
            test = X_test[training_columns]
        else:
            test = X_test
        """Predict method for model

        Args:
            test (DataSet): Input X test

        Returns:
            float: model prediction
        """
        ad_scores = self.AD_model.predict(X_test)
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        return ad_scores
    
    def predict_on_embedding(self, X_test: DataSet) -> npt.NDArray[np.float64]:
        
        """Predict method for model

        Args:
            test (DataSet): Input X test

        Returns:
            float: model prediction
        """
        test_dict = {'feature_'+str(i) : X_test[:,i] for i in range(X_test.shape[1])}

        ad_scores = self.AD_model.predict(test_dict)
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        return ad_scores


    def plot_loss(self):
        print("Not implemented for tree based methods")
        
    def distance(self,test,training_columns):
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