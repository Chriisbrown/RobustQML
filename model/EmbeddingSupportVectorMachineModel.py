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

from sklearn import svm

import pickle

# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('EmbeddingSVMModel')
class EmbeddingSVMModel(ADModel):

    """EmbeddingSVMModel class

    Args:
        EmbeddingSVMModel (_type_): Base class of a PennyLaneQAEModel
    """
    
    def __init__(self,output_dir):
        super().__init__(output_dir)
        
    def load_embedding_model(self,model_folder):
        self.embedding_model = fromFolder(model_folder)

    def build_model(self, inputs_shape: tuple, xmin,xmax):
        """build model override

        Args:
            inputs_shape (tuple): Shape of the input
        """

        self.AD_model = svm.OneClassSVM(verbose=True,max_iter=1000)
        
        self.xmin = xmin
        self.xmax = xmax
        
        
    def compile_model(self,num_samples):
        """compile the model generating callbacks and loss function
        Args:
        """
        pass

    def fit(
        self,
        train_embeddings
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        
        
        
        #train_embeddings = self.embedding_model.encoder_predict(X_train[training_columns].to_numpy(),training_columns) 
        train_embeddings = np.clip(train_embeddings, self.xmin, self.xmax)
        x_train = np.array((((train_embeddings - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
        print('SVM start fit')
        self.AD_model.fit(x_train[0:1000000])
        print('SVM stop fit')

        

        

    def predict(self, embeddings,return_score = True) -> npt.NDArray[np.float64]:
        """
        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """
        
        #embeddings = self.embedding_model.encoder_predict(test,training_columns) 
        #x = np.zeros_like(embeddings)
        #for i_embedding in range(embeddings.shape[1]):
        embeddings = np.clip(embeddings, self.xmin, self.xmax)
        x = (((embeddings - self.xmin) / (self.xmax- self.xmin)) * 2*np.pi) - np.pi
        
        model_outputs = self.AD_model.score_samples(x)
        ad_scores = 1 - model_outputs
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
        export_path = os.path.join(out_dir, "model/saved_model.pkl")
        
        with open(export_path, "wb") as f:
            pickle.dump(self.AD_model, f, protocol=5)
        
        with open(out_dir+"/model/min.txt", "a") as f:
            f.write(str(self.xmin))
        with open(out_dir+"/model/max.txt", "a") as f:
            f.write(str(self.xmax))
            
        print(f"Model saved to {export_path}")

    @ADModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model        
        with open(f"{out_dir}/model/saved_model.pkl", "rb") as f:
            self.AD_model = pickle.load(f)
        
        with open(f"{out_dir}/model/min.txt") as f:
            self.xmin = float(f.read())
        with open(f"{out_dir}/model/max.txt") as f:
            self.xmax = float(f.read())

