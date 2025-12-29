"""Anomaly Detector Model base class and additional functionality for model registering

Written 23/12/2025, cebrown@cern.ch
"""

import functools
import json
import os
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import yaml

class ADModel(ABC):
    """Parent Class for Anomaly Detection Models

    Abstract Base Class not for use directly
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir (str): Saving directory for model artefacts
        """
        self.output_directory = output_dir

        self.AD_model = None

        self.input_vars = []

        self.run_config = {}
        self.model_config = {}
        self.training_config = {}

        self.callbacks = []

        self.history = None

    def load_yaml(self, yaml_path: str):
        """Load config dictionaries

        Args:
            yaml_path (str): Path to yaml file
        """

        with open(yaml_path, 'r') as stream:
            yaml_dict = yaml.safe_load(stream)

        self.run_config = yaml_dict['run_config']
        self.model_config = yaml_dict['model_config']
        self.training_config = yaml_dict['training_config']

    @abstractmethod
    def build_model(self, **kwargs):
        """
        Build the model layers, must be written for child class
        """

    @abstractmethod
    def compile_model(self, **kwargs):
        """
        Compile the model, adding loss function and callbacks
        Must be written for child class
        """

    @abstractmethod
    def fit(self, **kwargs):
        """
        Fit the model to the training data
        Must be written for child class
        """


    def plot_loss(self):
        """Plot the loss of the model to the output directory"""
        out_dir = self.output_directory
        # Produce some basic plots with the training for diagnostics
        plot_path = os.path.join(out_dir, "plots/training")
        os.makedirs(plot_path, exist_ok=True)

        # Plot history
        loss_history(plot_path, ['mae'], self.history)
        
    def predict(self, **kwargs):
        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            tuple: (class_predictions , pt_ratio_predictions)
        """

    def save_decorator(save_func):
        """Decorator used to include additional
        saving functionality for child classes
        """

        @functools.wraps(save_func)
        def wrapper(self, out_dir: str = "None"):
            """Wrapper adding saving functionality

            Args:
                out_dir (str): Where to save the model. Defaults to
                None but overridden to output_directory.
            """
            if out_dir == "None":
                out_dir = self.output_directory
            # Save additional jsons associated with model
            # Dump input variables
            with open(os.path.join(out_dir, "input_vars.json"), "w") as f:
                json.dump(self.input_vars, f, indent=4)

            save_func(self, out_dir)

        return wrapper

    def load_decorator(load_func):
        """Decorator used to include additional
        loading functionality for child classes
        """

        @functools.wraps(load_func)
        def wrapper(self, out_dir: str = "None"):
            """Wrapper adding loading functionality

            Args:
                out_dir (str): Where to load the model from. Defaults to
                None but overridden to output_directory.
            """
            if out_dir == "None":
                out_dir = self.output_directory
            # Save additional jsons associated with model
            # Dump input variables
            with open(os.path.join(out_dir, "input_vars.json"), "r") as f:
                self.input_vars = json.load(f)
            # Do the rest of the loading, defined in child class
            load_func(self, out_dir)

        return wrapper

    def set_labels(self, input_vars: str):
        """Set internal labels

        Args:
            input_vars (str): Input variable names
        """
        self.input_vars = input_vars

class ADModelFactory:
    """The factory class for creating Anomlay Detection Models"""

    registry = {}
    """ Internal registry for available Anomaly Detection Models """

    @classmethod
    def register(cls, name: str):
        """Decorator for registering new anomaly detection models

        Args:
            name (str): Name of the model
        """

        def inner_wrapper(wrapped_class: ADModel):
            if name in cls.registry:
                print('Anomaly Detection Model %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_ADModel(cls, name: str, folder: str, **kwargs) -> 'ADModel':
        """Factory command to create the Anomaly Detection Model"""

        ad_class = cls.registry[name]
        model = ad_class(folder, **kwargs)

        return model