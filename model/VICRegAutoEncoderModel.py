"""AutoEncoder model child class

Written 02/01/2026 cebrown@cern.ch
"""

import json
import os
import time 

import numpy as np
import numpy.typing as npt
import pandas as pd

from model.AnomalyDetectionModel import ADModelFactory, ADModel
from data.dataset import DataSet

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import keras
from keras.models import load_model
from keras.layers import Dense,BatchNormalization,ReLU

import tensorflow as tf

from sklearn.metrics import mean_absolute_error

from model.VariationalAutoEncoderModel import VAE

from tqdm import tqdm

  
class VICRegPreprocessing(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.augment = tf.keras.Sequential([
            tf.keras.layers.GaussianNoise(0.01)
        ])
        
    def call(self, x):
        return self.augment(x), self.augment(x)

def off_diagonal(x):
    n = tf.shape(x)[0]
    mask = ~tf.cast(tf.eye(n), tf.bool)
    return tf.boolean_mask(x, mask)

@keras.saving.register_keras_serializable(package="VICReg")
class VICReg(keras.Model):
    def __init__(self, latent_dim, input_shape,batch_size):
        super(VICReg, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        
        self.num_features = input_shape
        self.batch_size = batch_size
        
        self.backbone = keras.Sequential(
            [
                keras.layers.InputLayer(shape=(input_shape,), name='model_input'),
                Dense(32,activation='relu'),
                Dense(16,activation='relu'),
                Dense(8,activation='relu'),
                Dense(latent_dim),
            ]
        )
        
        self.projector = keras.Sequential(
            [
                Dense(self.num_features,activation='linear'),
                BatchNormalization(),
                ReLU(),
                Dense(self.num_features,activation='linear'),
                BatchNormalization(),
                ReLU(),
                Dense(self.num_features,activation='linear'),
                BatchNormalization(),
                ReLU(),
                Dense(self.num_features,activation='linear'),
            ]
        )
        
        self.sim_coeff = 50
        self.std_coeff = 50
        self.cov_coeff = 1

        self.loss_tracker = tf.keras.metrics.Mean(name="Total_Loss")
        self.loss_tracker_repr = tf.keras.metrics.Mean(name="Total_Loss_repr")
        self.loss_tracker_std = tf.keras.metrics.Mean(name="Total_Loss_std")
        self.loss_tracker_cov = tf.keras.metrics.Mean(name="Total_Loss_cov")


    @tf.function
    def train_step(self, x_in):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        x,x_p = x_in
        with tf.GradientTape() as tape:
            x = self.projector(self.backbone(x, training=True) , training=True)
            x_p = self.projector(self.backbone(x_p, training=True) , training=True)
            
            repr_loss = keras.losses.mean_squared_error(x,x_p)
            
            x = x - tf.reduce_mean(x, axis=0, keepdims=True)
            x_p = x_p - tf.reduce_mean(x_p, axis=0, keepdims=True)
            
            std_x = tf.sqrt(tf.math.reduce_variance(x, axis=0) + 0.0001)
            std_x_p = tf.sqrt(tf.math.reduce_variance(x_p, axis=0) + 0.0001)
            
            std_loss = tf.reduce_mean(tf.nn.relu(1.0 - std_x)) / 2 + tf.reduce_mean(tf.nn.relu(1.0 - std_x_p)) / 2
    
            cov_x = tf.linalg.matmul(x, x, transpose_a=True) / (self.batch_size - 1.0)
            cov_x_p = tf.linalg.matmul(x_p, x_p, transpose_a=True) / (self.batch_size - 1.0)
            
            cov_loss = (tf.reduce_sum(tf.square(off_diagonal(cov_x))) +  tf.reduce_sum(tf.square(off_diagonal(cov_x_p)))) / float(self.num_features)
    
            loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
            
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            self.loss_tracker.update_state(loss)
            self.loss_tracker_repr.update_state(repr_loss)
            self.loss_tracker_cov.update_state(cov_loss)
            self.loss_tracker_std.update_state(std_loss)

            return {"Loss":self.loss_tracker.result(),
                    "Representation Loss":self.loss_tracker_repr.result(),
                    "Covariance Loss":self.loss_tracker_cov.result(),
                    "Standard Deviation Loss":self.loss_tracker_std.result()
               } 
           
    def get_config(self):
            return {
                "latent_dim" : self.latent_dim,
                "input_shape" : self.input_shape,
            }
    


# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('VICRegModel')
class VICRegModel(ADModel):

    """VICRegModel class

    Args:
        VICRegModel (_type_): Base class of a AutoEncoderModel
    """

    def build_model(self, inputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
        """
        latent_dim = 8
        
        self.vicreg_model = VICReg(latent_dim=latent_dim, input_shape=inputs_shape,batch_size=self.training_config['batch_size'])
        self.vae_model = VAE(input_dim=latent_dim, latent_dim=4)
        print(self.vicreg_model.summary())
        print(self.vae_model.summary())

    def compile_model(self):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """
        
        self.repr_optimizer = keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])

        # compile the tensorflow model setting the loss and metrics
        self.vicreg_model.compile(
            optimizer=self.repr_optimizer
        )
        
        self.history = { 'Vicreg Loss' : [], 'Representation Loss' : [], 'Covariance Loss' : [], 'Standard Deviation Loss' : [],'vae loss' : [], 'vae val_loss' : []}
        
        self.vae_callbacks = [
            EarlyStopping(monitor='vae val_loss', patience=self.training_config['EarlyStopping_patience']),
            ReduceLROnPlateau(
                monitor='vae val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
            ),
        ]
        
        self.vae_optimizer = keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])

        # compile the tensorflow model setting the loss and metrics
        self.vae_model.compile(
            optimizer=self.vae_optimizer
        )


    def fit(
        self,
        X_train: DataSet,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
            y_train (npt.NDArray[np.float64]): y train classification targets
            pt_target_train (npt.NDArray[np.float64]): y train pt regression targets
            sample_weight (npt.NDArray[np.float64]): sample weighting
        """
        # Train the model using hyperparameters in yaml config
        keras.config.disable_traceback_filtering()
        augment = VICRegPreprocessing()
        train = X_train.get_training_dataset()
        ds = (
            tf.data.Dataset.from_tensor_slices(train)
            .shuffle(self.training_config['batch_size'])
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )
        
        
        for epoch in tqdm(range(0, self.training_config['constrastive_epochs'],1)):
            running_loss = {'total': 0, 'repr':0,'cov':0, 'std':0}
            ibatch = 0
            for train_x,train_x_p in ds:
                ibatch += 1

                loss = self.vicreg_model.train_step((train_x,train_x_p))
                
                running_loss['total'] += loss["Loss"]
                running_loss['repr'] += loss["Representation Loss"]
                running_loss['cov'] += loss["Covariance Loss"]
                running_loss['std'] += loss["Standard Deviation Loss"]
                
            self.history['Vicreg Loss'].append(running_loss['total'] /ibatch)
            self.history['Representation Loss'].append(running_loss['repr'] /ibatch)
            self.history['Covariance Loss'].append(running_loss['cov'] /ibatch)
            self.history['Standard Deviation Loss'].append(running_loss['std'] /ibatch)
            
            print('Epoch: {}, total loss: {}, represenation loss: {},cov loss: {}, std loss: {}'
                .format(epoch, running_loss['total']/ibatch, running_loss['repr']/ibatch, running_loss['cov']/ibatch, running_loss['std']/ibatch))
        
        ds = (
            tf.data.Dataset.from_tensor_slices(train)
            .shuffle(self.training_config['batch_size'])
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )
        
        train_size = int(len(train) * (1 - self.training_config['validation_split']) / self.training_config['batch_size'])
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)
        
        callbacks = tf.keras.callbacks.CallbackList(self.callbacks, add_history=True, model=self.AD_model)
        logs = {'vae val_loss' : 0}
        callbacks.on_train_begin(logs=logs)
        
        for epoch in range(1, self.training_config['epochs'] + 1):
            start_time = time.time()
            callbacks.on_epoch_begin(epoch, logs=logs)
            losses = []
            ibatch = 0
            loss = tf.keras.metrics.Mean()
            for train_x in train_ds:
                latent_x = self.vicreg_model.backbone(train_x)
                ibatch += 1
                callbacks.on_train_batch_begin(ibatch, logs=logs)
                self.vae_model.train_step(latent_x)
                callbacks.on_train_batch_end(ibatch, logs=logs)
                loss(self.vae_model.compute_loss(latent_x))
            self.history['vae loss'].append(loss.result())
            end_time = time.time()

            itest_batch = 0
            val_loss = tf.keras.metrics.Mean()
            for test_x in val_ds:
                itest_batch += 1
                latent_test = self.vicreg_model.backbone(test_x)
                callbacks.on_test_batch_begin(itest_batch, logs=logs)
                val_loss(self.vae_model.compute_loss(latent_test))
                callbacks.on_test_batch_end(itest_batch, logs=logs)
            elbo = val_loss.result()
            
            print('Epoch: {}, Test set loss: {}, time elapse for current epoch: {}, current lr: {}'
            .format(epoch, elbo, end_time - start_time, self.vae_optimizer.learning_rate.numpy()))
            
            self.history['vae val_loss'].append(elbo)
            logs['vae val_loss'] = elbo
            callbacks.on_epoch_end(epoch, logs=logs)
        callbacks.on_train_end(logs=logs) 

                
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
            tuple: (class_predictions , pt_ratio_predictions)
        """
        
        x = tf.cast(test, tf.float32)
        x_latent = self.vicreg_model.backbone(x)
        mean, logvar = self.vae_model.encode(x_latent)
        mu2 = np.linalg.vector_norm(mean,axis=1)
        z = self.vae_model.reparameterize(mean, logvar)
        x_logit = self.vae_model.decode(z)
        ad_scores = tf.keras.losses.mae(x_logit,x_latent)
        ad_scores = ad_scores._numpy()
        if return_score:
            return ad_scores
        else:
            return x_logit

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
        export_path = os.path.join(out_dir, "model/vicreg_saved_model.keras")
        self.vicreg_model.save(export_path)
        print(f"Model saved to {export_path}")
        export_path = os.path.join(out_dir, "model/vae_saved_model.keras")
        self.vae_model.save(export_path)
        print(f"Model saved to {export_path}")

    @ADModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file

        Args:
            out_dir (str, optional): Where to load it if not in the output_directory. Defaults to "None".
        """
        # Load the model
        self.vicreg_model = load_model(f"{out_dir}/model/vicreg_saved_model.keras")
        self.vae_model = load_model(f"{out_dir}/model/vae_saved_model.keras")