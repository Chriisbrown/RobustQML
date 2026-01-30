import json
import os
import time 

import numpy as np
import numpy.typing as npt
import pandas as pd


from model.AnomalyDetectionModel import ADModelFactory, ADModel
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data.dataset import DataSet

import keras
from keras.models import load_model
from keras.layers import Dense

import tensorflow as tf

from sklearn.metrics import mean_absolute_error

#keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable(package="VariationalAutoEncoder")
class AXOVAE(keras.Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super(AXOVAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential(
        [
            keras.layers.InputLayer(shape=(input_dim,), name='model_input'),
            Dense(32,activation='relu'),
            Dense(16,activation='relu'),
            Dense(8,activation='relu'),
            Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(latent_dim,)),
                Dense(8,activation='relu'),
                Dense(16,activation='relu'),
                Dense(32,activation='relu'),
                Dense(input_dim,activation='linear',name='model_output')
            ]
        )

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reco_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def reparameterization(self, mean, log_var):
        epsilon = tf.random.normal(tf.shape(log_var))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs):
        mean, log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
    
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            x_hat, mean, log_var = self(data)
            
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(data, x_hat)
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mean) - tf.exp(log_var)
            )
            
            total_loss = reconstruction_loss + kl_loss
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }
        
    @tf.function
    def test_step(self, data):
        x_hat, mean, log_var = self(data)
            
        reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(data, x_hat)
            )
            
        kl_loss = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mean) - tf.exp(log_var)
            )
            
        total_loss = reconstruction_loss + kl_loss
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }
        
        
    def get_config(self):
        return {
            "latent_dim" : self.latent_dim,
            "input_dim" : self.input_dim,
        }


# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('AXOVariationalAutoEncoderModel')
class AXOVariationalAutoEncoderModel(ADModel):

    """AXOVariationalAutoEncoder class

    Args:
        AXOVariationalAutoEncoder (_type_): Base class of a AutoEncoderModel
    """

    def build_model(self, inputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
        """
        latent_dim = 8
        
        self.AD_model = AXOVAE( inputs_shape, latent_dim)
        print(self.AD_model.summary())

    def compile_model(self):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
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
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])


        # compile the tensorflow model setting the loss and metrics
        self.AD_model.compile(
            optimizer=self.optimizer
        )
        
        self.history = { 'loss' : [], 'val_loss' : []}


    def fit(
        self,
        X_train: DataSet,
        training_features : list,
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
        train = X_train[training_features]
        
        ds = (
            tf.data.Dataset.from_tensor_slices(train)
            .shuffle(self.training_config['batch_size'])
            #.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )
        train_size = int(len(train) * (1 - self.training_config['validation_split']) / self.training_config['batch_size'])
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)
        
        callbacks = tf.keras.callbacks.CallbackList(self.callbacks, add_history=True, model=self.AD_model)
        logs = {'val_loss' : 0}
        callbacks.on_train_begin(logs=logs)
        
        for epoch in range(1, self.training_config['epochs'] + 1):
            start_time = time.time()
            callbacks.on_epoch_begin(epoch, logs=logs)
            ibatch = 0
            running_loss = 0
            for train_x in train_ds:
                ibatch += 1
                callbacks.on_train_batch_begin(ibatch, logs=logs)
                running_loss += self.AD_model.train_step(train_x)['total_loss']
                callbacks.on_train_batch_end(ibatch, logs=logs)
            self.history['loss'].append(running_loss/ibatch)
            end_time = time.time()

            itest_batch = 0
            running_val_loss = {'total': 0, 'reco':0,'kl':0}
            for test_x in val_ds:
                itest_batch += 1
                callbacks.on_test_batch_begin(itest_batch, logs=logs)
                losses = self.AD_model.test_step(test_x)
                running_val_loss['total'] += losses['total_loss']
                running_val_loss['reco'] += losses['reconstruction_loss']
                running_val_loss['kl'] += losses['kl_loss']

                callbacks.on_test_batch_end(itest_batch, logs=logs)
            
            print('Epoch: {}, Test set total loss: {}, total reco loss: {},total kl loss: {}, time elapse for current epoch: {}, current lr: {}'
            .format(epoch, running_val_loss['total']/itest_batch, running_val_loss['reco']/itest_batch, running_val_loss['kl']/itest_batch, end_time - start_time, self.optimizer.learning_rate.numpy()))
            
            self.history['val_loss'].append(running_val_loss['total']/itest_batch)
            logs['val_loss'] = running_val_loss['total']/itest_batch
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
        
        x_hat, mean, log_var = self.AD_model(test)
        mu2 = np.linalg.vector_norm(mean,axis=1)
        ad_scores = tf.keras.losses.mse(x_hat, test)
        ad_scores = ad_scores._numpy()
        #ad_scores = mu2
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        if return_score:
            return ad_scores
        else:
            return x_hat
        
    def encoder_predict(self,X_test) -> npt.NDArray[np.float64]:
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test.to_numpy()
        else:
            test = X_test
        latent = self.AD_model.encoder(test)
        return latent
    
    def var_predict(self,X_test) -> npt.NDArray[np.float64]:
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test.to_numpy()
        else:
            test = X_test
        x_hat, mean, log_var = self.AD_model(test)
        return mean, log_var
    

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