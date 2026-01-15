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
from keras.layers import Dense

import tensorflow as tf

from sklearn.metrics import mean_absolute_error

keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable(package="VAE")
class VAE(keras.Model):
  """variational autoencoder adapted from:
     https://www.tensorflow.org/tutorials/generative/cvae
  """

  def __init__(self, latent_dim, input_shape):
    super(VAE, self).__init__()
    self.latent_dim = latent_dim
    self.input_shape = input_shape
    self.encoder = keras.Sequential(
        [
            keras.layers.InputLayer(shape=(input_shape,), name='model_input'),
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
            Dense(input_shape,activation='linear',name='model_output')
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
    
  def get_config(self):
        return {
            "latent_dim" : self.latent_dim,
            "input_shape" : self.input_shape,
        }
    
    


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('VariationalAutoEncoderModel')
class VariationalAutoEncoderModel(ADModel):

    """VariationalAutoEncoderModel class

    Args:
        VariationalAutoEncoderModel (_type_): Base class of a AutoEncoderModel
    """

    def build_model(self, inputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
        """
        latent_dim = 8
        
        self.AD_model = VAE( latent_dim, inputs_shape)
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

        
    
    def compute_loss(self, x):
        #print("======")
        x = tf.cast(x, tf.float32)
        #print(tf.math.reduce_max(x))
        #print(tf.math.reduce_min(x))
        mean, logvar = self.AD_model.encode(x)
        #print('mean:' , mean)
        #print('logvar:',logvar)
        z = self.AD_model.reparameterize(mean, logvar)
        #print('z:',z)
        x_logit = self.AD_model.decode(z)
        #print('x_logit:',x_logit)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        #print('cross_ent:',cross_ent)
        logpx_z = -tf.reduce_sum(cross_ent,axis=1)
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        #print('logpx_z:',logpx_z)
        #print('logpz:',logpz)
        #print('logqz_x:',logqz_x)
        #print('reduced mean:',tf.reduce_mean(logpx_z + logpz - logqz_x))
        #print('mse:',tf.reduce_mean(tf.keras.losses.mse(x_logit, x)))
        return tf.reduce_mean(tf.keras.losses.mse(x_logit, x))-0.001*tf.reduce_mean(logpx_z + logpz - logqz_x)
        
        
    @tf.function
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.AD_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.AD_model.trainable_variables))
        return loss

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
        train = X_train.get_training_dataset()
        
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
            losses = []
            ibatch = 0
            loss = tf.keras.metrics.Mean()
            for train_x in train_ds:
                ibatch += 1
                callbacks.on_train_batch_begin(ibatch, logs=logs)
                self.train_step(train_x)
                callbacks.on_train_batch_end(ibatch, logs=logs)
                loss(self.compute_loss(train_x))
            self.history['loss'].append(loss.result())
            end_time = time.time()

            itest_batch = 0
            val_loss = tf.keras.metrics.Mean()
            for test_x in val_ds:
                itest_batch += 1
                callbacks.on_test_batch_begin(itest_batch, logs=logs)
                val_loss(self.compute_loss(test_x))
                running_loss = self.compute_loss(test_x)
                callbacks.on_test_batch_end(itest_batch, logs=logs)
            elbo = val_loss.result()
            
            print('Epoch: {}, Test set loss: {}, time elapse for current epoch: {}, current lr: {}'
            .format(epoch, elbo, end_time - start_time, self.optimizer.learning_rate.numpy()))
            
            self.history['val_loss'].append(elbo)
            logs['val_loss'] = elbo
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
        mean, logvar = self.AD_model.encode(x)
        mu2 = np.linalg.vector_norm(mean,axis=1)
        z = self.AD_model.reparameterize(mean, logvar)
        x_logit = self.AD_model.decode(z)
        ad_scores = tf.keras.losses.mae(x_logit, x)
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