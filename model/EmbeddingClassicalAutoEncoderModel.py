import os
import time
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

from model.VariationalAutoEncoderModel import VAE


import matplotlib.pyplot as plt
from plot import style

import numpy as np
import scipy
# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('EmbeddingVAEModel')
class EmbeddingVAEModel(ADModel):

    """EmbeddingVAEModel class

    Args:
        EmbeddingVAEModel (_type_): Base class of a AEModel
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
        
        
        self.AD_model = VAE(self.model_config['embedding_dim'], self.model_config['latent_dim'],self.model_config['encoder_layers'],self.model_config['decoder_layers'])
        print(self.AD_model.summary())
        
        self.xmin = xmin
        self.xmax = xmax
        
        
    def compile_model(self,num_samples):
        """compile the model generating callbacks and loss function
        Args:
        """
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience'],mode='min'),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
                mode='min'
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
        training_columns: list,
    ):
        pass   
    
    def only_VAE_fit(
        self,
        X_train: DataSet,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        
        
        
        #X_train = np.clip(X_train, self.xmin, self.xmax)
        #x_train = np.array((((X_train - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
        x_train = X_train
        ds = (
            tf.data.Dataset.from_tensor_slices(x_train)
            .shuffle(self.training_config['batch_size'])
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )
        train_size = int(len(x_train) * (1 - self.training_config['validation_split']) / self.training_config['batch_size'])
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
                self.AD_model.train_step(train_x)
                callbacks.on_train_batch_end(ibatch, logs=logs)
                loss(self.AD_model.compute_loss(train_x))
            self.history['loss'].append(loss.result())
            end_time = time.time()

            itest_batch = 0
            val_loss = tf.keras.metrics.Mean()
            for test_x in val_ds:
                itest_batch += 1
                callbacks.on_test_batch_begin(itest_batch, logs=logs)
                val_loss(self.AD_model.compute_loss(test_x))
                running_loss = self.AD_model.compute_loss(test_x)
                callbacks.on_test_batch_end(itest_batch, logs=logs)
            elbo = val_loss.result()
            
            print('Epoch: {}, Test set loss: {}, time elapse for current epoch: {}, current lr: {}'
            .format(epoch, elbo, end_time - start_time, self.optimizer.learning_rate.numpy()))
            
            self.history['val_loss'].append(elbo)
            logs['val_loss'] = elbo
            callbacks.on_epoch_end(epoch, logs=logs)
        callbacks.on_train_end(logs=logs)
        
    def predict(self, X_test,return_score = True):
        pass
        
    def only_VAE_predict(self, X_test,return_score = True) -> npt.NDArray[np.float64]:

        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """

        #X_test = np.clip(X_test, self.xmin, self.xmax)
        #x = (((X_test - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi
        x = X_test
        mean, logvar = self.AD_model.encode(x)
        mu2 = scipy.linalg.norm(mean,axis=1)
        z = self.AD_model.reparameterize(mean, logvar)
        x_logit = self.AD_model.decode(z)
        ad_scores = tf.keras.losses.mse(x_logit, x)
        ad_scores = ad_scores._numpy()
        #ad_scores = mu2

        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        if return_score:
            return ad_scores
        else:
            return x_logit

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
        self.AD_model = load_model(f"{out_dir}/model/saved_model.keras")
        
        with open(f"{out_dir}/model/min.txt") as f:
            self.xmin = float(f.read())
        with open(f"{out_dir}/model/max.txt") as f:
            self.xmax = float(f.read())

# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('EmbeddingCAEModel')
class EmbeddingCAEModel(ADModel):

    """EmbeddingCAEModel class

    Args:
        EmbeddingCAEModel (_type_): Base class of a PennyLaneQAEModel
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
        
        inputs = keras.layers.Input(shape=(self.model_config['embedding_dim'],), name='model_input')
        #bn_inputs = keras.layers.BatchNormalization()(inputs)
        for ienc, depthenc in enumerate(self.model_config['encoder_layers']):
            if ienc == 0:
                encoder = Dense(depthenc, activation='relu')(inputs)
            else:
                encoder = Dense(depthenc, activation='relu')(encoder)
        #encoder = keras.layers.BatchNormalization()(encoder)
        encoder = Dense(self.model_config['latent_dim'],activation='relu')(encoder)
                
        for idec, depthdec in enumerate(self.model_config['decoder_layers']):
            if idec == 0:
                decoder = Dense(depthdec, activation='relu')(encoder)
            else:
                decoder = Dense(depthdec, activation='relu')(decoder)
        #decoder = keras.layers.BatchNormalization()(decoder)
        decoder = Dense(self.model_config['embedding_dim'],activation='linear',name='model_output')(decoder)
                
        self.AD_model = keras.Model(inputs=inputs, outputs=decoder)
        
        self.xmin = xmin
        self.xmax = xmax
        
        self.AD_model.summary()
        
        
    def compile_model(self,num_samples):
        """compile the model generating callbacks and loss function
        Args:
        """
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])
        
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
            optimizer=self.optimizer,
            loss='mse',
            metrics= ['mae', 'mean_squared_error'],
        )

        

    def fit(
        self,
        X_train: DataSet,
        training_columns: list,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        
        
        
        train_embeddings = self.embedding_model.encoder_predict(X_train[training_columns].to_numpy(),training_columns) 
        train_embeddings = np.clip(train_embeddings, self.xmin, self.xmax)
        x_train = np.array((((train_embeddings - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
                  
                  
        keras.config.disable_traceback_filtering()
        history = self.AD_model.fit(
            x_train,
            x_train,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            verbose=self.run_config['verbose'],
            validation_split=self.training_config['validation_split'],
            shuffle=True,
            callbacks=self.callbacks,
        )
        
        self.history = history.history
        
    def only_CAE_fit(
        self,
        X_train: DataSet,
    ):
        """Fit the model to the training dataset

        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
        """
        
        print("DEBUG CAE")
        print(self.xmin)
        print(self.xmax)
        print(np.max(X_train))
        print(np.min(X_train))
        X_train = np.clip(X_train, self.xmin, self.xmax)
        x_train = np.array((((X_train - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi)
                  
        keras.config.disable_traceback_filtering()
        history = self.AD_model.fit(
            x_train,
            x_train,
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
        
        embeddings = self.embedding_model.encoder_predict(test,training_columns) 
        #x = np.zeros_like(embeddings)
        #for i_embedding in range(embeddings.shape[1]):
        embeddings = np.clip(embeddings, self.xmin, self.xmax)
        x = (((embeddings - self.xmin) / (self.xmax- self.xmin)) * 2*np.pi) - np.pi
        
        model_outputs = self.AD_model.predict(x)
        ad_scores = tf.keras.losses.mse(model_outputs, x)
        ad_scores = ad_scores._numpy()
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        if return_score:
            return ad_scores
        else:
            return model_outputs
        
        
    def only_CAE_predict(self, X_test,return_score = True) -> npt.NDArray[np.float64]:

        """Predict method for model

        Args:
            X_test (npt.NDArray[np.float64]): Input X test

        Returns:
            float: model prediction
        """
        print("DEBUG CAE")

        print(np.max(X_test))
        print(np.min(X_test))
        X_test = np.clip(X_test, self.xmin, self.xmax)
        print(self.xmin)
        print(self.xmax)
        
        x = (((X_test - self.xmin) / (self.xmax - self.xmin)) * 2*np.pi) - np.pi

        model_outputs = self.AD_model(x)
        ad_scores = tf.keras.losses.mse(model_outputs, x)
        ad_scores = ad_scores._numpy()
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
        export_path = os.path.join(out_dir, "model/saved_model.keras")
        self.AD_model.save(export_path)
        
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
        self.AD_model = load_model(f"{out_dir}/model/saved_model.keras")
        
        with open(f"{out_dir}/model/min.txt") as f:
            self.xmin = float(f.read())
        with open(f"{out_dir}/model/max.txt") as f:
            self.xmax = float(f.read())
