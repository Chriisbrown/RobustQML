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

def SimCLR_contrastive_loss(z1, z2, temperature=0.5,**kwargs):
        # First: Concatenate both batches of embeddings (positive pairs)
        z = tf.concat([z1, z2], axis=0)  # shape: (2N, D), where N is batch size

        # Cosine similarity matrix between all embeddings (assumes z is L2-normalized)
        sim = tf.matmul(z, z, transpose_b=True)  # shape: (2N, 2N), sim[i][j] = similarity between sample i and j
        sim /= temperature  # scale similarities by temperature (sharpening)

        # Create some positive/negative pair labels — position i matches with i + N ( same image, different view)
        batch_size = tf.shape(z1)[0]
        labels = tf.range(batch_size)
        labels = tf.concat([labels, labels], axis=0)  # shape: (2N,)

        # Remove self-similarities (the diagonal) from similarity matrix, dont need to do similarity with itselt
        mask = tf.eye(2 * batch_size)  # identity matrix
        sim = sim - 1e9 * mask  # set diagonal to a large negative number so it's ignored in softmax, HACKY! COuld do masking but expensive

        # Get positive similarities from the similarity matrix
        # Positive pairs are offset by +N and -N in the 2N batch
        positives = tf.concat([
            tf.linalg.diag_part(sim, k=batch_size),   # sim[i][i+N]
            tf.linalg.diag_part(sim, k=-batch_size)   # sim[i+N][i]
        ], axis=0)  # shape: (2N,)

        # Step 6: Compute the famous NT-Xent loss
        numerator = tf.exp(positives)  # exp(similarity of positive pairs)
        denominator = tf.reduce_sum(tf.exp(sim), axis=1)  # sum over all other similarities for each sample
        loss = -tf.math.log(numerator / denominator)  # -log(positive / all)

        # Step 7: Return average loss over the batch
        return tf.reduce_mean(loss)
    
    
def VICReg_contrastive_loss(x,x_p,batch_size,num_features,sim_coeff=50,std_coeff=50,cov_coeff=1,**kwargs):
        repr_loss = keras.losses.mean_squared_error(x,x_p)
            
        x = x - tf.reduce_mean(x, axis=0, keepdims=True)
        x_p = x_p - tf.reduce_mean(x_p, axis=0, keepdims=True)
            
        std_x = tf.sqrt(tf.math.reduce_variance(x, axis=0) + 0.0001)
        std_x_p = tf.sqrt(tf.math.reduce_variance(x_p, axis=0) + 0.0001)
            
        std_loss = tf.reduce_mean(tf.nn.relu(1.0 - std_x)) / 2 + tf.reduce_mean(tf.nn.relu(1.0 - std_x_p)) / 2
    
        cov_x = tf.linalg.matmul(x, x, transpose_a=True) / (batch_size - 1.0)
        cov_x_p = tf.linalg.matmul(x_p, x_p, transpose_a=True) / (batch_size - 1.0)
            
        cov_loss = (tf.reduce_sum(tf.square(off_diagonal(cov_x))) +  tf.reduce_sum(tf.square(off_diagonal(cov_x_p)))) / float(num_features)
    
        return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss


def choose_loss(choice: str):
    """Choose the aggregator keras object based on an input string."""
    if choice not in ["SimCLR", "VICReg"]:
        raise ValueError(
            choice, "Not implemented"
        )
    if choice == "SimCLR":
        return SimCLR_contrastive_loss
    elif choice == "VICReg":
        return VICReg_contrastive_loss


  
class L2NormalizeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
  
  
class Preprocessing(tf.keras.layers.Layer):
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

@keras.saving.register_keras_serializable(package="Contrastive")
class Contrastive(keras.Model):
    def __init__(self, projection_dim, input_shape,batch_size,backbone_layers,projection_blocks, loss):
        super(Contrastive, self).__init__()
        self.latent_dim = projection_dim
        self.input_shape = input_shape
        
        self.num_features = input_shape
        self.batch_size = batch_size
        
        self.backbone_layers = backbone_layers
        self.projection_blocks = projection_blocks
        
        bb_layers = [Dense(backbone_layer,activation='relu') for backbone_layer in backbone_layers]
        bb_layers.insert(0,keras.layers.InputLayer(shape=(input_shape,), name='model_input'))
        bb_layers.append(Dense(projection_dim))
        
        self.backbone = keras.Sequential(bb_layers)
        
        project_layers = []
        for i in range(projection_blocks):
            project_layers.append(Dense(self.num_features,activation='linear'))
            project_layers.append(BatchNormalization())
            project_layers.append(ReLU())
        project_layers.append(Dense(self.num_features, use_bias=False,activation='linear'))
        project_layers.append(L2NormalizeLayer())
                
        self.projector = keras.Sequential(project_layers)
        
        self.sim_coeff = 50
        self.std_coeff = 50
        self.cov_coeff = 1
        self.temperature = 0.5

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
        self.loss_func = choose_loss(loss)
        
    

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
            
            loss = self.loss_func(x, x_p,batch_size=self.batch_size,num_features=self.num_features)
            #loss = VICReg_contrastive_loss(x, x_p,s)
            
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


            return {"loss":loss} 
           
    def get_config(self):
            return {
                "projection_dim" : self.latent_dim,
                "input_shape" : self.input_shape,
                "backbone_layers" : self.backbone_layers,
                "projection_blocks" : self.projection_blocks
            }
    


# Register the model in the factory with the string name corresponding to what is in the yaml config
@ADModelFactory.register('ContrastiveEmbeddingModel')
class ContrastiveEmbeddingModel(ADModel):

    """ContrastiveEmbeddingModel class

    Args:
        ContrastiveEmbeddingModel (_type_): Base class of a AutoEncoderModel
    """

    def build_model(self, inputs_shape: tuple):
        """build model override, makes the model layer by layer

        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        self.contrastive_model = Contrastive(projection_dim=self.model_config['projection_dim'],
                                   input_shape=inputs_shape,
                                   batch_size=self.training_config['batch_size'],
                                   backbone_layers=self.model_config['backbone_layers'],
                                   projection_blocks=self.model_config['projection_blocks'],
                                   loss=self.training_config['embedding_loss'])
        
        self.vae_model = VAE(input_dim=self.model_config['projection_dim'],
                             latent_dim=self.model_config['latent_dim'],
                             encoder_layers=self.model_config['encoder_layers'],
                             decoder_layers=self.model_config['decoder_layers'])
        print(self.contrastive_model.summary())
        print(self.vae_model.summary())

    def compile_model(self):
        """compile the model generating callbacks and loss function
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """
        
        self.repr_optimizer = keras.optimizers.Adam(learning_rate=self.training_config['learning_rate'])

        # compile the tensorflow model setting the loss and metrics
        self.contrastive_model.compile(
            optimizer=self.repr_optimizer
        )
        
        self.history = { 'Embedding Loss' : [],'loss' : [], 'val_loss' : []}
        
        self.vae_callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.training_config['EarlyStopping_patience'],mode='min'),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
                mode='min'
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
        augment = Preprocessing()
        train = X_train.get_training_dataset()
        ds = (
            tf.data.Dataset.from_tensor_slices(train)
            .shuffle(self.training_config['batch_size'])
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )

        for epoch in tqdm(range(0, self.training_config['constrastive_epochs'],1)):
            running_loss = 0
            ibatch = 0
            for train_x,train_x_p in ds:
                ibatch += 1

                loss = self.contrastive_model.train_step((train_x,train_x_p))
                
                running_loss += loss["loss"]
                
            self.history['loss'].append(running_loss /ibatch)

            
            print('Embedding Epoch: {}, total loss: {}'
                .format(epoch, running_loss/ibatch))
            self.history['loss'].append(running_loss)
        ds = (
            tf.data.Dataset.from_tensor_slices(train)
            .shuffle(self.training_config['batch_size'])
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )
        
        train_size = int(len(train) * (1 - self.training_config['validation_split']) / self.training_config['batch_size'])
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)
        
        callbacks = tf.keras.callbacks.CallbackList(self.vae_callbacks, add_history=True, model=self.vae_model)
        logs = {'val_loss' : 0}
        callbacks.on_train_begin(logs=logs)
        
        for epoch in range(1, self.training_config['epochs'] + 1):
            start_time = time.time()
            callbacks.on_epoch_begin(epoch, logs=logs)
            losses = []
            ibatch = 0
            loss = tf.keras.metrics.Mean()
            for train_x in train_ds:
                latent_x = self.contrastive_model.backbone(train_x)
                ibatch += 1
                callbacks.on_train_batch_begin(ibatch, logs=logs)
                self.vae_model.train_step(latent_x)
                callbacks.on_train_batch_end(ibatch, logs=logs)
                loss(self.vae_model.compute_loss(latent_x))
            self.history['loss'].append(loss.result())
            end_time = time.time()

            itest_batch = 0
            val_loss = tf.keras.metrics.Mean()
            for test_x in val_ds:
                itest_batch += 1
                latent_test = self.contrastive_model.backbone(test_x)
                callbacks.on_test_batch_begin(itest_batch, logs=logs)
                val_loss(self.vae_model.compute_loss(latent_test))
                callbacks.on_test_batch_end(itest_batch, logs=logs)
            elbo = val_loss.result()
            
            print('Epoch: {}, Test set loss: {}, time elapse for current epoch: {}, current lr: {}'
            .format(epoch, elbo, end_time - start_time, self.vae_optimizer.learning_rate.numpy()))
            
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
        x_latent = self.contrastive_model.backbone(x)
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
        export_path = os.path.join(out_dir, "model/contrastive_saved_model.keras")
        self.contrastive_model.save(export_path)
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
        self.contrastive_model = load_model(f"{out_dir}/model/contrastive_saved_model.keras")
        self.vae_model = load_model(f"{out_dir}/model/vae_saved_model.keras")