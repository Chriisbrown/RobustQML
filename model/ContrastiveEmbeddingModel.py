"""AutoEncoder model child class

Written 02/01/2026 cebrown@cern.ch
"""

import json
import os
import time 
import random
import sys

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

from sklearn.metrics import mean_absolute_error,pairwise_distances

from model.VariationalAutoEncoderModel import VAE

from tqdm import tqdm

def SimCLR_contrastive_loss(zs, temperature=0.5,**kwargs):
        # First: Concatenate both batches of embeddings (positive pairs)
        z = tf.concat([zs[0],zs[1]], axis=0)  # shape: (2N, D), where N is batch size

        # Cosine similarity matrix between all embeddings (assumes z is L2-normalized)
        sim = tf.matmul(z, z, transpose_b=True)  # shape: (2N, 2N), sim[i][j] = similarity between sample i and j
        sim /= temperature  # scale similarities by temperature (sharpening)

        # Create some positive/negative pair labels — position i matches with i + N ( same image, different view)
        batch_size = tf.shape(zs)[1]
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
    
def supervised_SimCLR_contrastive_loss(zs, labels, temperature=0.07,**kwargs):
        labels = tf.reshape(labels, [-1, 1])
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
        n_views = tf.shape(zs)[0]
                
        # First: Concatenate batches of embeddings 
        contrast_feature = tf.concat(tf.unstack(zs, axis=1), axis=0)
        anchor_feature = contrast_feature
        anchor_count = n_views
                
        # Cosine similarity matrix between all embeddings (assumes z is L2-normalized)
        sim = tf.matmul(anchor_feature, contrast_feature, transpose_b=True)  # shape: (2N, 2N), sim[i][j] = similarity between sample i and j
        sim /= temperature  # scale similarities by temperature (sharpening)
                
        logits_max = tf.reduce_max(sim, axis=1, keepdims=True)
        sim = sim - tf.stop_gradient(logits_max)


        # Create some positive/negative pair labels — position i matches with i + N ( same image, different view)
        batch_size = tf.shape(zs)[1]
        labels = tf.range(batch_size)
        labels = tf.concat([labels, labels], axis=0)  # shape: (2N,)

        mask = tf.ensure_shape(mask, [None, None])
        mask = tf.tile(mask, [anchor_count,n_views])

        logits_mask = tf.ones_like(mask)
        diag_indices = tf.range(batch_size * anchor_count)
        diag_indices = tf.stack([diag_indices, diag_indices], axis=1)
        logits_mask = tf.tensor_scatter_nd_update(
            logits_mask, diag_indices, tf.zeros(batch_size * anchor_count)
        )
        

        mask = mask * logits_mask

        # Step 6: Compute the famous NT-Xent loss
        exp_logits = tf.exp(sim) * logits_mask # exp(similarity of positive pairs)
        log_prob = sim - tf.math.log( tf.reduce_sum(exp_logits, axis=1))  # sum over all other similarities for each sample

        mask_sum = tf.reduce_sum(mask, axis=1)
        mean_log_prob_pos = - tf.reduce_sum(mask * log_prob, axis=1) / mask_sum
        
        loss = tf.reshape(mean_log_prob_pos, [anchor_count, batch_size])
        # Step 7: Return average loss over the batch
        return tf.reduce_mean(loss)
    
    
def SimCLRLoss(zs, labels, temperature=0.07,**kwargs):
        # def SimCLRLoss(features, labels, temperature = 0.07):
        '''
        Computes SimCLRLoss as defined in https://arxiv.org/pdf/2004.11362.pdf
        '''

        # Generates mask indicating what samples are considered pos/neg
        labels = tf.reshape(labels, [-1, 1])
        positive_mask = tf.equal(labels, tf.transpose(labels))
        negative_mask = tf.logical_not(positive_mask)
        positive_mask = tf.cast(positive_mask, dtype=tf.float32)
        negative_mask = tf.cast(negative_mask, dtype=tf.float32)

        # Computes dp between pairs
        contrast_feature = tf.concat(tf.unstack(zs, axis=1), axis=0)
        logits = tf.matmul(zs, zs, transpose_b=True)
        temperature = tf.cast(temperature, tf.float32)
        logits = logits / temperature

        # Subtract largest |logits| elt for numerical stability
        # Simply for numerical precision -> stop gradient
        max_logit = tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
        logits = logits - max_logit

        exp_logits = tf.exp(logits)
        num_positives_per_row = tf.reduce_sum(positive_mask, axis=1)

        denominator = tf.reduce_sum(exp_logits * negative_mask, axis = 1, keepdims=True)
        denominator += tf.reduce_sum(exp_logits * positive_mask, axis = 1, keepdims=True)

        # Compute L OUTSIDE -> defined in eq 2 of paper
        log_probs = (logits - tf.math.log(denominator)) * positive_mask
        log_probs = tf.reduce_sum(log_probs, axis=1)
        log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
        loss = -log_probs * temperature 
        loss = tf.reduce_mean(loss)
        return loss
    
def VICReg_contrastive_loss(zs,batch_size,num_features,sim_coeff=50,std_coeff=50,cov_coeff=1,**kwargs):
        repr_loss = keras.losses.mean_squared_error(zs[0],zs[1])
            
        x = zs[0] - tf.reduce_mean(zs[0], axis=0, keepdims=True)
        x_p = zs[1] - tf.reduce_mean(zs[1], axis=0, keepdims=True)
            
        std_x = tf.sqrt(tf.math.reduce_variance(x, axis=0) + 0.0001)
        std_x_p = tf.sqrt(tf.math.reduce_variance(x_p, axis=0) + 0.0001)
            
        std_loss = tf.reduce_mean(tf.nn.relu(1.0 - std_x)) / 2 + tf.reduce_mean(tf.nn.relu(1.0 - std_x_p)) / 2
    
        cov_x = tf.linalg.matmul(x, x, transpose_a=True) / (batch_size - 1.0)
        cov_x_p = tf.linalg.matmul(x_p, x_p, transpose_a=True) / (batch_size - 1.0)
            
        cov_loss = (tf.reduce_sum(tf.square(off_diagonal(cov_x))) +  tf.reduce_sum(tf.square(off_diagonal(cov_x_p)))) / float(num_features)
    
        return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss


def choose_loss(choice: str):
    """Choose the aggregator keras object based on an input string."""
    if choice not in ["SimCLR", "VICReg","SupSimCLR"]:
        raise ValueError(
            choice, "Not implemented"
        )
    if choice == "SimCLR":
        return SimCLR_contrastive_loss
    elif choice == "VICReg":
        return VICReg_contrastive_loss
    elif choice == "SupSimCLR":
        return SimCLRLoss


  
class L2NormalizeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
    
class FlatMaskingLayer(keras.layers.Layer):
    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def call(self, inputs):
        mask = np.random.rand((inputs.shape[0]))
        idx = mask < self.probability
        mask[idx] = 0
        mask[~idx] = 1
        inputs = inputs * mask
        return inputs
    
class PerObjectMaskingLayer(keras.layers.Layer):
    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def call(self, inputs):
        input_shape = inputs.shape
        total_object = int(inputs.shape[0] / 3)
        inputs = tf.reshape(inputs,(total_object,3))
        mask = np.random.rand(inputs.shape[0],inputs.shape[1])
        idx = mask < self.probability
        mask[idx] = 0
        mask[~idx] = 1
        inputs = inputs * mask
        inputs = tf.reshape(inputs,input_shape)
        return inputs
    
class PhiRotationLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        
        input_copy = inputs
        total_object = int(inputs.shape[0] / 3)
        phi_columns = tf.constant([[int(i*3+2)] for i in range(total_object)])
        pt_columns = tf.constant([[int(i*3)] for i in range(total_object)])
        phi = tf.gather(input_copy,indices=phi_columns)
        pt = tf.gather(input_copy,indices=pt_columns)
        non_zeros =  tf.cast(pt != 0, tf.float32)
        rot_angle = tf.ones_like(phi) * tf.random.uniform(shape=[1], minval=-np.pi, maxval=np.pi)
        phi  = phi + rot_angle * non_zeros
        phi = tf.where(phi > np.pi, phi - 2*np.pi, phi)
        phi = tf.where(phi < -np.pi, phi + 2*np.pi, phi)
        indices = tf.expand_dims(phi_columns, axis=1)  # shape [N, 1]
        inputs = tf.tensor_scatter_nd_update(input_copy, indices, phi)
        inputs = tf.reshape(inputs,inputs.shape)
        return inputs
  
class Preprocessing(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.augment = tf.keras.Sequential([
            PerObjectMaskingLayer(0.2),
            #PhiRotationLayer(),
        ])
        
    def call(self, x,y):
        return x, self.augment(x), y

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
        self.loss=loss
        self.loss_func = choose_loss(self.loss)
    

    @tf.function
    def train_step(self, x_in,y):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        x,x_p = x_in
        with tf.GradientTape() as tape:
            x = self.projector(self.backbone(x, training=True) , training=True)
            x_p = self.projector(self.backbone(x_p, training=True) , training=True)
            
            loss = self.loss_func([x, x_p], labels=y, batch_size=self.batch_size,num_features=self.num_features)
            #loss = VICReg_contrastive_loss(x, x_p,s)
            
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


            return {"loss":loss} 
           
    def get_config(self):
            return {
                "projection_dim" : self.latent_dim,
                "input_shape" : self.input_shape,
                "backbone_layers" : self.backbone_layers,
                "projection_blocks" : self.projection_blocks,
                "batch_size" : self.batch_size,
                "loss" : self.loss,
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
        
        self.repr_optimizer = keras.optimizers.Adam(learning_rate=self.training_config['emb_learning_rate'])

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
        X_train: pd.DataFrame,
        training_columns: list
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
        train = X_train
        ds = (
            tf.data.Dataset.from_tensor_slices((train[training_columns],train['event_label']))
            .shuffle(self.training_config['batch_size'])
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )

        for epoch in tqdm(range(0, self.training_config['constrastive_epochs'],1)):
            running_loss = 0
            ibatch = 0
            for train_x,train_x_p,y in ds:
                ibatch += 1

                loss = self.contrastive_model.train_step((train_x,train_x_p),y)
                
                running_loss += loss["loss"]
                
            self.history['loss'].append(running_loss /ibatch)

            
            print('Embedding Epoch: {}, total loss: {}'
                .format(epoch, running_loss/ibatch))
            self.history['loss'].append(running_loss)
        ds = (
            tf.data.Dataset.from_tensor_slices((train[training_columns]))
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
        ad_scores = tf.keras.losses.mse(x_logit,x_latent)
        ad_scores = ad_scores._numpy()
        ad_scores = mu2
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        if return_score:
            return ad_scores
        else:
            return x_logit

    def distance(self, test):
        x_hat = self.predict(test, return_score=False)
        x_latent = self.contrastive_model.backbone(test)
        return pairwise_distances(x_latent,x_hat)
    
    
    def encoder_predict(self,X_test) -> npt.NDArray[np.float64]:
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test.to_numpy()
        else:
            test = X_test
        latent = self.contrastive_model.backbone(test)
        return latent
    
    def var_predict(self,X_test) -> npt.NDArray[np.float64]:
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test.to_numpy()
        else:
            test = X_test
        x_latent = self.contrastive_model.backbone(test)
        mean, logvar = self.vae_model.encode(x_latent)
        return mean, logvar
    
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