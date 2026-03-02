"""Transformer Contrastive Embedding Model

Uses Transformer encoder layers as the backbone for contrastive learning,
as implemented in the paper: https://arxiv.org/pdf/2502.15926

Written 03/02/2026 cebrown@cern.ch
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
from data.EOSdataset import DataSet

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import keras
from keras.models import load_model
from keras.layers import Dense, BatchNormalization, ReLU, LayerNormalization, MultiHeadAttention, Add
from keras import layers

import tensorflow as tf

from sklearn.metrics import mean_absolute_error, pairwise_distances

from model.VariationalAutoEncoderModel import VAE

from tqdm import tqdm

from model.ContrastiveEmbeddingModel import (
    SimCLR_contrastive_loss,
    supervised_SimCLR_contrastive_loss,
    SimCLRLoss,
    VICReg_contrastive_loss,
    choose_loss,
    L2NormalizeLayer,
    PerObjectMaskingLayer,
    PhiRotationLayer,
    Preprocessing,
)


class TransformerEmbeddingBlock(keras.layers.Layer):
    """Transformer encoder block for particle embeddings.
    
    Processes unordered sets of particles using self-attention,
    treating each particle as a token.
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate
        )
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim, activation='linear'),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        attn_output = self.attn(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


class ParticleTokenizer(keras.layers.Layer):
    """Tokenizes particle physics data into separate particles for transformer input.
    
    Converts flat feature vector into shape (num_particles, features_per_particle)
    for transformer processing.
    
    Input order: MET (3) + electrons (12) + muons (12) + jets (30) = 57 features
    """
    
    def __init__(self, num_jets=10, num_muons=4, num_electrons=4, 
                 features_per_particle=3, include_met=True, **kwargs):
        super().__init__(**kwargs)
        self.num_jets = num_jets
        self.num_muons = num_muons
        self.num_electrons = num_electrons
        self.features_per_particle = features_per_particle
        self.include_met = include_met
        self.total_particles = num_jets + num_muons + num_electrons
        self.met_features = 3 if include_met else 0
        
    def call(self, inputs):
        # inputs shape: (batch, total_features)
        # Default: MET (3) + 4*3 (electrons) + 4*3 (muons) + 10*3 (jets) = 57 features
        
        offset = 0
        
        if self.include_met:
            met = inputs[:, :self.met_features]
            offset = self.met_features
        else:
            met = None
        
        electron_features = self.num_electrons * self.features_per_particle
        electrons = inputs[:, offset:offset + electron_features]
        offset += electron_features
        
        muon_features = self.num_muons * self.features_per_particle
        muons = inputs[:, offset:offset + muon_features]
        offset += muon_features
        
        jet_features = self.num_jets * self.features_per_particle
        jets = inputs[:, offset:offset + jet_features]
        
        electrons = tf.reshape(electrons, (-1, self.num_electrons, self.features_per_particle))
        muons = tf.reshape(muons, (-1, self.num_muons, self.features_per_particle))
        jets = tf.reshape(jets, (-1, self.num_jets, self.features_per_particle))
        
        particles = tf.concat([jets, muons, electrons], axis=1)
        
        return particles, met
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_jets': self.num_jets,
            'num_muons': self.num_muons,
            'num_electrons': self.num_electrons,
            'features_per_particle': self.features_per_particle,
        })
        return config


class TokenProjection(keras.layers.Layer):
    """Projects particle features to embedding dimension for transformer."""
    
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.projection = Dense(self.embed_dim, activation='relu')
        
    def call(self, inputs):
        return self.projection(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})
        return config


class LearnablePositionalEmbedding(keras.layers.Layer):
    """Learnable positional embeddings for particles.
    
    Since particles are unordered, we use a learned positional embedding
    that can be trained to find optimal ordering.
    """
    
    def __init__(self, num_positions, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, self.num_positions, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, inputs):
        return inputs + self.pos_embedding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_positions': self.num_positions,
            'embed_dim': self.embed_dim,
        })
        return config


class MeanPooling(keras.layers.Layer):
    """Mean pooling over particle dimension to get event-level embedding."""
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)


@keras.saving.register_keras_serializable(package="TransformerContrastive")
class TransformerContrastive(keras.Model):
    """Transformer-based contrastive learning model.
    
    Uses transformer encoder architecture as backbone for learning
    particle physics event embeddings via contrastive learning.
    """
    
    def __init__(self, projection_dim, input_shape, batch_size,
                 num_heads=4, num_layers=2, ff_dim=64, embed_dim=32,
                 dropout_rate=0.1,
                 projection_blocks=3,
                 loss='SimCLR',
                 num_jets=10, num_muons=4, num_electrons=4,
                 use_positional_embeddings=True,
                 **kwargs):
        super(TransformerContrastive, self).__init__()
        self.latent_dim = projection_dim
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_features = input_shape
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.projection_blocks = projection_blocks
        self.num_jets = num_jets
        self.num_muons = num_muons
        self.num_electrons = num_electrons
        self.total_particles = num_jets + num_muons + num_electrons
        self.use_positional_embeddings = use_positional_embeddings
        
        self.tokenizer = ParticleTokenizer(
            num_jets=num_jets,
            num_muons=num_muons,
            num_electrons=num_electrons,
            features_per_particle=3
        )
        
        self.token_projection = TokenProjection(embed_dim=embed_dim)
        
        if use_positional_embeddings:
            self.pos_embedding = LearnablePositionalEmbedding(
                num_positions=self.total_particles,
                embed_dim=embed_dim
            )
        
        self.transformer_blocks = [
            TransformerEmbeddingBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        
        self.pooling = MeanPooling()
        
        self.backbone_projection = Dense(projection_dim, activation='linear')
        
        project_layers = []
        for i in range(projection_blocks):
            project_layers.append(Dense(self.num_features, activation='linear'))
            project_layers.append(ReLU())
        project_layers.append(Dense(self.num_features, use_bias=False, activation='linear'))
        project_layers.append(L2NormalizeLayer())
        
        self.projector = keras.Sequential(project_layers)
        
        self.loss = loss
        self.loss_func = choose_loss(self.loss)
        
        self.temperature = 0.5

    def call(self, inputs, training=None):
        # Tokenize: (batch, features) -> (batch, num_particles, features_per_particle), met
        particles, met = self.tokenizer(inputs)
        
        # Project to embedding dimension: (batch, num_particles, embed_dim)
        embeddings = self.token_projection(particles)
        
        # Add positional embeddings
        if self.use_positional_embeddings:
            embeddings = self.pos_embedding(embeddings)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            embeddings = block(embeddings, training=training)
        
        # Pool to get event-level embedding: (batch, embed_dim)
        pooled = self.pooling(embeddings)
        
        # Project to projection dimension: (batch, projection_dim)
        output = self.backbone_projection(pooled)
        
        return output
    
    @tf.function
    def train_step(self, x_in, y):
        """Executes one training step."""
        x, x_p = x_in
        
        with tf.GradientTape() as tape:
            x_emb = self.call(x, training=True)
            x_p_emb = self.call(x_p, training=True)
            
            x_proj = self.projector(x_emb, training=True)
            x_p_proj = self.projector(x_p_emb, training=True)
            
            loss = self.loss_func(
                [x_proj, x_p_proj], 
                labels=y, 
                batch_size=self.batch_size,
                num_features=self.num_features
            )
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": loss}
    
    def get_config(self):
        return {
            "projection_dim": self.latent_dim,
            "input_shape": self.input_shape,
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "embed_dim": self.embed_dim,
            "dropout_rate": self.dropout_rate,
            "projection_blocks": self.projection_blocks,
            "loss": self.loss,
            "num_jets": self.num_jets,
            "num_muons": self.num_muons,
            "num_electrons": self.num_electrons,
            "use_positional_embeddings": self.use_positional_embeddings,
        }


# Register the model in the factory
@ADModelFactory.register('TransformerContrastiveEmbeddingModel')
class TransformerContrastiveEmbeddingModel(ADModel):
    """Transformer Contrastive Embedding Model class.
    
    Uses Transformer encoder layers as backbone for contrastive learning,
    following the approach in https://arxiv.org/pdf/2502.15926
    
    Args:
        ADModel: Base class
    """

    def build_model(self, inputs_shape: tuple):
        """Build model override, makes the model layer by layer
        
        Args:
            inputs_shape (tuple): Shape of the input
        """
        
        self.transformer_model = TransformerContrastive(
            projection_dim=self.model_config['projection_dim'],
            input_shape=inputs_shape,
            batch_size=self.training_config['batch_size'],
            num_heads=self.model_config.get('num_heads', 4),
            num_layers=self.model_config.get('num_layers', 2),
            ff_dim=self.model_config.get('ff_dim', 64),
            embed_dim=self.model_config.get('embed_dim', 32),
            dropout_rate=self.model_config.get('dropout_rate', 0.1),
            projection_blocks=self.model_config.get('projection_blocks', 3),
            loss=self.training_config['embedding_loss'],
            num_jets=self.model_config.get('num_jets', 10),
            num_muons=self.model_config.get('num_muons', 4),
            num_electrons=self.model_config.get('num_electrons', 4),
            use_positional_embeddings=self.model_config.get('use_positional_embeddings', True)
        )
        
        dummy_input = tf.zeros((1, inputs_shape))
        _ = self.transformer_model(dummy_input, training=False)
        
        self.vae_model = VAE(
            input_dim=self.model_config['projection_dim'],
            latent_dim=self.model_config['latent_dim'],
            encoder_layers=self.model_config['encoder_layers'],
            decoder_layers=self.model_config['decoder_layers']
        )
        
        print(self.transformer_model.summary())
        print(self.vae_model.summary())

    def compile_model(self, input_length):
        """Compile the model generating callbacks and loss function
        
        Args:
            num_samples (int): Number of samples in the training set used for scheduling
        """
        
        scheduler = keras.optimizers.schedules.CosineDecay(
            self.training_config['emb_learning_rate'],
            int(input_length / (self.training_config['batch_size'] * self.training_config['contrastive_epochs'])),
            alpha=0.0,
            name="CosineDecay",
            warmup_target=None,
            warmup_steps=0,
        )
        
        self.repr_optimizer = keras.optimizers.AdamW(
            learning_rate=scheduler,
            weight_decay=self.training_config['emb_learning_rate_decay']
        )
        
        self.transformer_model.compile(
            optimizer=self.repr_optimizer
        )
        
        self.history = {'Embedding Loss': [], 'loss': [], 'val_loss': []}
        
        self.vae_callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=self.training_config['EarlyStopping_patience'],
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['ReduceLROnPlateau_factor'],
                patience=self.training_config['ReduceLROnPlateau_patience'],
                min_lr=self.training_config['ReduceLROnPlateau_min_lr'],
                mode='min'
            ),
        ]
        
        self.vae_optimizer = keras.optimizers.Adam(
            learning_rate=self.training_config['learning_rate']
        )
        
        self.vae_model.compile(
            optimizer=self.vae_optimizer
        )

    def fit(self, X_train: pd.DataFrame, training_columns: list):
        """Fit the model to the training dataset
        
        Args:
            X_train (npt.NDArray[np.float64]): X train dataset
            training_columns (list): List of column names for training
        """
        keras.config.disable_traceback_filtering()
        augment = Preprocessing()
        train = X_train
        
        ds = (
            tf.data.Dataset.from_tensor_slices((train[training_columns], train['event_label']))
            .shuffle(self.training_config['batch_size'])
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )
        
        contrastive_epochs = self.training_config.get('contrastive_epochs', 
                                                       self.training_config.get('constrastive_epochs', 2))
        
        for epoch in tqdm(range(0, contrastive_epochs, 1), desc="Contrastive Training"):
            running_loss = 0
            ibatch = 0
            
            for train_x, train_x_p, y in ds:
                loss = self.transformer_model.train_step((train_x, train_x_p), y)
                running_loss += loss["loss"]
                ibatch += 1
            
            print('Embedding Epoch: {}, total loss: {}'.format(epoch, running_loss / ibatch))
            self.history['loss'].append(running_loss / ibatch)
        
        ds = (
            tf.data.Dataset.from_tensor_slices(train[training_columns])
            .shuffle(self.training_config['batch_size'])
            .batch(self.training_config['batch_size'])
            .prefetch(tf.data.AUTOTUNE)
        )
        
        train_size = int(len(train) * (1 - self.training_config['validation_split']) / self.training_config['batch_size'])
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)
        
        callbacks = tf.keras.callbacks.CallbackList(
            self.vae_callbacks, 
            add_history=True, 
            model=self.vae_model
        )
        logs = {'val_loss': 0}
        callbacks.on_train_begin(logs=logs)
        
        for epoch in range(1, self.training_config['epochs'] + 1):
            start_time = time.time()
            callbacks.on_epoch_begin(epoch, logs=logs)
            
            ibatch = 0
            loss = tf.keras.metrics.Mean()
            
            for train_x in train_ds:
                latent_x = self.transformer_model.call(train_x, training=False)
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
                latent_test = self.transformer_model.call(test_x, training=False)
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

    def predict(self, X_test, training_columns, return_score=True) -> npt.NDArray[np.float64]:
        """Predict method for model
        
        Args:
            X_test: Input X test
            training_columns: List of column names
            return_score: Whether to return anomaly score
            
        Returns:
            Anomaly scores or reconstructions
        """
        
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test[training_columns].to_numpy()
        else:
            test = X_test
        
        x = tf.cast(test, tf.float32)
        x_latent = self.transformer_model.call(x, training=False)
        mean, logvar = self.vae_model.encode(x_latent)
        mu2 = np.linalg.norm(mean, axis=1)
        z = self.vae_model.reparameterize(mean, logvar)
        x_logit = self.vae_model.decode(z)
        ad_scores = tf.keras.losses.mse(x_logit, x_latent)
        ad_scores = ad_scores.numpy()
        ad_scores = (ad_scores - np.min(ad_scores)) / (np.max(ad_scores) - np.min(ad_scores))
        
        if return_score:
            return ad_scores
        else:
            return x_logit

    def distance(self, test, training_columns):
        """Compute distance between input and reconstruction"""
        x_hat = self.predict(test, training_columns, return_score=False)
        x_latent = self.transformer_model.call(test)
        return pairwise_distances(x_latent, x_hat)

    def encoder_predict(self, X_test, training_columns) -> npt.NDArray[np.float64]:
        """Get embeddings from the transformer backbone"""
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test[training_columns].to_numpy()
        else:
            test = X_test
        
        x = tf.cast(test, tf.float32)
        latent = self.transformer_model.call(x, training=False)
        return latent

    def var_predict(self, X_test, training_columns) -> npt.NDArray[np.float64]:
        """Get mean and logvar from VAE encoder"""
        if isinstance(X_test, DataSet):
            test = X_test.get_training_dataset()
        elif isinstance(X_test, pd.DataFrame):
            test = X_test[training_columns].to_numpy()
        else:
            test = X_test
        
        x = tf.cast(test, tf.float32)
        x_latent = self.transformer_model.call(x, training=False)
        mean, logvar = self.vae_model.encode(x_latent)
        return mean, logvar

    @ADModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the model file
        
        Args:
            out_dir (str, optional): Where to save it if not in the output_directory
        """
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        
        export_path = os.path.join(out_dir, "model/transformer_contrastive_saved_model.keras")
        self.transformer_model.save(export_path)
        print(f"Model saved to {export_path}")
        
        export_path = os.path.join(out_dir, "model/vae_saved_model.keras")
        self.vae_model.save(export_path)
        print(f"Model saved to {export_path}")

    @ADModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the model file
        
        Args:
            out_dir (str, optional): Where to load it if not in the output_directory
        """
        self.transformer_model = load_model(
            f"{out_dir}/model/transformer_contrastive_saved_model.keras",
            compile=False
        )
        self.vae_model = load_model(f"{out_dir}/model/vae_saved_model.keras")
