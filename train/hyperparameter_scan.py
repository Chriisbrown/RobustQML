#!/usr/bin/env python3
"""Hyperparameter scan script for model optimization.

Optimizes hyperparameters to maximize AUC ROC for minbias vs HH_4b classification.

Usage:
    python train/hyperparameter_scan.py --config model/configs/AutoEncoderModel.yaml --output output/hyperopt
    python train/hyperparameter_scan.py --config model/configs/VariationalAutoEncoderModel.yaml --output output/hyperopt_vae
"""

import os
import sys
import shutil
import yaml
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from itertools import product

from model.common import fromDict
from data.EOSdataset import DataSet
from sklearn.metrics import roc_curve, auc


def compute_auc_roc(model, background_data, signal_data, training_columns):
    """Compute AUC ROC for minbias (background) vs HH_4b (signal)."""
    print(f"Background type: {type(background_data)}, Signal type: {type(signal_data)}")
    print(f"Training columns: {training_columns}")
    if isinstance(background_data, pd.DataFrame):
        print("Using DataFrame path")
        background_outputs = model.predict(background_data[training_columns], training_columns)
        signal_outputs = model.predict(signal_data[training_columns], training_columns)
    else:
        print("Using DataSet path")
        background_outputs = model.predict(background_data, training_columns)
        signal_outputs = model.predict(signal_data, training_columns)
    
    true_labels = np.concatenate([
        np.zeros(background_outputs.shape[0]),
        np.ones(signal_outputs.shape[0])
    ])
    predictions = np.concatenate([background_outputs, signal_outputs])
    
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    return auc(fpr, tpr)


def load_evaluation_data(normalise=True, max_events=-1):
    """Load background (minbias) and signal (HH_4b) data for evaluation."""
    background = DataSet.fromH5('/eos/user/c/cebrown/RobustQML/training_data/minbias/test')
    signal = DataSet.fromH5('/eos/user/c/cebrown/RobustQML/training_data/HH_4b/test')
    
    if normalise:
        background.normalise()
        signal.normalise()
    else:
        background.max_number_of_jets = 10
        background.max_number_of_objects = 4
        background.generate_feature_lists()
        signal.max_number_of_jets = 10
        signal.max_number_of_objects = 4
        signal.generate_feature_lists()
    
    training_columns = background.training_columns
    
    if max_events > 0:
        background_df = background.data_frame.sample(n=max_events)
        signal_df = signal.data_frame.sample(n=max_events)
        return background_df, signal_df, training_columns
    
    return background, signal, training_columns


def load_training_data(normalise=True, max_events=-1):
    """Load training data (minbias only for autoencoder-like models)."""
    labels = {"minbias": 1}
    dataset_list = []
    
    for dataset_name, label in labels.items():
        data = DataSet.fromH5(f'/eos/user/c/cebrown/RobustQML/training_data/{dataset_name}/train/')
        if normalise:
            data.normalise()
        else:
            data.max_number_of_jets = 5
            data.max_number_of_objects = 2
            data.generate_feature_lists()
        data.set_label(label)
        dataset_list.append(data)
    
    full_data_frame = pd.concat([d.data_frame for d in dataset_list])
    full_data_frame = full_data_frame.sample(frac=1)
    
    if max_events > 0:
        full_data_frame = full_data_frame.sample(n=max_events)
    
    return full_data_frame


def generate_hyperparameter_combinations(param_grid):
    """Generate all combinations of hyperparameters from param_grid dict."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))


def update_nested_dict(base_dict, updates):
    """Update nested dictionary with new values."""
    result = base_dict.copy()
    for key, value in updates.items():
        if '.' in key:
            parts = key.split('.')
            d = result
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        else:
            result[key] = value
    return result


HYPERPARAM_GRIDS = {
    'VICRegModel': {
        'training_config.epochs': [30, 50, 80],
        'training_config.batch_size': [256, 512, 1024],
        'training_config.learning_rate': [0.0001, 0.0005, 0.001],
    },
    'AutoEncoderModel': {
        'model_config.encoder_layers': [[32, 16], [64, 32], [64, 32, 16], [128, 64, 32], [256, 128, 64]],
        'model_config.decoder_layers': [[16, 32], [32, 64], [16, 32, 64], [32, 64, 128], [64, 128, 256]],
        'model_config.latent_dim': [4, 8, 16, 32],
        'training_config.epochs': [30, 50, 80],
        'training_config.batch_size': [256, 512, 1024, 2048],
        'training_config.learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    },
    'VariationalAutoEncoderModel': {
        'model_config.encoder_layers': [[32, 16, 8], [64, 32, 16], [128, 64, 32], [64, 32], [32, 16]],
        'model_config.decoder_layers': [[8, 16, 32], [16, 32, 64], [32, 64, 128], [32, 64], [16, 32]],
        'model_config.latent_dim': [4, 8, 16, 32],
        'training_config.epochs': [30, 50, 80],
        'training_config.batch_size': [256, 512, 1024],
        'training_config.learning_rate': [0.00005, 0.0001, 0.0005, 0.001],
    },
    'ContrastiveEmbeddingModel': {
        'model_config.backbone_layers': [[32, 16], [64, 32], [128, 64, 32], [64, 32, 16]],
        'model_config.projection_dim': [4, 8, 16, 32],
        'model_config.latent_dim': [4, 8, 16],
        'training_config.epochs': [20, 40, 60],
        'training_config.batch_size': [32, 64, 128],
        'training_config.learning_rate': [0.0001, 0.0005, 0.001],
    },
    'IsolationTreeModel': {
        'model_config.n_estimators': [50, 100, 200, 300],
        'model_config.max_depth': [-1, 8, 16, 32],
        'model_config.min_examples': [3, 5, 10, 20],
        'model_config.split_axis': ['AXIS_ALIGNED', 'SPARSE_OBLIQUE'],
        'model_config.sparse_oblique_weights': ['CONTINUOUS', 'BINARY'],
        'model_config.sparse_oblique_projection_density_factor': [1.0, 3.0, 5.0],
    },
    'VICRegAutoEncoderModel': {
        'model_config.encoder_layers': [[32, 16], [64, 32, 16], [128, 64, 32]],
        'model_config.decoder_layers': [[16, 32], [16, 32, 64], [32, 64, 128]],
        'model_config.latent_dim': [8, 16, 32],
        'training_config.epochs': [30, 50, 80],
        'training_config.batch_size': [256, 512, 1024],
        'training_config.learning_rate': [0.0001, 0.0005, 0.001],
    },
    'PennyLaneAutoEncoderModel': {
        'model_config.latent_dim': [4, 8, 16],
        'model_config.num_layers': [2, 3, 4, 5],
        'training_config.epochs': [30, 50, 80],
        'training_config.batch_size': [256, 512],
    },
    'AXOVariationalAutoEncoderModel': {
        'model_config.encoder_layers': [[32, 16], [64, 32, 16], [128, 64, 32]],
        'model_config.decoder_layers': [[16, 32], [16, 32, 64], [32, 64, 128]],
        'model_config.latent_dim': [8, 16, 32],
        'training_config.epochs': [30, 50, 80],
        'training_config.batch_size': [256, 512],
        'training_config.learning_rate': [0.0001, 0.0005, 0.001],
    },
}


def get_default_param_grid(model_type):
    """Get default hyperparameter grid for a model type."""
    return HYPERPARAM_GRIDS.get(model_type, {})


def run_single_experiment(base_config_path, params, output_dir, normalise=True, max_train_events=-1, max_eval_events=-1):
    """Run a single experiment with given parameters."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_type = config['model']
    config = update_nested_dict(config, params)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        model = fromDict(config, output_dir, recreate=True)
        
        training_df = load_training_data(normalise, max_train_events)
        training_columns = training_df.columns.tolist()
        
        background_data, signal_data, eval_columns = load_evaluation_data(normalise, max_eval_events)
        
        if not isinstance(background_data, pd.DataFrame):
            background_df = background_data.data_frame
            signal_df = signal_data.data_frame
        else:
            background_df = background_data
            signal_df = signal_data
        
        common_cols = [col for col in training_columns if col in background_df.columns and col in signal_df.columns]
        
        if len(common_cols) != len(training_columns):
            print(f"Warning: Using {len(common_cols)} common columns (training has {len(training_columns)})")
            training_columns = common_cols
            training_df = training_df[training_columns]
        
        if 'index' in training_columns:
            training_columns.remove('index')
            training_df = training_df[training_columns]
            print("Removed 'index' column from training")
        
        model.build_model(len(training_columns))
        model.compile_model(len(training_df))
        model.fit(training_df, training_columns)
        model.save()
        
        auc_score = compute_auc_roc(model, background_df, signal_df, training_columns)
        
        return {
            'params': params,
            'auc_roc': auc_score,
            'output_dir': output_dir,
            'success': True
        }
    except Exception as e:
        import traceback
        return {
            'params': params,
            'auc_roc': 0.0,
            'output_dir': output_dir,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def run_hyperparameter_scan(
    base_config_path,
    output_dir,
    param_grid=None,
    max_trials=None,
    normalise=True,
    max_train_events=-1,
    max_eval_events=-1,
    keep_all_models=False
):
    """Run hyperparameter optimization."""
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    model_type = base_config['model']
    print(f"Optimizing {model_type}")
    
    if param_grid is None:
        param_grid = get_default_param_grid(model_type)
    
    if not param_grid:
        print(f"No hyperparameter grid defined for {model_type}. Using default config.")
        param_grid = {}
    
    combinations = list(generate_hyperparameter_combinations(param_grid))
    print(f"Total hyperparameter combinations: {len(combinations)}")
    
    if max_trials is not None and max_trials < len(combinations):
        np.random.seed(42)
        indices = np.random.choice(len(combinations), max_trials, replace=False)
        combinations = [combinations[i] for i in sorted(indices)]
        print(f"Running {max_trials} random trials")
    
    results = []
    best_auc = 0.0
    best_result = None
    
    scan_dir = os.path.join(output_dir, 'scan_results')
    os.makedirs(scan_dir, exist_ok=True)
    
    for i, params in enumerate(combinations):
        print(f"\n{'='*60}")
        print(f"Trial {i+1}/{len(combinations)}")
        print(f"Parameters: {params}")
        
        trial_dir = os.path.join(scan_dir, f'trial_{i:04d}')
        
        result = run_single_experiment(
            base_config_path,
            params,
            trial_dir,
            normalise,
            max_train_events,
            max_eval_events
        )
        
        results.append(result)
        
        if result['success']:
            print(f"AUC ROC: {result['auc_roc']:.4f}")
            if result['auc_roc'] > best_auc:
                best_auc = result['auc_roc']
                best_result = result.copy()
                best_result['trial_dir'] = trial_dir
                print(f"New best model!")
                
                best_dir = os.path.join(output_dir, 'best_model')
                if os.path.exists(best_dir):
                    shutil.rmtree(best_dir)
                shutil.copytree(trial_dir, best_dir)
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(result['traceback'])
        
        if not keep_all_models and result['success']:
            if trial_dir != best_result.get('trial_dir'):
                if os.path.exists(trial_dir):
                    shutil.rmtree(trial_dir)
    
    results_summary = {
        'model_type': model_type,
        'base_config': base_config_path,
        'total_trials': len(results),
        'successful_trials': sum(1 for r in results if r['success']),
        'best_auc_roc': best_auc,
        'best_params': best_result['params'] if best_result else None,
        'best_model_dir': best_result['output_dir'] if best_result else None,
        'all_results': [
            {
                'params': r['params'],
                'auc_roc': r['auc_roc'],
                'success': r['success'],
                'output_dir': r['output_dir']
            }
            for r in results
        ]
    }
    
    with open(os.path.join(scan_dir, 'results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SCAN COMPLETE")
    print(f"Best AUC ROC: {best_auc:.4f}")
    print(f"Best Parameters: {best_result['params'] if best_result else 'None'}")
    print(f"Best model saved to: {os.path.join(output_dir, 'best_model')}")
    print(f"Full results saved to: {scan_dir}/results.json")
    
    return results_summary


if __name__ == "__main__":
    parser = ArgumentParser(description='Hyperparameter scan for model optimization')
    parser.add_argument('-c', '--config', required=True, help='Base YAML config for model')
    parser.add_argument('-o', '--output', default='output/hyperparameter_scan', help='Output directory')
    parser.add_argument('-n', '--normalise', default='True', help='Normalise input data?')
    parser.add_argument('--max-trials', type=int, default=None, help='Maximum number of trials (random subset if fewer combinations)')
    parser.add_argument('--max-train-events', type=int, default=-1, help='Max training events (for quick testing)')
    parser.add_argument('--max-eval-events', type=int, default=-1, help='Max evaluation events (for quick testing)')
    parser.add_argument('--keep-all', action='store_true', help='Keep all trained models (default: only best)')
    parser.add_argument('--param', action='append', help='Override parameter (format: key=value)')
    
    args = parser.parse_args()
    
    normalise = args.normalise == 'True'
    
    param_grid = get_default_param_grid(yaml.safe_load(open(args.config))['model'])
    
    if args.param:
        overrides = {}
        for p in args.param:
            if '=' in p:
                key, value = p.split('=', 1)
                try:
                    value = eval(value)
                except:
                    pass
                overrides[key] = value
        
        for key, value in overrides.items():
            if key in param_grid:
                if isinstance(param_grid[key], list):
                    param_grid[key] = [v for v in param_grid[key] if v == value]
                else:
                    param_grid[key] = [value]
    
    run_hyperparameter_scan(
        args.config,
        args.output,
        param_grid=param_grid,
        max_trials=args.max_trials,
        normalise=normalise,
        max_train_events=args.max_train_events,
        max_eval_events=args.max_eval_events,
        keep_all_models=args.keep_all
    )
