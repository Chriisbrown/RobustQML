"""GPU configuration utilities for TensorFlow models.

This module provides functions to detect and configure GPU usage for TensorFlow.
Falls back gracefully to CPU if no GPU is available.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_gpu_memory_growth():
    """Enable memory growth for GPU devices to avoid OOM errors.
    
    This allows GPU memory to be allocated incrementally rather than
    all at once.
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU(s) found: {len(gpus)}. GPU training enabled.")
                return True
            except RuntimeError as e:
                print(f"Error setting GPU memory growth: {e}")
                return False
        else:
            print("No GPU found. Using CPU for training.")
            return False
            
    except ImportError:
        print("TensorFlow not available. Cannot configure GPU.")
        return False


def get_device_name():
    """Get the name of the device being used for computation.
    
    Returns:
        str: 'GPU' if GPU is available, 'CPU' otherwise
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            return 'GPU'
        return 'CPU'
    except ImportError:
        return 'CPU'


def is_gpu_available():
    """Check if a GPU is available for TensorFlow.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus) > 0
    except ImportError:
        return False
