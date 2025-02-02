import tensorflow as tf
import numpy as np

# Auxiliary variable parameters
AUX_SCALE = tf.constant(3.0)  # X = SCALE * u scaling factor

# Prior parameters
PRIOR_VARIANCE = tf.constant(100.0)  # Prior variance for parameters

# MCMC sampling parameters (not used in log_likelihood_stable.py)
MCMC_PARAMS = {
    'M': 200,  # Total number of iterations
    'BURN_IN': 0,  # Number of burn-in iterations
    'H': 0.025,  # Step size
    'L': 50,  # Number of leapfrog steps
    'N': 128,  # Number of auxiliary variables
    'P': 1,  # Dimension of auxiliary variables
    'RHO_SIZE': 10.0,  # Rho scale factor
}

# Data generation parameters
DATA_PARAMS = {
    'T': 500,  # Number of time points
    'N_SUBJECTS': 6,  # Number of subjects
    'NUM_FEATURES': 8,  # Number of features (beta dimension)
    'MU1': 0.0,  # Mean of first component
    'MU2': 3.0,  # Mean of second component
    'LAMBDA1': 10.0,  # Precision of first component
    'LAMBDA2': 3.0,  # Precision of second component
    'W1': 0.8,  # Weight of first component
}

# Initial parameter values (not used in log_likelihood_stable.py)
INIT_PARAMS = {
    'BETA': None,  # Will be initialized randomly
    'MU1': 0.0,
    'MU2': 0.0,
    'LOG_LAMBDA1': 0.0,  # log(1.0)
    'LOG_LAMBDA2': -2.3,  # log(0.1)
    'ETA': 0.0,  # logit(0.5)
}

# Neural Network parameters
NN_PARAMS = {
    # Original parameters
    'HIDDEN_DIM': 256,  # Base hidden layer dimension (reserved for other modules)
    'CONV_DIM': 64,  # Base convolution channel number (reserved for other modules)
    'NUM_HEADS': 8,
    'KEY_DIM': 256,  # Attention key dimension (changed from 32 to 256 to match code)
    'NUM_RES_BLOCKS': 2,  # Number of residual blocks (changed from 3 to 2 to match double ResBlock in code)
    'DROPOUT_RATE': 0.1,
    'USE_BATCH_NORM': True,

    # Additional HNN specific parameters
    'HNN_HIDDEN_DIM': 512,  # HNN backbone network hidden layer dimension
    'HNN_KEY_DIM': 256,  # Cross-attention key dimension (merged with KEY_DIM or defined separately)
    'HNN_CONV_FILTERS': [128, 256],  # Convolution channel numbers for U feature extractor
    'HNN_CONV_KERNELS': [5, 3],  # Convolution kernel sizes
    'HNN_ATTENTION_DROPOUT': 0.1,  # Dropout rate for attention module (optional)
}

# Trajectory collection parameters
TRAJ_PARAMS = {
    'SAVE_INTERVAL': 1,  # Save trajectory every N iterations
    'TRAJ_LENGTH': 50,  # Number of trajectory points to save
    'SAVE_DIR': 'trajectories',  # Directory to save trajectories
    'M': 1500,  # Total number of iterations
    'num_chains': 1
}

# Training parameters
TRAIN_PARAMS = {
    'NUM_OUTER_LOOPS': 2,  # Number of outer loops
    'NUM_EPOCHS': 3000,  # Number of epochs per outer loop
    'BATCH_SIZE': 256,  # Batch size
    'INITIAL_LR': 1e-3,  # Initial learning rate
    #  decay parameters
    'LR_DECAY': {
        'DECAY_EPOCHS': 50,  # Decay learning rate every N epochs
        'DECAY_RATE': 0.83,  # Decay rate
        'STAIRCASE': True  # Whether to use staircase decay
    }
}

# HNN mode selection
USE_HNN = False  # True for HNN mode, False for traditional mode. Use False during training for data collection

# HNN weights path (without file extension)
HNN_WEIGHTS_PATH = 'ckpt/weights_hamiltonian_20250201_212453'  # 128 Hidden

# NUTS parameters
NUTS_PARAMS = {
    'hnn_threshold': 20.0,  # HNN error threshold
    'leapfrog_threshold': 1000.0,  # Leapfrog error threshold
    'n_cooldown': 20,  # Cooldown period
    'total_samples': 10,  # Total number of samples
    'burn_in': 5,  # Number of burn-in samples
    'nuts_step_size': 0.025,  # Initial step size
    'rho_size': 10.0,  # Rho scale factor
    'traditional_only': False,  # Whether to use only traditional leapfrog
    'max_depth': 1000,  # Maximum tree depth
    'hnn_error_threshold': 20.0,  # Same as hnn_threshold, for compatibility
    'stop_criteria_threshold': 50000.0
}


def initialize_theta(seed=50):
    """
    Initialize theta according to INIT_PARAMS

    Returns:
        theta: tensor of shape [num_features + 5]
    """
    tf.random.set_seed(seed)
    beta = tf.random.normal([DATA_PARAMS['NUM_FEATURES']], seed=seed)
    other_params = tf.constant([
        INIT_PARAMS['MU1'],
        INIT_PARAMS['MU2'],
        INIT_PARAMS['LOG_LAMBDA1'],
        INIT_PARAMS['LOG_LAMBDA2'],
        INIT_PARAMS['ETA']
    ], dtype=tf.float32)
    return tf.concat([beta, other_params], axis=0)
