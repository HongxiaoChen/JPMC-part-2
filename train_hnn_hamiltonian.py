import tensorflow as tf
import time
from datetime import datetime
import logging
from pathlib import Path
from generate_samples import generate_samples
from collect_hamiltonian_trajectories import collect_training_data
from nn_architectures_hamiltonian import HNN
from params import DATA_PARAMS, MCMC_PARAMS, TRAJ_PARAMS, NN_PARAMS, TRAIN_PARAMS


def setup_train_logger():
    """Set up a logger for training process"""
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_hamiltonian_{timestamp}.log'

    logger = logging.getLogger('Hamiltonian_HNN_Training')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    for handler in [fh, ch]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_train_logger()


def setup_checkpoints():
    """Set up checkpoint directory"""
    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir


def create_dataset(data_dict, batch_size):
    """
    Convert training data to tf.data.Dataset format

    Args:
        data_dict: Dictionary containing training data
        batch_size: Batch size for training

    Returns:
        tf.data.Dataset: Dataset for training
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        (data_dict['thetas'], data_dict['rhos']),  # inputs
        (data_dict['hamiltonians'], data_dict['grad_thetas'], data_dict['grad_rhos'])  # targets
    ))
    return dataset.shuffle(75000000).batch(batch_size)


def compute_loss(model, thetas, rhos, hamiltonians_true, grad_thetas_true, grad_rhos_true):
    """
    Compute model loss with dimension-wise RMS normalization for gradient components.

    Args:
        model: Neural network model that computes Hamiltonian and its gradients
        thetas: Parameter vectors, shape [batch_size, 13]
        rhos: Momentum vectors, shape [batch_size, 13]
        hamiltonians_true: True Hamiltonian values, shape [batch_size]
        grad_thetas_true: True gradients w.r.t theta, shape [batch_size, 13]
        grad_rhos_true: True gradients w.r.t rho, shape [batch_size, 13]

    Returns:
        tuple: (total_loss, hamiltonian_loss, grad_theta_loss, grad_rho_loss)
            - total_loss: Combined loss value, shape []
            - hamiltonian_loss: Loss of Hamiltonian prediction, shape []
            - grad_theta_loss: Loss of theta gradient prediction, shape []
            - grad_rho_loss: Loss of rho gradient prediction, shape []
    """
    thetas = tf.cast(thetas, tf.float32)
    rhos = tf.cast(rhos, tf.float32)

    # Get predictions from model
    _, grad_thetas_pred, grad_rhos_pred = model.compute_gradients(thetas, rhos)
    hamiltonian_loss = 0.0

    # grad_thetas_true/grad_thetas_pred: shape [batch_size, 13]
    # First compute RMS (or max) of true values for each dimension i
    # Shape [13]
    grad_theta_rms_per_dim = tf.sqrt(
        tf.reduce_mean(tf.square(grad_thetas_true), axis=0)  # axis=0 => average over batch
    ) + 1e-8

    # Compute error (pred - true)
    grad_theta_diff = grad_thetas_pred - grad_thetas_true  # shape [batch, 13]

    # Divide by dimension RMS => shape [batch, 13]
    grad_theta_diff_norm = grad_theta_diff / grad_theta_rms_per_dim

    # Compute mean squared error over all elements
    grad_theta_loss = tf.reduce_mean(tf.square(grad_theta_diff_norm))

    # ========== Similar normalization for grad_rho with manual weight (e.g. 5.0) ==========
    grad_rho_rms_per_dim = tf.sqrt(
        tf.reduce_mean(tf.square(grad_rhos_true), axis=0)
    ) + 1e-8

    grad_rho_diff = grad_rhos_pred - grad_rhos_true
    grad_rho_diff_norm = grad_rho_diff / grad_rho_rms_per_dim
    grad_rho_loss = tf.reduce_mean(tf.square(grad_rho_diff_norm)) * 1.0  # Example weight of 5

    # ========== Combine total loss ==========
    total_loss = hamiltonian_loss + grad_theta_loss + grad_rho_loss

    # Check for NaN values
    if tf.math.is_nan(total_loss):
        logger.warning("NaN loss detected!")
        return tf.constant(1e10, dtype=tf.float32), hamiltonian_loss, grad_theta_loss, grad_rho_loss

    return total_loss, hamiltonian_loss, grad_theta_loss, grad_rho_loss

@tf.function
def train_step(model, optimizer, thetas, rhos, hamiltonians_true, grad_thetas_true, grad_rhos_true):
    """
    Single training step

    Args:
        model: HNN model
        optimizer: Optimizer instance
        thetas, rhos: Input tensors
        hamiltonians_true, grad_thetas_true, grad_rhos_true: Target tensors

    Returns:
        tuple: Losses (total, hamiltonian, grad_theta, grad_rho)
    """
    with tf.GradientTape() as tape:
        total_loss, hamiltonian_loss, grad_theta_loss, grad_rho_loss = compute_loss(
            model, thetas, rhos, hamiltonians_true, grad_thetas_true, grad_rhos_true
        )

    # Compute gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)

    # Gradient clipping
    gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, hamiltonian_loss, grad_theta_loss, grad_rho_loss


def train_model():
    """Train the Hamiltonian Neural Network"""
    # Initialize model and checkpoint directory
    model = HNN(activation='sin')
    ckpt_dir = setup_checkpoints()

    # Create fixed timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = ckpt_dir / f'weights_hamiltonian_{timestamp}'

    for outer_loop in range(TRAIN_PARAMS['NUM_OUTER_LOOPS']):
        logger.info(f"\nStarting outer loop {outer_loop + 1}/{TRAIN_PARAMS['NUM_OUTER_LOOPS']}")

        # Generate new training data
        logger.info("Generating new training data...")
        start_time = time.time()

        # Generate sample data
        Y, _, Z, _ = generate_samples(
            T=DATA_PARAMS['T'],
            n=DATA_PARAMS['N_SUBJECTS'],
            num_features=DATA_PARAMS['NUM_FEATURES']
        )

        # Collect trajectory data
        training_data = collect_training_data(
            Y, Z,
            num_samples=TRAJ_PARAMS['M'],
            L=TRAJ_PARAMS['TRAJ_LENGTH'],
            num_chains=TRAJ_PARAMS['num_chains']
        )

        data_gen_time = time.time() - start_time
        logger.info(f"Data generation completed in {data_gen_time:.2f} seconds")

        # Create dataset
        dataset = create_dataset(training_data, TRAIN_PARAMS['BATCH_SIZE'])
        steps_per_epoch = tf.data.experimental.cardinality(dataset).numpy()

        # Initialize optimizer with learning rate schedule
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=TRAIN_PARAMS['INITIAL_LR'],
            decay_steps=TRAIN_PARAMS['LR_DECAY']['DECAY_EPOCHS'] * steps_per_epoch,
            decay_rate=TRAIN_PARAMS['LR_DECAY']['DECAY_RATE'],
            staircase=TRAIN_PARAMS['LR_DECAY']['STAIRCASE']
        )
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

        # Training loop
        logger.info(f"Starting training for {TRAIN_PARAMS['NUM_EPOCHS']} epochs...")
        epoch_times = []

        for epoch in range(TRAIN_PARAMS['NUM_EPOCHS']):
            epoch_start = time.time()
            total_loss = 0
            hamiltonian_loss = 0
            grad_theta_loss = 0
            grad_rho_loss = 0

            for batch_inputs, batch_targets in dataset:
                thetas, rhos = batch_inputs
                hamiltonians_true, grad_thetas_true, grad_rhos_true = batch_targets

                batch_losses = train_step(
                    model, optimizer, thetas, rhos,
                    hamiltonians_true, grad_thetas_true, grad_rhos_true
                )

                total_loss += batch_losses[0]
                hamiltonian_loss += batch_losses[1]
                grad_theta_loss += batch_losses[2]
                grad_rho_loss += batch_losses[3]

            # Compute average losses
            total_loss /= steps_per_epoch
            hamiltonian_loss /= steps_per_epoch
            grad_theta_loss /= steps_per_epoch
            grad_rho_loss /= steps_per_epoch

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # Estimate remaining time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = TRAIN_PARAMS['NUM_EPOCHS'] - (epoch + 1)
            estimated_remaining_time = avg_epoch_time * remaining_epochs

            logger.info(
                f"Outer loop {outer_loop + 1}/{TRAIN_PARAMS['NUM_OUTER_LOOPS']}, "
                f"Epoch {epoch + 1}/{TRAIN_PARAMS['NUM_EPOCHS']}, "
                f"Loss: {total_loss:.4f} (H: {hamiltonian_loss:.4f}, "
                f"grad_θ: {grad_theta_loss:.4f}, grad_ρ: {grad_rho_loss:.4f}), "
                f"Time: {epoch_time:.2f}s, "
                f"Est. remaining: {estimated_remaining_time / 60:.1f}min"
            )

        # Save weights
        model.save_weights(str(save_path))
        logger.info(f"Saved model weights to {save_path}")

        # Clean up memory
        del training_data
        del dataset
        del optimizer
        tf.keras.backend.clear_session()

    return model


if __name__ == "__main__":
    trained_model = train_model()
    logger.info("Training completed successfully!")