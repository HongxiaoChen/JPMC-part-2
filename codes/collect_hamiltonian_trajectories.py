import tensorflow as tf
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import logging
from params import TRAJ_PARAMS, DATA_PARAMS, MCMC_PARAMS, INIT_PARAMS, initialize_theta
from log_likelihood_stable import compute_log_likelihood_and_gradients
# from log_likelihood_auto import compute_log_likelihood_and_gradients_auto as compute_log_likelihood_and_gradients
from pm_hmc_steps import compute_hamiltonian, full_step_A, full_step_B
from generate_samples import generate_samples


def setup_logger(name):
    """
    Set up a logger for tracking trajectory collection

    Args:
        name: Logger name

    Returns:
        logger: Configured logging object
    """
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'hamiltonian_trajectories_{timestamp}.log'

    logger = logging.getLogger(name)
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


logger = setup_logger('Hamiltonian_Trajectory_Collection')


@tf.function
def collect_single_trajectory(theta, u, y, Z, h, L, rho_size):
    """
    Collect one complete trajectory for Hamiltonian training data

    Args:
        theta: Initial parameter vector [13]
        u: Initial auxiliary variable [T, N]
        y: Observations [T, n]
        Z: Covariates [T, n, p_z]
        h: Step size
        L: Number of leapfrog steps
        rho_size: Scale factor for rho

    Returns:
        tuple: (thetas, rhos, hamiltonians, grad_thetas, grad_rhos,
                final_theta, final_u)
    """
    # Initialize momenta
    d = theta.shape[0]
    rho = tf.random.normal([d]) * tf.sqrt(rho_size)
    p = tf.random.normal(tf.shape(u))

    # Initialize TensorArrays
    thetas = tf.TensorArray(tf.float32, size=L, clear_after_read=False)
    rhos = tf.TensorArray(tf.float32, size=L, clear_after_read=False)
    hamiltonians = tf.TensorArray(tf.float32, size=L, clear_after_read=False)
    grad_thetas = tf.TensorArray(tf.float32, size=L, clear_after_read=False)
    grad_rhos = tf.TensorArray(tf.float32, size=L, clear_after_read=False)

    # Current state
    current_theta, current_rho = theta, rho
    current_u, current_p = u, p

    for l in tf.range(L):
        # Half step A
        next_theta, next_rho, next_u, next_p = full_step_A(
            current_theta, current_rho, current_u, current_p,
            h / 2, y, Z, rho_size
        )

        # Store states and compute Hamiltonian after half step A
        thetas = thetas.write(l, next_theta)
        rhos = rhos.write(l, next_rho)
        hamiltonians = hamiltonians.write(l, compute_hamiltonian(
            next_theta, next_rho, next_u, next_p,
            y, Z, rho_size
        ))
        grad_rhos = grad_rhos.write(l, next_rho / rho_size)
        _, _, grad_theta = compute_log_likelihood_and_gradients(next_theta, next_u, y, Z)
        grad_theta = -1.0 * grad_theta
        grad_thetas = grad_thetas.write(l, grad_theta)
        # Full step B
        next_theta, next_rho, next_u, next_p = full_step_B(
            next_theta, next_rho, next_u, next_p,
            h, y, Z, rho_size, None, True, False
        )

        # Half step A
        current_theta, current_rho, current_u, current_p = full_step_A(
            next_theta, next_rho, next_u, next_p,
            h / 2, y, Z, rho_size
        )

    return (thetas.stack(), rhos.stack(), hamiltonians.stack(),
            grad_thetas.stack(), grad_rhos.stack(),
            current_theta, current_u)


@tf.function
def collect_chain_samples(chain_id, num_samples, initial_theta, initial_u, Y, Z, h, L, rho_size, start_time):
    """
    Collect samples for a single chain

    Args:
        chain_id: ID of current chain
        num_samples: Number of samples to collect
        initial_theta, initial_u: Initial parameter and auxiliary values
        Y, Z: Observations and covariates
        h, L: Step size and number of steps
        rho_size: Scale factor for rho
        start_time: Start time for tracking

    Returns:
        tuple: Chain results containing trajectories of states and gradients
    """
    chain_thetas = tf.TensorArray(tf.float32, size=num_samples, clear_after_read=False)
    chain_rhos = tf.TensorArray(tf.float32, size=num_samples, clear_after_read=False)
    chain_hamiltonians = tf.TensorArray(tf.float32, size=num_samples, clear_after_read=False)
    chain_grad_thetas = tf.TensorArray(tf.float32, size=num_samples, clear_after_read=False)
    chain_grad_rhos = tf.TensorArray(tf.float32, size=num_samples, clear_after_read=False)

    current_theta = initial_theta
    current_u = initial_u

    for i in tf.range(num_samples):
        results = collect_single_trajectory(
            current_theta, current_u, Y, Z, h, L, rho_size
        )

        chain_thetas = chain_thetas.write(i, results[0])
        chain_rhos = chain_rhos.write(i, results[1])
        chain_hamiltonians = chain_hamiltonians.write(i, results[2])
        chain_grad_thetas = chain_grad_thetas.write(i, results[3])
        chain_grad_rhos = chain_grad_rhos.write(i, results[4])

        current_theta = results[5]
        current_u = results[6]

    return (chain_thetas.stack(), chain_rhos.stack(),
            chain_hamiltonians.stack(), chain_grad_thetas.stack(),
            chain_grad_rhos.stack())


def collect_training_data(Y, Z, num_samples, L, num_chains=4):
    """
    Collect training data from multiple chains for Hamiltonian learning

    Args:
        Y: Observations [T, n]
        Z: Covariates [T, n, p_z]
        num_samples: Number of samples per chain
        L: Number of steps per trajectory
        num_chains: Number of parallel chains

    Returns:
        dict: Training data containing:
            - thetas: [valid_chains * num_samples * L, 13]
            - rhos: [valid_chains * num_samples * L, 13]
            - hamiltonians: [valid_chains * num_samples * L]
            - grad_thetas: [valid_chains * num_samples * L, 13]
            - grad_rhos: [valid_chains * num_samples * L, 13]
    """
    logger.info(f"Starting trajectory collection with {num_chains} chains")
    logger.info(f"Configuration:")
    logger.info(f"- Number of samples per chain: {num_samples}")
    logger.info(f"- Trajectory length (L): {L}")
    logger.info(f"- Total trajectories to collect: {num_chains * num_samples}")
    logger.info(f"- Total data points expected: {num_chains * num_samples * L}")

    start_time = time.time()

    # Get parameters
    T = DATA_PARAMS['T']
    N = MCMC_PARAMS['N']
    h = MCMC_PARAMS['H']
    rho_size = MCMC_PARAMS['RHO_SIZE']

    # Initialize states
    initial_theta = initialize_theta()
    initial_u = tf.random.normal([T, N])

    logger.info("\nInitialization complete:")
    logger.info(f"- Initial theta shape: {initial_theta.shape}")
    logger.info(f"- Initial u shape: {initial_u.shape}")

    # Store results for all chains
    valid_chains_thetas = tf.TensorArray(tf.float32, size=num_chains, dynamic_size=True, clear_after_read=False)
    valid_chains_rhos = tf.TensorArray(tf.float32, size=num_chains, dynamic_size=True, clear_after_read=False)
    valid_chains_hamiltonians = tf.TensorArray(tf.float32, size=num_chains, dynamic_size=True, clear_after_read=False)
    valid_chains_grad_thetas = tf.TensorArray(tf.float32, size=num_chains, dynamic_size=True, clear_after_read=False)
    valid_chains_grad_rhos = tf.TensorArray(tf.float32, size=num_chains, dynamic_size=True, clear_after_read=False)

    valid_chain_count = 0
    total_trajectories = num_chains * num_samples
    trajectories_completed = 0

    for chain in tf.range(num_chains):
        chain_start_time = time.time()
        logger.info(f"\nStarting chain {chain + 1}/{num_chains}")

        chain_results = collect_chain_samples(
            chain, num_samples, initial_theta, initial_u,
            Y, Z, h, L, rho_size, start_time
        )

        # Check for NaN values in any component
        has_nan = tf.reduce_any([
            tf.reduce_any(tf.math.is_nan(chain_results[0])),  # thetas
            tf.reduce_any(tf.math.is_nan(chain_results[1])),  # rhos
            tf.reduce_any(tf.math.is_nan(chain_results[2])),  # hamiltonians
            tf.reduce_any(tf.math.is_nan(chain_results[3])),  # grad_thetas
            tf.reduce_any(tf.math.is_nan(chain_results[4]))  # grad_rhos
        ])

        if not has_nan:
            valid_chains_thetas = valid_chains_thetas.write(valid_chain_count, chain_results[0])
            valid_chains_rhos = valid_chains_rhos.write(valid_chain_count, chain_results[1])
            valid_chains_hamiltonians = valid_chains_hamiltonians.write(valid_chain_count, chain_results[2])
            valid_chains_grad_thetas = valid_chains_grad_thetas.write(valid_chain_count, chain_results[3])
            valid_chains_grad_rhos = valid_chains_grad_rhos.write(valid_chain_count, chain_results[4])
            valid_chain_count += 1
            trajectories_completed += num_samples

            chain_time = time.time() - chain_start_time
            total_time = time.time() - start_time

            progress = trajectories_completed / total_trajectories
            estimated_remaining = (total_time / trajectories_completed) * (total_trajectories - trajectories_completed)

            logger.info(f"Chain {chain + 1} completed in {chain_time:.2f} seconds")
            logger.info(f"Progress: {trajectories_completed}/{total_trajectories} trajectories ({progress * 100:.1f}%)")
            logger.info(f"Time elapsed: {total_time:.2f}s, "
                        f"Estimated remaining: {estimated_remaining:.2f}s ({estimated_remaining / 60:.1f}min)")

            # Output chain statistics
            logger.info("\nChain statistics:")
            logger.info(
                f"- Theta range: [{tf.reduce_min(chain_results[0]):.4f}, {tf.reduce_max(chain_results[0]):.4f}]")
            logger.info(f"- Rho range: [{tf.reduce_min(chain_results[1]):.4f}, {tf.reduce_max(chain_results[1]):.4f}]")
            logger.info(
                f"- Hamiltonian range: [{tf.reduce_min(chain_results[2]):.4f}, {tf.reduce_max(chain_results[2]):.4f}]")
            logger.info(
                f"- Grad theta range: [{tf.reduce_min(chain_results[3]):.4f}, {tf.reduce_max(chain_results[3]):.4f}]")
            logger.info(
                f"- Grad rho range: [{tf.reduce_min(chain_results[4]):.4f}, {tf.reduce_max(chain_results[4]):.4f}]")
        else:
            logger.warning(f"Chain {chain + 1} contains NaN values - discarding chain")
            continue

    # Check if we have any valid chains
    if valid_chain_count == 0:
        raise ValueError("All chains contained NaN values!")

    # Reshape all valid data
    all_thetas = tf.reshape(valid_chains_thetas.stack()[:valid_chain_count], [-1, initial_theta.shape[0]])
    all_rhos = tf.reshape(valid_chains_rhos.stack()[:valid_chain_count], [-1, initial_theta.shape[0]])
    all_hamiltonians = tf.reshape(valid_chains_hamiltonians.stack()[:valid_chain_count], [-1])
    all_grad_thetas = tf.reshape(valid_chains_grad_thetas.stack()[:valid_chain_count], [-1, initial_theta.shape[0]])
    all_grad_rhos = tf.reshape(valid_chains_grad_rhos.stack()[:valid_chain_count], [-1, initial_theta.shape[0]])

    training_data = {
        'thetas': all_thetas,
        'rhos': all_rhos,
        'hamiltonians': all_hamiltonians,
        'grad_thetas': all_grad_thetas,
        'grad_rhos': all_grad_rhos
    }

    total_time = time.time() - start_time
    logger.info("\nTrajectory collection completed:")
    logger.info(f"Valid chains: {valid_chain_count}/{num_chains}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time / 60:.1f}min)")
    logger.info(f"Final data shapes:")
    for key, value in training_data.items():
        logger.info(f"- {key}: {value.shape}")

    return training_data


def save_trajectories(thetas, rhos, hamiltonians, grad_thetas, grad_rhos):
    """
    Save trajectory data to npz file

    Args:
        thetas: Tensor of shape [num_chains * num_samples * L, 13]
        rhos: Tensor of shape [num_chains * num_samples * L, 13]
        hamiltonians: Tensor of shape [num_chains * num_samples * L]
        grad_thetas: Tensor of shape [num_chains * num_samples * L, 13]
        grad_rhos: Tensor of shape [num_chains * num_samples * L, 13]
    """
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Ensure save directory exists
    save_path = Path(TRAJ_PARAMS['SAVE_DIR'])
    save_path.mkdir(exist_ok=True)

    # Convert to numpy arrays and save
    np.savez(
        save_path / f'hamiltonian_trajectories_{timestamp}.npz',
        thetas=thetas.numpy(),
        rhos=rhos.numpy(),
        hamiltonians=hamiltonians.numpy(),
        grad_thetas=grad_thetas.numpy(),
        grad_rhos=grad_rhos.numpy()
    )

    # Log information
    logger.info(f"Saved trajectory data to {save_path}/hamiltonian_trajectories_{timestamp}.npz")
    logger.info("Data shapes:")
    logger.info(f"thetas: {thetas.shape}")
    logger.info(f"rhos: {rhos.shape}")
    logger.info(f"hamiltonians: {hamiltonians.shape}")
    logger.info(f"grad_thetas: {grad_thetas.shape}")
    logger.info(f"grad_rhos: {grad_rhos.shape}")

    # Log basic statistics
    logger.info("\nBasic statistics:")
    logger.info(f"theta mean: {tf.reduce_mean(thetas):.4f}")
    logger.info(f"theta std: {tf.math.reduce_std(thetas):.4f}")
    logger.info(f"rho mean: {tf.reduce_mean(rhos):.4f}")
    logger.info(f"rho std: {tf.math.reduce_std(rhos):.4f}")
    logger.info(f"hamiltonian mean: {tf.reduce_mean(hamiltonians):.4f}")
    logger.info(f"hamiltonian std: {tf.math.reduce_std(hamiltonians):.4f}")
    logger.info(f"grad_theta mean: {tf.reduce_mean(grad_thetas):.4f}")
    logger.info(f"grad_theta std: {tf.math.reduce_std(grad_thetas):.4f}")
    logger.info(f"grad_rho mean: {tf.reduce_mean(grad_rhos):.4f}")
    logger.info(f"grad_rho std: {tf.math.reduce_std(grad_rhos):.4f}")

    # Log data ranges
    logger.info("\nData ranges:")
    logger.info(f"theta range: [{tf.reduce_min(thetas):.4f}, {tf.reduce_max(thetas):.4f}]")
    logger.info(f"rho range: [{tf.reduce_min(rhos):.4f}, {tf.reduce_max(rhos):.4f}]")
    logger.info(f"hamiltonian range: [{tf.reduce_min(hamiltonians):.4f}, {tf.reduce_max(hamiltonians):.4f}]")
    logger.info(f"grad_theta range: [{tf.reduce_min(grad_thetas):.4f}, {tf.reduce_max(grad_thetas):.4f}]")
    logger.info(f"grad_rho range: [{tf.reduce_min(grad_rhos):.4f}, {tf.reduce_max(grad_rhos):.4f}]")


if __name__ == "__main__":
    # Generate test data
    T = DATA_PARAMS['T']
    n = DATA_PARAMS['N_SUBJECTS']
    num_features = DATA_PARAMS['NUM_FEATURES']

    Y, X, Z, beta = generate_samples(T, n, num_features)

    # Collect trajectories
    training_data = collect_training_data(
        Y, Z,
        num_samples=TRAJ_PARAMS['M'],
        L=TRAJ_PARAMS['TRAJ_LENGTH'],
        num_chains=TRAJ_PARAMS['num_chains']
    )

    print("Collection completed.")
    print("Data shapes:")
    for key, value in training_data.items():
        print(f"{key}: {value.shape}")