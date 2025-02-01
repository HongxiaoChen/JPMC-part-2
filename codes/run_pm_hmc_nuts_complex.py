import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from run_pm_hmc import (ensure_directories, create_log_file, log_message)
from nuts_hnn_olm_complex import nuts_hnn_sample
from params import (MCMC_PARAMS, DATA_PARAMS, INIT_PARAMS, NUTS_PARAMS, USE_HNN, initialize_theta)
import time
from generate_samples import generate_samples
from pm_hmc_steps import initialize_hnn
import os


def plot_parameter_traces_nuts(samples, step_size):
    """
    Plot parameter traces for NUTS sampling results using post-burn-in samples

    Args:
        samples: Sampling results array [num_samples, dim]
        step_size: NUTS step size [1]

    """
    # Get post-burn-in samples
    burn_in = NUTS_PARAMS['burn_in']
    post_burn_samples = samples[burn_in:]

    num_features = DATA_PARAMS['NUM_FEATURES']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Calculate indices for parameters
    mu_start_idx = num_features  # mu starts after beta
    lambda_start_idx = num_features + 2  # lambda starts after mu

    # Left plot: μ₁ and μ₂ trajectories
    ax1.plot(post_burn_samples[:, mu_start_idx], color='blue', alpha=0.7, label='μ₁')
    ax1.plot(post_burn_samples[:, mu_start_idx + 1], color='red', alpha=0.7, label='μ₂')
    ax1.axhline(y=np.mean(post_burn_samples[:, mu_start_idx]), color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=np.mean(post_burn_samples[:, mu_start_idx + 1]), color='red', linestyle='--', alpha=0.5)
    ax1.set_title('μ₁ and μ₂ (Post Burn-in)')
    ax1.set_xlabel('Post Burn-in Iteration')
    ax1.set_ylabel('Value')
    ax1.set_ylim(-2, 6)
    ax1.legend()

    # Right plot: 1/λ₁ and 1/λ₂ trajectories
    inv_lambda1 = 1.0 / np.exp(post_burn_samples[:, lambda_start_idx])
    inv_lambda2 = 1.0 / np.exp(post_burn_samples[:, lambda_start_idx + 1])

    mean_inv_lambda1 = np.mean(inv_lambda1)
    mean_inv_lambda2 = np.mean(inv_lambda2)

    ax2.plot(inv_lambda1, color='blue', alpha=0.7, label='1/λ₁')
    ax2.plot(inv_lambda2, color='red', alpha=0.7, label='1/λ₂')
    ax2.axhline(y=mean_inv_lambda1, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(y=mean_inv_lambda2, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('1/λ₁ and 1/λ₂ (Post Burn-in)')
    ax2.set_xlabel('Post Burn-in Iteration')
    ax2.set_ylabel('Value')
    ax2.set_ylim(0, 10)
    ax2.legend()

    # Set title with sampling parameters
    plt.suptitle(
        f"{'HNN' if USE_HNN else 'Traditional'} NUTS - "
        'Parameter Traces (Post Burn-in)\n'
        f'Step size = {step_size}'
        f'Max stop criteria: {NUTS_PARAMS["stop_criteria_threshold"]} '
    )

    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'figures/nuts_parameter_traces_{timestamp}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def run_pm_hmc_nuts(Y, Z, initial_theta, beta_true, hnn_model=None):
    """
    Run PM-HMC with NUTS sampling

    Args:
        Y: Observed data [T, n]
        Z: Covariates [T, n, p_z]
        initial_theta: Initial parameter values [dim]
        hnn_model: HNN model (if using HNN mode)

    Returns:
        samples: Parameter samples [num_samples, dim]
        log_file: Log file handle
    """
    # Create log file
    log_file = create_log_file()
    log_message(log_file,
                f"Using {'HNN' if USE_HNN else 'Traditional'} NUTS")

    # Log initial parameters
    log_message(log_file, "Starting NUTS sampling with parameters:")
    log_message(log_file, f"Total samples: {NUTS_PARAMS['total_samples']}")
    log_message(log_file, f"Burn-in: {NUTS_PARAMS['burn_in']}")
    log_message(log_file, f"Step size: {NUTS_PARAMS['nuts_step_size']}")
    log_message(log_file, f"HNN threshold: {NUTS_PARAMS['hnn_threshold']}")
    log_message(log_file, f"Leapfrog threshold: {NUTS_PARAMS['leapfrog_threshold']}")
    log_message(log_file, f"Cooldown period: {NUTS_PARAMS['n_cooldown']}")
    log_message(log_file, f"Rho size: {NUTS_PARAMS['rho_size']}")
    log_message(log_file, f"Max depth: {NUTS_PARAMS['max_depth']}")
    log_message(log_file, f"Max stop criteria: {NUTS_PARAMS['stop_criteria_threshold']}")
    # Initialize auxiliary variables [T, N]
    current_u = tf.random.normal([DATA_PARAMS['T'], MCMC_PARAMS['N']])

    # Record start time
    start_time = time.time()

    # Run NUTS sampling
    samples, _, errors, (final_theta, final_u) = nuts_hnn_sample(
        theta=initial_theta,
        u=current_u,
        y=Y,
        Z=Z,
        hnn_model=hnn_model,
        traditional_only=NUTS_PARAMS['traditional_only'],
        logger=log_file
    )

    # Calculate statistics and log results
    total_time = time.time() - start_time


    # Calculate posterior means
    burn_in = NUTS_PARAMS['burn_in']
    effective_samples = samples[burn_in:]
    num_features = DATA_PARAMS['NUM_FEATURES']

    posterior_mean_beta = tf.reduce_mean(effective_samples[:, :num_features], axis=0)
    posterior_mean_mu = tf.reduce_mean(effective_samples[:, num_features:num_features + 2], axis=0)

    lambda1 = tf.exp(effective_samples[:, num_features + 2])
    lambda2 = tf.exp(effective_samples[:, num_features + 3])

    log_lambda_1 = effective_samples[:, num_features + 2]
    log_lambda_2 = effective_samples[:, num_features + 3]

    inv_lambda1 = 1.0 / tf.exp(effective_samples[:, num_features + 2])
    inv_lambda2 = 1.0 / tf.exp(effective_samples[:, num_features + 3])
    posterior_mean_lambda = tf.stack([
        tf.reduce_mean(log_lambda_1),
        tf.reduce_mean(log_lambda_2)
    ])

    posterior_mean_w = 1.0 - tf.sigmoid(tf.reduce_mean(effective_samples[:, -1]))

    # Log results
    log_message(log_file, f"\nSampling completed in {total_time / 60:.1f} minutes")

    log_message(log_file, "\nParameter estimates:")
    log_message(log_file, f"True Beta: {beta_true.numpy().tolist()}")
    log_message(log_file, f"Estimated Beta: {posterior_mean_beta.numpy().tolist()}")
    log_message(log_file, f"mu1: {posterior_mean_mu[0].numpy()}")
    log_message(log_file, f"mu2: {posterior_mean_mu[1].numpy()}")
    log_message(log_file, f"log(lambda1): {posterior_mean_lambda[0].numpy()}")
    log_message(log_file, f"log(lambda2): {posterior_mean_lambda[1].numpy()}")
    log_message(log_file, f"w1: {posterior_mean_w.numpy()}")

    return samples, errors, log_file


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Initialize HNN model if needed
    hnn_model = initialize_hnn() if USE_HNN else None

    # Generate synthetic data
    T = DATA_PARAMS['T']
    n = DATA_PARAMS['N_SUBJECTS']
    num_features = DATA_PARAMS['NUM_FEATURES']
    Y, X, Z, beta_true = generate_samples(T, n, num_features)

    # Initialize parameters
    initial_theta = initialize_theta()

    # Ensure directories exist
    ensure_directories()

    # Run NUTS sampling
    samples, errors, log_file = run_pm_hmc_nuts(
        Y, Z, initial_theta, beta_true, hnn_model)

    # Generate plots
    log_message(log_file, "\nGenerating parameter trace plots...")
    plot_parameter_traces_nuts(samples, NUTS_PARAMS['nuts_step_size'])
    log_message(log_file, "Parameter trace plots saved to figures directory.")

    # Close log file
    log_file.close()

    # Print final results
    print("\nNUTS sampling completed:")
    print(f"Total samples: {NUTS_PARAMS['total_samples']}")
    print(f"Burn-in samples: {NUTS_PARAMS['burn_in']}")
