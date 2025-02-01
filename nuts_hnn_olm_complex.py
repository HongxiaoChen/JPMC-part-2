import tensorflow as tf
from pm_hmc_steps import full_step_A, full_step_B, compute_hamiltonian, metropolis_step
from params import NUTS_PARAMS, USE_HNN

@tf.function
def single_leapfrog_update(theta, rho, u, p, h, y, Z, hnn_model, traditional_only):
    """
    Perform a single leapfrog update (half step A + full step B + half step A)

    Args:
        theta: Parameter vector [13]
        rho: Momentum vector [13]
        u: Auxiliary variable [T,N,p]
        p: Auxiliary momentum [T,N,p]
        h: Step size (scalar)
        y: Observations [T,n]
        Z: Covariates [T,n,p_z]
        hnn_model: HNN model
        traditional_only: Whether to use only traditional leapfrog

    Returns:
        Updated theta, rho, u, p
    """
    # Cast inputs to appropriate types
    h = tf.cast(h, tf.float32)
    rho_size = tf.cast(NUTS_PARAMS['rho_size'], tf.float32)

    # Half step A
    theta, rho, u, p = full_step_A(theta, rho, u, p, h / 2, y, Z, rho_size)

    # Full step B
    theta, rho, u, p = full_step_B(theta, rho, u, p, h, y, Z, rho_size, hnn_model, traditional_only)

    # Half step A
    theta, rho, u, p = full_step_A(theta, rho, u, p, h / 2, y, Z, rho_size)

    return theta, rho, u, p


def build_tree(theta, rho, u, p, slice_variable, v, j, step_size, H0, y, Z, hnn_model,
               theta_dim, hnn_threshold, leapfrog_threshold, use_leapfrog, traditional_only):
    """
    Recursive function to build the NUTS binary tree

    Args:
        theta: Parameter vector [13]
        rho: Momentum vector [13]
        u: Auxiliary variable [T,N,p]
        p: Auxiliary momentum [T,N,p]
        slice_variable: Slice sampling variable (scalar)
        v: Direction {-1, 1} (scalar)
        j: Tree depth (scalar)
        step_size: Leapfrog step size (scalar)
        H0: Initial Hamiltonian value (scalar)
        y: Observations [T,n]
        Z: Covariates [T,n,p_z]
        hnn_model: HNN model
        theta_dim: Dimension of theta
        hnn_threshold: HNN error threshold
        leapfrog_threshold: Leapfrog error threshold
        use_leapfrog: Whether to use leapfrog
        traditional_only: Whether to use only traditional leapfrog

    Returns:
        Updated variables and statistics
    """
    # Cast inputs to appropriate types
    v = tf.cast(v, tf.float32)
    j = tf.cast(j, tf.int32)
    step_size = tf.cast(step_size, tf.float32)
    H0 = tf.cast(H0, tf.float32)
    slice_variable = tf.cast(slice_variable, tf.float32)
    hnn_threshold = tf.cast(hnn_threshold, tf.float32)
    leapfrog_threshold = tf.cast(leapfrog_threshold, tf.float32)

    if tf.equal(j, 0):  # Base case - single leapfrog step
        if traditional_only:
            # Traditional leapfrog case
            theta_prime, rho_prime, u_prime, p_prime = single_leapfrog_update(
                theta, rho, u, p, v * step_size, y, Z, hnn_model, True)

            H_prime = compute_hamiltonian(theta_prime, rho_prime, u_prime, p_prime, y, Z, NUTS_PARAMS['rho_size'])
            s_prime = tf.math.log(slice_variable) + tf.squeeze(H_prime) <= leapfrog_threshold
            n_prime = tf.cast(slice_variable <= tf.exp(-tf.squeeze(H_prime)), tf.int32)

            return (theta_prime, rho_prime, theta_prime, rho_prime, theta_prime, rho_prime,
                    u_prime, p_prime, u_prime, p_prime, u_prime, p_prime,
                    n_prime, s_prime, tf.minimum(1.0, tf.exp(tf.squeeze(H0 - H_prime))), 1,
                    0.0, True)
        else:
            # HNN case
            theta_prime, rho_prime, u_prime, p_prime = single_leapfrog_update(
                theta, rho, u, p, v * step_size, y, Z, hnn_model, False)

            H_prime = compute_hamiltonian(theta_prime, rho_prime, u_prime, p_prime, y, Z, NUTS_PARAMS['rho_size'])
            error = tf.math.log(slice_variable) + tf.squeeze(H_prime)
            use_leapfrog_new = tf.logical_or(
                tf.cast(use_leapfrog, tf.bool),
                error > hnn_threshold
            )

            if use_leapfrog_new:
                # Switch to traditional leapfrog if needed
                theta_prime, rho_prime, u_prime, p_prime = single_leapfrog_update(
                    theta, rho, u, p, v * step_size, y, Z, hnn_model, True)

                H_prime = compute_hamiltonian(theta_prime, rho_prime, u_prime, p_prime, y, Z, NUTS_PARAMS['rho_size'])
                s_prime = tf.math.log(slice_variable) + tf.squeeze(H_prime) <= leapfrog_threshold
                n_prime = tf.cast(slice_variable <= tf.exp(-tf.squeeze(H_prime)), tf.int32)

                return (theta_prime, rho_prime, theta_prime, rho_prime, theta_prime, rho_prime,
                        u_prime, p_prime, u_prime, p_prime, u_prime, p_prime,
                        n_prime, s_prime, tf.minimum(1.0, tf.exp(tf.squeeze(H0 - H_prime))), 1,
                        error, use_leapfrog_new)
            else:
                # Continue with HNN
                s_prime = error <= hnn_threshold
                n_prime = tf.cast(slice_variable <= tf.exp(-tf.squeeze(H_prime)), tf.int32)

                return (theta_prime, rho_prime, theta_prime, rho_prime, theta_prime, rho_prime,
                        u_prime, p_prime, u_prime, p_prime, u_prime, p_prime,
                        n_prime, s_prime, tf.minimum(1.0, tf.exp(tf.squeeze(H0 - H_prime))), 1,
                        error, use_leapfrog_new)

    else:  # Recursive case

        # Build first subtree
        (theta_minus, rho_minus, theta_plus, rho_plus, theta_prime, rho_prime,
         u_minus, p_minus, u_plus, p_plus, u_prime, p_prime,
         n_prime, s_prime, alpha, n_alpha, error, use_leapfrog) = build_tree(
            theta, rho, u, p, slice_variable, v, j - 1, step_size, H0, y, Z,
            hnn_model, theta_dim, hnn_threshold, leapfrog_threshold, use_leapfrog,
            traditional_only)

        if tf.reduce_all(tf.cast(s_prime, tf.bool)):
            if tf.equal(v, -1):
                # Build left subtree
                (theta_minus_2, rho_minus_2, _, _, theta_prime_2, rho_prime_2,
                 u_minus_2, p_minus_2, _, _, u_prime_2, p_prime_2,
                 n_prime_2, s_prime_2, alpha_2, n_alpha_2, error_2,
                 use_leapfrog) = build_tree(
                    theta_minus, rho_minus, u_minus, p_minus,
                    slice_variable, v, j - 1, step_size, H0, y, Z,
                    hnn_model, theta_dim, hnn_threshold, leapfrog_threshold,
                    use_leapfrog, traditional_only)
                theta_minus = theta_minus_2
                rho_minus = rho_minus_2
                u_minus = u_minus_2
                p_minus = p_minus_2

            elif tf.equal(v, 1):
                # Build right subtree
                (_, _, theta_plus_2, rho_plus_2, theta_prime_2, rho_prime_2,
                 _, _, u_plus_2, p_plus_2, u_prime_2, p_prime_2,
                 n_prime_2, s_prime_2, alpha_2, n_alpha_2, error_2,
                 use_leapfrog) = build_tree(
                    theta_plus, rho_plus, u_plus, p_plus,
                    slice_variable, v, j - 1, step_size, H0, y, Z,
                    hnn_model, theta_dim, hnn_threshold, leapfrog_threshold,
                    use_leapfrog, traditional_only)
                theta_plus = theta_plus_2
                rho_plus = rho_plus_2
                u_plus = u_plus_2
                p_plus = p_plus_2

            else:
                tf.print('v not equal to 1 or -1')

            # Decide whether to accept new point
            accept_prob = tf.cast(n_prime_2, tf.float32) / tf.cast(n_prime + n_prime_2, tf.float32)
            should_accept = tf.less(tf.random.uniform([]), accept_prob)

            if should_accept:
                theta_prime = theta_prime_2
                rho_prime = rho_prime_2
                u_prime = u_prime_2
                p_prime = p_prime_2

            # Update stopping condition
            theta_delta = theta_plus - theta_minus
            u_delta = u_plus - u_minus

            # Calculate dot products for both parameter space and latent space
            theta_dot_minus = tf.tensordot(theta_delta, rho_minus, 1)
            theta_dot_plus = tf.tensordot(theta_delta, rho_plus, 1)

            u_dot_minus = tf.reduce_sum(u_delta * p_minus)
            u_dot_plus = tf.reduce_sum(u_delta * p_plus)
            # Combine dots for parameters and latent variables
            total_dot_minus = theta_dot_minus + u_dot_minus
            total_dot_plus = theta_dot_plus + u_dot_plus

            # Ensure s_prime_2 is a scalar boolean
            s_prime_2_scalar = tf.reduce_all(tf.cast(s_prime_2, tf.bool))

            # Check both U-turn condition and numerical stability
            s_prime = tf.logical_and(
                s_prime_2_scalar,
                tf.logical_and(
                    # Original NUTS U-turn condition
                    tf.logical_and(
                        tf.greater_equal(total_dot_minus, 0.0),
                        tf.greater_equal(total_dot_plus, 0.0)
                    ),
                    # Numerical stability condition
                    tf.logical_and(
                        tf.less(tf.abs(total_dot_minus), NUTS_PARAMS['stop_criteria_threshold']),
                        tf.less(tf.abs(total_dot_plus), NUTS_PARAMS['stop_criteria_threshold'])
                    )
                )
            )

            # Update statistics
            n_prime += n_prime_2
            alpha += alpha_2
            n_alpha += n_alpha_2
            error = tf.maximum(error, error_2)

        return (theta_minus, rho_minus, theta_plus, rho_plus, theta_prime, rho_prime,
                u_minus, p_minus, u_plus, p_plus, u_prime, p_prime,
                n_prime, s_prime, alpha, n_alpha, error, use_leapfrog)


def nuts_hnn_sample(theta, u, y, Z, hnn_model, traditional_only=False, logger=None):
    """
    Perform NUTS-HNN sampling

    Args:
        theta: Initial parameter vector [13]
        u: Initial auxiliary variable [T,N,p]
        y: Observations [T,n]
        Z: Covariates [T,n,p_z]
        hnn_model: HNN model
        traditional_only: Whether to use only traditional leapfrog
        logger: Logger for output messages

    Returns:
        samples: [num_samples, dim]
        acceptance: [num_samples]
        errors: [num_samples]
    """
    step_size = tf.cast(NUTS_PARAMS['nuts_step_size'], tf.float32)
    rho_size = tf.cast(NUTS_PARAMS['rho_size'], tf.float32)
    hnn_threshold = tf.cast(NUTS_PARAMS['hnn_error_threshold'], tf.float32)
    leapfrog_threshold = tf.cast(NUTS_PARAMS['leapfrog_threshold'], tf.float32)

    logger = logger

    # Get parameters from NUTS_PARAMS
    num_samples = NUTS_PARAMS['total_samples']
    num_burnin = NUTS_PARAMS['burn_in']
    n_cooldown = NUTS_PARAMS['n_cooldown']
    max_depth = NUTS_PARAMS['max_depth']

    if traditional_only:
        hnn_threshold = -1e-6  # Disable HNN if only traditional leapfrog is used

    # Initialize storage
    theta_dim = theta.shape[0]
    samples = tf.zeros([num_samples, theta_dim])
    acceptance = tf.zeros([num_samples])
    errors = tf.zeros([num_samples])

    # Initialize counters
    burnin_accepted = 0
    sampling_accepted = 0
    total_burnin_tried = 0
    total_sampling_tried = 0

    # Current state
    current_theta = theta
    current_u = u
    use_leapfrog = False
    leapfrog_count = 0

    for sample in range(num_samples):
        phase = "Burn-in" if sample < num_burnin else "Sampling"

        # Sample momentum
        rho = tf.random.normal([theta_dim]) * tf.sqrt(rho_size)
        p = tf.random.normal(tf.shape(u))

        # Compute initial Hamiltonian
        H0 = compute_hamiltonian(current_theta, rho, current_u, p, y, Z, rho_size)

        # Sample slice variable
        slice_variable = tf.random.uniform([]) * tf.exp(-tf.squeeze(H0))
        slice_variable = tf.cast(slice_variable, tf.float32)

        # Initialize tree
        theta_minus = current_theta
        theta_plus = current_theta
        rho_minus = rho
        rho_plus = rho
        u_minus = current_u
        u_plus = current_u
        p_minus = p
        p_plus = p
        j = 0
        n = 1
        s = True

        # Switch back to HNN after cooldown
        if use_leapfrog:
            leapfrog_count += 1
        if leapfrog_count == n_cooldown:
            use_leapfrog = False
            leapfrog_count = 0

        # Build tree
        max_error = tf.constant(-float('inf'))
        sample_accepted = False

        while s and (j < max_depth):
            # Choose direction
            v = tf.where(tf.random.uniform([]) < 0.5, tf.constant(1), tf.constant(-1))

            if tf.equal(v, -1):
                # Build left subtree
                (theta_minus, rho_minus, _, _, theta_prime, rho_prime,
                 u_minus, p_minus, _, _, u_prime, p_prime,
                 n_prime, s_prime, alpha, n_alpha, error,
                 use_leapfrog) = build_tree(
                    theta_minus, rho_minus, u_minus, p_minus,
                    slice_variable, v, j, step_size, H0, y, Z,
                    hnn_model, theta_dim, hnn_threshold, leapfrog_threshold,
                    use_leapfrog, traditional_only)
            else:
                # Build right subtree
                (_, _, theta_plus, rho_plus, theta_prime, rho_prime,
                 _, _, u_plus, p_plus, u_prime, p_prime,
                 n_prime, s_prime, alpha, n_alpha, error,
                 use_leapfrog) = build_tree(
                    theta_plus, rho_plus, u_plus, p_plus,
                    slice_variable, v, j, step_size, H0, y, Z,
                    hnn_model, theta_dim, hnn_threshold, leapfrog_threshold,
                    use_leapfrog, traditional_only)

            max_error = tf.maximum(max_error, error)

            # Metropolis step
            if s_prime:
                ratio = tf.cast(n_prime, tf.float32) / tf.cast(n, tf.float32)
                accept_prob = tf.minimum(1.0, ratio)
                should_accept = tf.less(tf.random.uniform([]), accept_prob)

                if should_accept:
                    current_theta = theta_prime
                    current_u = u_prime
                    sample_accepted = True

            # Update number of valid points
            n += n_prime

            # Update stopping criterion
            theta_delta = theta_plus - theta_minus
            u_delta = u_plus - u_minus

            theta_dot_minus = tf.tensordot(theta_delta, rho_minus, 1)
            theta_dot_plus = tf.tensordot(theta_delta, rho_plus, 1)

            u_dot_minus = tf.reduce_sum(u_delta * p_minus)
            u_dot_plus = tf.reduce_sum(u_delta * p_plus)
            total_dot_minus = theta_dot_minus + u_dot_minus
            total_dot_plus = theta_dot_plus + u_dot_plus

            s = tf.logical_and(
                s_prime,
                tf.logical_and(
                    # Original NUTS U-turn condition
                    tf.logical_and(
                        tf.greater_equal(total_dot_minus, 0.0),
                        tf.greater_equal(total_dot_plus, 0.0)
                    ),
                    # Numerical stability condition
                    tf.logical_and(
                        tf.less(tf.abs(total_dot_minus), NUTS_PARAMS['stop_criteria_threshold']),
                        tf.less(tf.abs(total_dot_plus), NUTS_PARAMS['stop_criteria_threshold'])
                    )
                )
            )

            j += 1

        # Record results
        if sample_accepted:
            if sample < num_burnin:
                burnin_accepted += 1
                total_burnin_tried += 1
            else:
                sampling_accepted += 1
                total_sampling_tried += 1

            acceptance = tf.tensor_scatter_nd_update(
                acceptance,
                [[sample]],
                [1.0]
            )

            # Print progress
            if sample < num_burnin:
                current_rate = burnin_accepted / total_burnin_tried
            else:
                current_rate = sampling_accepted / total_sampling_tried

            print(f"Sample {sample + 1}/{num_samples} ({phase}): "
                  f"ACCEPTED (current rate: {current_rate:.4f})",
                  file=logger)
        else:
            if sample < num_burnin:
                total_burnin_tried += 1
            else:
                total_sampling_tried += 1

        # Record sample and error
        samples = tf.tensor_scatter_nd_update(
            samples,
            [[sample]],
            [current_theta]
        )

        errors = tf.tensor_scatter_nd_update(
            errors,
            [[sample]],
            [max_error]
        )

    # Print final statistics
    burnin_rate = burnin_accepted / total_burnin_tried if total_burnin_tried > 0 else 0.0
    sampling_rate = sampling_accepted / total_sampling_tried if total_sampling_tried > 0 else 0.0

    print("\nFinal Statistics:", file=logger)
    print(f"Burn-in phase acceptance rate: {burnin_rate:.4f} "
          f"({burnin_accepted}/{total_burnin_tried})",
          file=logger)
    print(f"Sampling phase acceptance rate: {sampling_rate:.4f} "
          f"({sampling_accepted}/{total_sampling_tried})",
          file=logger)

    return samples, acceptance[num_burnin:], errors, (current_theta, current_u)
