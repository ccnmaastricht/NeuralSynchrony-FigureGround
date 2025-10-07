"""
This script performs simulatiosn of the full learning experiment at high resolution of experimental conditions.
Results are saved in results/arnold_tongue.npy.
"""
import os
import tomllib
import numpy as np

from src.sim_utils import initialize_simulation_classes, setup_parallel_processing, generate_stimulus_conditions, generate_time_index
from src.anl_utils import order_parameter, compute_weighted_locking, expand_matrix, compute_firing_rate, convert_to_frequency

from multiprocessing import Pool, Array


def load_configurations():
    """
    Load the model, stimulus, simulation, and experiment parameters.

    Returns
    -------
    model_parameters : dict
        The model parameters.
    stimulus_parameters : dict
        The stimulus parameters.
    simulation_parameters : dict
        The simulation parameters.
    experiment_parameters : dict
        The experiment parameters.
    """
    parameters = {}
    config_files = ['model', 'stimulus', 'simulation']

    for config_file in config_files:
        with open(f'config/simulation/{config_file}.toml', 'rb') as f:
            parameters[config_file] = tomllib.load(f)

    with open('config/analysis/experiment_actual.toml', 'rb') as f:
        parameters['experiment'] = tomllib.load(f)

    return parameters['model'], parameters['stimulus'], parameters[
        'simulation'], parameters['experiment']


def run_block(block, experiment_parameters, simulation_parameters,
              stimulus_conditions, simulation_classes, indexing):
    """
    Run a block of the Arnold tongue. This function is used for parallel processing.

    Parameters
    ----------
    block : int
        The block number.
    experiment_parameters : dict
        The experiment parameters.
    simulation_parameters : dict
        The simulation parameters.
    stimulus_conditions : tuple
        The stimulus conditions.
    simulation_classes : tuple
        The simulation classes.
    indexing : tuple
        The indexing for synchronization.
    """
    global arnold_tongue, effective_firing_rate, intrinsic_firing_rate

    grid_coarseness, contrast_heterogeneity = stimulus_conditions
    model, stimulus_generator = simulation_classes
    sync_index, _ = indexing

    np.random.seed(simulation_parameters['random_seed'] + block)

    for condition, (scaling_factor, contrast_range) in enumerate(
            zip(grid_coarseness, contrast_heterogeneity)):

        stimulus = stimulus_generator.generate(
            scaling_factor, contrast_range,
            experiment_parameters['mean_contrast'])

        model.compute_omega(stimulus.flatten())
        intrinsic_frequency = convert_to_frequency(model.omega)

        state_variables, _ = model.simulate(simulation_parameters)
        synchronization = np.abs(order_parameter(state_variables))
        effective_frequency = compute_firing_rate(
            state_variables, sync_index, simulation_parameters['time_step'])

        index = block * experiment_parameters['num_conditions'] + condition
        arnold_tongue[index] = np.mean(synchronization[sync_index])
        effective_firing_rate[index] = np.mean(effective_frequency)
        intrinsic_firing_rate[index] = np.mean(intrinsic_frequency)


def run_simulation(experiment_parameters, simulation_parameters,
                   stimulus_conditions, simulation_classes, indexing):
    """
    Run the simulation.

    Parameters
    ----------
    experiment_parameters : dict
        The experiment parameters.
    simulation_parameters : dict
        The simulation parameters.
    stimulus_conditions : tuple
        The stimulus conditions.
    simulation_classes : tuple
        The simulation classes.
    indexing : tuple
        The indexing for synchronization.
        
    Returns
    -------
    arnold_tongue : array_like
        The Arnold tongue.
    """

    global arnold_tongue, effective_firing_rate, intrinsic_firing_rate

    # Initialize the Arnold tongue
    arnold_tongue = np.zeros((experiment_parameters['num_blocks'],
                              experiment_parameters['num_conditions']))
    arnold_tongue = Array('d', arnold_tongue.reshape(-1))

    # Initialize the effective firing rate
    effective_firing_rate = np.zeros((experiment_parameters['num_blocks'],
                                      experiment_parameters['num_conditions']))
    effective_firing_rate = Array('d', effective_firing_rate.reshape(-1))

    # Initialize the intrinsic firing rate
    intrinsic_firing_rate = np.zeros((experiment_parameters['num_blocks'],
                                      experiment_parameters['num_conditions']))
    intrinsic_firing_rate = Array('d', intrinsic_firing_rate.reshape(-1))

    # Run batches of blocks in parallel
    for batch in range(simulation_parameters['num_batches']):
        with Pool(experiment_parameters['num_blocks']) as p:
            p.starmap(
                run_block,
                [(block, experiment_parameters, simulation_parameters,
                  stimulus_conditions, simulation_classes, indexing)
                 for block in range(batch * simulation_parameters['num_cores'],
                                    (batch + 1) *
                                    simulation_parameters['num_cores'])])

    # Collect simulation results
    arnold_tongue = np.array(arnold_tongue).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_conditions'])

    effective_firing_rate = np.array(effective_firing_rate).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_conditions'])

    intrinsic_firing_rate = np.array(intrinsic_firing_rate).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_conditions'])

    return arnold_tongue, effective_firing_rate, intrinsic_firing_rate


if __name__ == '__main__':

    # Load the parameters
    model_parameters, stimulus_parameters, simulation_parameters, experiment_parameters = load_configurations(
    )

    # Initialize the model and stimulus generator
    simulation_classes = initialize_simulation_classes(model_parameters,
                                                       stimulus_parameters)

    # Set up parallel processing
    simulation_parameters, experiment_parameters = setup_parallel_processing(
        simulation_parameters, experiment_parameters)

    # Set up the stimulus conditions
    stimulus_conditions = generate_stimulus_conditions(experiment_parameters)

    # Set up the synchronization index and timepoint
    indexing = generate_time_index(simulation_parameters)

    # Run simulation
    arnold_tongues, effective_firing_rates, intrinsic_firing_rates = run_simulation(
        experiment_parameters, simulation_parameters, stimulus_conditions,
        simulation_classes, indexing)

    # Save the results
    arnold_tongues = arnold_tongues.reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_grid_coarseness'],
        experiment_parameters['num_contrast_heterogeneity'])
    file = 'results/simulation/first_session_arnold_tongues.npy'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.save(file, arnold_tongues)

    effective_firing_rates = effective_firing_rates.reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_grid_coarseness'],
        experiment_parameters['num_contrast_heterogeneity'])
    file = 'results/simulation/first_session_effective_firing_rates.npy'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.save(file, effective_firing_rates)

    intrinsic_firing_rates = intrinsic_firing_rates.reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_grid_coarseness'],
        experiment_parameters['num_contrast_heterogeneity'])
    file = 'results/simulation/first_session_intrinsic_firing_rates.npy'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.save(file, intrinsic_firing_rates)
