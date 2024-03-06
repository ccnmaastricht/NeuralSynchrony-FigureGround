"""
This script simulates the first session of the experiment and generate an Arnold Tongue.
Results are saved in results/arnold_tongue.npy and correspond to section X of the paper.
"""
import os
import tomllib
import numpy as np

from src.sim_utils import initialize_simulation_classes, setup_parallel_processing, generate_stimulus_conditions, generate_time_index
from src.anl_utils import order_parameter, compute_coherence, compute_weighted_coherence, expand_matrix

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

    with open('config/analysis/experiment_extended.toml', 'rb') as f:
        parameters['experiment_extended'] = tomllib.load(f)

    return parameters['model'], parameters['stimulus'], parameters[
        'simulation'], parameters['experiment_extended']


def run_block(block, experiment_parameters, simulation_parameters, num_entries,
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
    global arnold_tongue, coherence

    grid_coarseness, contrast_heterogeneity = stimulus_conditions
    model, stimulus_generator = simulation_classes
    sync_index, timepoints = indexing

    np.random.seed(simulation_parameters['random_seed'] + block)

    for condition, (scaling_factor, contrast_range) in enumerate(
            zip(grid_coarseness, contrast_heterogeneity)):
        stimulus = stimulus_generator.generate(
            scaling_factor, contrast_range,
            experiment_parameters['mean_contrast'])
        model.compute_omega(stimulus.flatten())
        state_variables, _ = model.simulate(simulation_parameters)
        synchronization = np.abs(order_parameter(state_variables))
        index = block * experiment_parameters['num_conditions'] + condition
        arnold_tongue[index] = np.mean(synchronization[sync_index])
        lower_index = index * num_entries
        upper_index = lower_index + num_entries
        coherence_placeholder = [
            compute_coherence(state_variables[timepoint])
            for timepoint in timepoints
        ]
        coherence[lower_index:upper_index] = np.mean(coherence_placeholder,
                                                     axis=0)


def run_simulation(experiment_parameters, simulation_parameters, num_entries,
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

    global arnold_tongue, coherence

    # Initialize the Arnold tongue
    arnold_tongue = np.zeros((experiment_parameters['num_blocks'],
                              experiment_parameters['num_conditions']))
    arnold_tongue = Array('d', arnold_tongue.reshape(-1))

    # Initialize the coherence
    coherence = np.zeros(
        (experiment_parameters['num_blocks'],
         experiment_parameters['num_conditions'], num_entries))
    coherence = Array('d', coherence.reshape(-1))

    # Run a batch of blocks in parallel
    for batch in range(simulation_parameters['num_batches']):
        with Pool(experiment_parameters['num_blocks']) as p:
            p.starmap(
                run_block,
                [(block, experiment_parameters, simulation_parameters,
                  num_entries, stimulus_conditions, simulation_classes,
                  indexing)
                 for block in range(batch * simulation_parameters['num_cores'],
                                    (batch + 1) *
                                    simulation_parameters['num_cores'])])

    # Collect simulation results
    arnold_tongue = np.array(arnold_tongue).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_conditions'])

    coherence = np.array(coherence).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_conditions'], num_entries)

    return arnold_tongue, coherence


def run_learning(learning_rate, optimal_psychometric, experiment_parameters,
                 simulation_parameters, num_entries, stimulus_conditions,
                 simulation_classes, indexing):
    """
    Run the learning simulation.

    Parameters
    ----------
    learning_rate : float
        The learning rate.
    num_sessions : int
        The number of sessions.
    experiment_parameters : dict
        The experiment parameters.
    simulation_parameters : dict
        The simulation parameters.
    num_entries : int
        The number of entries.
    stimulus_conditions : tuple
        The stimulus conditions.
    simulation_classes : tuple
        The simulation classes.
    indexing : tuple
        The indexing for synchronization.


    Returns
    -------
    correlation_fits : array_like
        The correlation fits.
    jaccard_fits : array_like
        The Jaccard fits.
    arnold_tongue_size : array_like
        The Arnold tongue size.
    """

    model, stimulus_generator = simulation_classes
    # Set the learning rate and generate the coupling
    model.effective_learning_rate = learning_rate

    # Initialize the arnold tongues
    arnold_tongues = np.zeros((experiment_parameters['num_training_sessions'],
                               experiment_parameters['num_blocks'],
                               experiment_parameters['num_conditions']))

    # Initialize the diagonal
    diagonal = np.ones(model.num_populations)

    # Run the learning simulation
    for session in range(experiment_parameters['num_training_sessions']):
        print(
            f'Running session {session + 1} of {experiment_parameters["num_training_sessions"]}'
        )

        simulation_classes = (model, stimulus_generator)

        # Run the simulation
        arnold_tongue, coherence = run_simulation(experiment_parameters,
                                                  simulation_parameters,
                                                  num_entries,
                                                  stimulus_conditions,
                                                  simulation_classes, indexing)

        # Add the Arnold tongue of the session
        arnold_tongues[session] = arnold_tongue

        # Compute the weighted coherence and update the coupling
        weighted_coherence = compute_weighted_coherence(
            experiment_parameters['num_conditions'],
            experiment_parameters['num_blocks'], num_entries, arnold_tongue,
            coherence, optimal_psychometric)
        weighted_coherence = expand_matrix(weighted_coherence, diagonal)
        model.update_coupling(weighted_coherence)

        return arnold_tongues


if __name__ == '__main__':

    # Load the parameters
    model_parameters, stimulus_parameters, simulation_parameters, experiment_parameters = load_configurations(
    )
    # Derive additional parameters
    num_entries = model_parameters['num_populations'] * (
        model_parameters['num_populations'] - 1) // 2

    # Load learning rates
    crossval_results = np.load('results/simulation/crossval_estimation.npz')
    learning_rates = crossval_results['learning_rate_crossval']
    learning_rate = learning_rates.mean()

    # Load the optimal psychometric curve
    file = 'results/analysis/session_1/optimal_psychometric_crossval.npy'
    optimal_psychometric = np.load(file).mean(axis=0)

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

    # Run learning simulation
    arnold_tongues = run_learning(learning_rate, optimal_psychometric,
                                  experiment_parameters, simulation_parameters,
                                  num_entries, stimulus_conditions,
                                  simulation_classes, indexing)

    # Save the results
    file = 'results/simulation/highres_arnold_tongues.npy'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.save(file, arnold_tongues)
