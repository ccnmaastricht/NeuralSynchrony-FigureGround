"""
This script generates results required for the Quantitative Model Predictions section of the paper. 
It performs a mixed effects analysis to investigate the relationship between the sizes of empirical and model 
Arnold tongues while accounting for subject variability. Results are saved in results/statistics/.
"""
import os
import tomllib
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az

from src.stat_utils import zscore_data
from src.anl_utils import compute_size
from src.sim_utils import generate_condition_space

DATA_PATH = 'results/empirical'


def compute_empirical_sizes(experiment_parameters, condition_space):
    """
    Get the empirical data.

    Returns
    -------
    bat_sizes : array_like
        The bat sizes.
    """

    bat_sizes = np.zeros((experiment_parameters['num_subjects'],
                          experiment_parameters['num_training_sessions']))

    for session in range(experiment_parameters['num_training_sessions']):
        session_path = os.path.join(DATA_PATH, f'session_{session + 1}')
        filename = os.path.join(session_path, 'individual_bats.npy')
        individual_bats = np.load(filename)
        for subject, bat in enumerate(individual_bats):
            bat_sizes[subject, session] = compute_size(bat, *condition_space)

    return bat_sizes


def load_data(config_path, simulation_path):
    """
    Load the data.

    Parameters
    ----------
    config_path : str
        The path to the configuration file.
    simulation_path : str
        The path to the simulation file.

    Returns
    -------
    model_sizes : array_like
        The model sizes.
    empirical_sizes : array_like
        The empirical sizes.
    experiment_parameters : dict
        The experiment parameters.
    """

    # Load configuration
    with open(config_path, 'rb') as f:
        experiment_parameters = tomllib.load(f)

    # Load model learning data
    data = np.load(simulation_path)
    model_sizes = data['arnold_tongue_size']

    # Get empirical learning data
    condition_space = generate_condition_space(experiment_parameters)
    empirical_sizes = compute_empirical_sizes(experiment_parameters,
                                              condition_space)

    return model_sizes, empirical_sizes, experiment_parameters


def prepare_dataframe(model_sizes, empirical_sizes, experiment_parameters):
    """
    Prepare the DataFrame.

    Parameters
    ----------
    model_sizes : array_like
        The model sizes.
    empirical_sizes : array_like
        The empirical sizes.
    experiment_parameters : dict
        The experiment parameters.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame.
    """
    n_subjects = experiment_parameters['num_subjects']
    n_sessions = experiment_parameters['num_training_sessions']
    subject_idx = np.repeat(np.arange(n_subjects), n_sessions)
    session_idx = np.tile(np.arange(n_sessions), n_subjects)

    df = pd.DataFrame(np.hstack((subject_idx[:, None], session_idx[:, None])),
                      columns=['subject', 'session'])

    df["model_size"] = model_sizes.reshape(-1)
    df["empirical_size"] = empirical_sizes.reshape(-1)

    df = zscore_data(df, ["model_size", "empirical_size"])

    return df


def run_mixed_effects_analysis(df, dependent_variable, independent_variable,
                               group_variable, output_path):
    """
    Run a mixed effects analysis and save the results.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame.
    dependent_variable : str
        The dependent variable.
    independent_variable : str
        The independent variable.

    Returns
    -------
    None
    """
    model = bmb.Model(
        f"{dependent_variable} ~ {independent_variable} + (1 | {group_variable})",
        data=df,
        family="gaussian")

    idata = model.fit(draws=2000,
                      tune=2000,
                      target_accept=0.9,
                      idata_kwargs={"log_likelihood": True},
                      progressbar=False)
    az.to_netcdf(idata, output_path)


if __name__ == '__main__':

    config_path = 'config/analysis/experiment_actual.toml'
    model_path = 'results/simulation/learning_simulation.npz'

    model_sizes, empirical_sizes, experiment_parameters = load_data(
        config_path, model_path)

    # Prepare the DataFrame
    df = prepare_dataframe(model_sizes, empirical_sizes, experiment_parameters)

    # Save the dataframe
    df.to_csv('results/empirical/learning.csv')

    # Exclude first two sessions
    df = df[df["session"] > 1]

    # Run the mixed effects analysis for size
    run_mixed_effects_analysis(df, "empirical_size", "model_size", "subject",
                               'results/statistics/mixed_effects_bat_size.nc')
