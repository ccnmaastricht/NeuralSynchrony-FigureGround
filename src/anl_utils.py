import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.integrate import simps


def load_data(path):
    """
    Load the behavioral data.

    Returns
    -------
    data : pandas.DataFrame
        The behavioral data.
    """
    data = pd.read_csv(path)
    return data


def get_session_data(data, session):
    """
    Get the data of a specific session.

    Parameters
    ----------
    data : pandas.DataFrame
        The behavioral data.
    session : int
        The session number.

    Returns
    -------
    session_data : pandas.DataFrame
        The data of the session.
    """
    session_data = data[data['SessionID'] == session]
    return session_data


def get_subject_data(data, subject):
    """
    Get the data of a specific subject.

    Parameters
    ----------
    data : pandas.DataFrame
        The behavioral data.
    subject : int
        The subject number.

    Returns
    -------
    subject_data : pandas.DataFrame
        The data of the subject.
    """
    subject_data = data[data['SubjectID'] == subject]
    return subject_data


def order_parameter(theta):
    """
    Compute the order parameter of a set of phases.

    Parameters
    ----------
    theta : array_like
        The phases.

    Returns
    ------- 
    float (complex)
        The order parameter.
    """

    return np.mean(np.exp(1j * theta), axis=1)


def compute_phase_difference(theta, condense=True):
    """
    Compute the phase difference between a set of phases.

    Parameters
    ----------
    theta : array_like
        The phases.
    condense : bool (optional)
        Whether to condense the phase difference matrix. The default is True.

    Returns
    -------
    array_like
        The phase difference.
    """
    theta = theta.T
    phase_difference = np.exp(1j * (theta[:, None] - theta[None, :]))

    if condense:
        upper_triangle = np.triu_indices(phase_difference.shape[0], k=1)
        phase_difference = phase_difference[upper_triangle]
    return phase_difference


def compute_size(arnold_tongue, grid_coarseness, contrast_heterogeneity):
    """
    Compute the size of the Arnold tongue as the volume under its surface.

    Parameters
    ----------
    arnold_tongue : array_like
        The Arnold tongue.
    grid_coarseness : array_like
        The grid coarseness.
    contrast_heterogeneity : array_like
        The contrast heterogeneity.

    Returns
    -------
    float
        The size of the Arnold tongue.
    """

    return simps(simps(arnold_tongue, contrast_heterogeneity), grid_coarseness)


def condense_matrix(matrix):
    """
    Condense a symmetric matrix.

    Parameters
    ----------
    matrix : array_like
        The matrix.

    Returns
    ------- 
    array_like
        The condensed matrix.
    """

    diagonal = np.diag(matrix)
    matrix = matrix - np.diag(diagonal)
    return squareform(matrix), diagonal


def expand_matrix(matrix, diagonal):
    """
    Expand a condensed matrix.

    Parameters
    ----------
    matrix : array_like
        The condensed matrix.
    diagonal : array_like
        The diagonal of the original matrix.

    Returns
    ------- 
    array_like
        The expanded matrix.
    """
    diagonal = np.diag(diagonal)
    return squareform(matrix) + diagonal


def welford_update(mean, count, new_value):
    """
    Update the mean and count of a set of values.

    Parameters
    ----------
    mean : float
        The current mean.
    count : int
        The current count.
    new_value : float
        The new value.

    Returns
    ------- 
    float
        The updated mean.
    int
        The updated count.
    """
    delta = new_value - mean
    mean += delta / count
    return mean


def psychometric_function(predictors, *parameters, chance_level=0.5):
    """
    Compute an n-dimensional psychometric function.

    Parameters
    ----------
    predictors : array_like
        The predictors.
    parameters : array_like
        The parameters.
    chance_level : float
        The chance level.

    Returns
    -------
    float
        The probability of a correct response.
    """
    logit = np.dot(parameters, predictors)
    probability = (1 - chance_level) / (1 + np.exp(-logit)) + chance_level
    return probability


def weighted_jaccard(X, Y):
    """
    Compute weighted Jaccard similarity between two sets of elements.

    Parameters
    ----------
    X : array_like
        The first set of elements.
    Y : array_like
        The second set of elements.

    Returns
    -------
    float
        The weighted Jaccard similarity.
    """
    minimum = np.minimum(X.flatten(), Y.flatten())
    maximum = np.maximum(X.flatten(), Y.flatten())

    numerator = np.sum(minimum)
    denominator = np.sum(maximum)

    return numerator / denominator


def min_max_normalize(X):
    """
    Normalize an array between 0 and 1.

    Parameters
    ----------
    X : array_like
        The array to normalize.

    Returns
    -------
    array_like
        The normalized array.
    """
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def compute_weighted_locking(num_conditions, num_blocks, num_entries,
                             arnold_tongue, locking,
                             optimal_psychometric_parameters):
    """
    Compute the weighted locking.  

    Parameters
    ----------
    counts_tuple : tuple
        The number of blocks, conditions, and entries.
    measurements : array_like
        The Arnold tongue and locking.
    optimal_psychometric_parameters : array_like
        The optimal psychometric parameters.

    Returns
    -------
    array_like
        The weighted locking.
    """

    predictors = np.ones((2, num_conditions))
    weighted_locking = np.zeros(num_entries)

    for block in range(num_blocks):
        predictors[0] = arnold_tongue[block]
        probability_correct = psychometric_function(
            predictors, *optimal_psychometric_parameters)

        probability_correct = np.tile(probability_correct, (num_entries, 1)).T
        weighted_locking = welford_update(weighted_locking, block + 1,
                                          (probability_correct *
                                           locking[block]).mean(axis=0))

    return weighted_locking


def convert_to_frequency(omega):
    """
    Convert angular frequency to frequency.

    Parameters
    ----------
    omega : array_like
        The angular frequency.
    
    Returns
    -------
    float
        The frequency in Hz.
    """
    return omega / (2 * np.pi)


def compute_firing_rate(state_variables, time_step):
    """
    Compute the firing rate from state variables (i.e., of each oscillator per time step).

    Parameters
    ----------
    state_variables : array_like
        The state variables of the oscillators.
    time_step : float
        The time step of the simulation.

    Returns
    -------
    array_like
        The firing rate in Hz of each oscillator.
    """
    angular_frequency = np.diff(state_variables,
                                axis=0).mean(axis=0) / time_step
    return convert_to_frequency(angular_frequency)
