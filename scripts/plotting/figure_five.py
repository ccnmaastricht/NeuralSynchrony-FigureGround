"""
This script creates all panels of the fifth figure of the paper. 
"""

import os
import pickle
import tomllib
import arviz as az

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from src.anl_utils import min_max_normalize, weighted_jaccard
from src.plot_utils import fit_barplot, convert_size

BASE_PATH = 'results/figures/figure_five'
os.makedirs(BASE_PATH, exist_ok=True)


def compute_noise_ceiling(individual_bats):
    """
    Compute the noise ceiling within a specific session using the leave-one-out CV approach also used for the model.

    Parameters
    ----------
    individual_bats : array_like
        The individual behavioural Arnold tongues.

    Returns
    -------
    float
        The noise ceiling using the correlation coefficient.
    float
        The noise ceiling using the weighted Jaccard similarity.
    """

    num_subjects = individual_bats.shape[0]
    corrcoefs = np.zeros(num_subjects)
    jaccards = np.zeros(num_subjects)

    for i in range(num_subjects):
        left_out_bat = individual_bats[i].flatten()
        left_out_bat = min_max_normalize(left_out_bat)
        average_bat = np.delete(individual_bats, i,
                                axis=0).mean(axis=0).flatten()
        average_bat = min_max_normalize(average_bat)

        corrcoefs[i] = np.corrcoef(left_out_bat, average_bat)[0, 1]
        jaccards[i] = weighted_jaccard(left_out_bat, average_bat)

    noise_ceiling_corrcoef = np.percentile(corrcoefs,
                                           25), np.percentile(corrcoefs, 75)
    noise_ceiling_jaccard = np.percentile(jaccards,
                                          25), np.percentile(jaccards, 75)

    return noise_ceiling_corrcoef, noise_ceiling_jaccard


def compute_corr_mean_error(correlation_fits):
    """
    Compute the mean and error value on the correlation scale for plotting error bars.

    Parameters
    ----------
    correlation_fits : array_like
        The correlation coefficients.

    Returns
    -------
    array_like
        The mean correlation fit.
    array_like
        The error value to add and subtract from the mean to create the 95% confidence interval.
    """

    # Fisher-z transform the correlation coefficients
    z_corr_fits = np.arctanh(correlation_fits)

    # Compute the mean and standard error on the Fisher-z scale
    mean_z_corr_fit = np.mean(z_corr_fits, axis=0)
    sem_z_corr_fit = np.std(z_corr_fits, axis=0) / np.sqrt(
        z_corr_fits.shape[0])

    # Calculate the 95% confidence interval on the Fisher-z scale
    bound = 1.96 * sem_z_corr_fit

    # Inverse Fisher-z transform the mean and confidence interval to get them back on the correlation scale
    mean_corr_fit = np.tanh(mean_z_corr_fit)
    lower_corr_fit = np.tanh(mean_z_corr_fit - bound)
    upper_corr_fit = np.tanh(mean_z_corr_fit + bound)

    # Calculate the error value as the difference between the mean and the bounds of the confidence interval
    error_value = np.array(
        (upper_corr_fit - mean_corr_fit, mean_corr_fit - lower_corr_fit))

    return mean_corr_fit, error_value


def get_intercept_and_slope(arviz_file_path):
    """
    Get the intercept and slope of the mixed effects model.

    Parameters
    ----------
    picke_file_path : str
        The path to the pickled mixed effects model.

    Returns
    -------
    float
        The intercept of the mixed effects model.
    float
        The slope of the mixed effects model.
    """
    idata = az.from_netcdf(arviz_file_path)

    intercept = idata.posterior['Intercept'].mean().item()
    slope = idata.posterior['model_size'].mean().item()
    return intercept, slope


def mean_and_sem(df, variable):
    """
    Compute the mean and standard error of the mean for a variable per session.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame.
    variable : str
        The variable of interest.

    Returns
    -------
    mean : pandas.Series
        The mean of the variable per session.
    sem : pandas.Series
        The standard error of the mean of the variable per session.
    """
    mean = df.groupby('session')[variable].mean()
    sem = df.groupby('session')[variable].sem()
    return mean, sem


def plot_model_vs_empirical(mean_model_size,
                            mean_empirical_size,
                            sem_empirical_size,
                            intercept,
                            slope,
                            labels,
                            fontsizes,
                            marker_color,
                            figsize,
                            filename=None,
                            dpi=300,
                            filetype='svg'):
    """
    Plot model vs empirical Arnold tongue size.

    Parameters
    ----------
    mean_model_size : pandas.Series
        The mean model sizes per session.
    mean_empirical_size : pandas.Series
        The mean empirical sizes per session.
    sem_empirical_size : pandas.Series
        The standard error of the mean for the empirical sizes per session.
    intercept : float
        The intercept of the mixed effects model.
    slope : float
        The slope of the mixed effects model.
    fig_params : dict
        The figure parameters.
    """
    figure = plt.figure(figsize=figsize,
                        dpi=dpi if filetype != 'svg' else None)

    title, xlabel, ylabel = labels
    title_fontsize, label_fontsize, tick_fontsize = fontsizes

    plt.errorbar(mean_model_size[:2],
                 mean_empirical_size[:2],
                 yerr=sem_empirical_size[:2],
                 fmt='o',
                 color=marker_color[0],
                 capsize=5,
                 capthick=2)

    plt.errorbar(mean_model_size[2:],
                 mean_empirical_size[2:],
                 yerr=sem_empirical_size[2:],
                 fmt='o',
                 color=marker_color[1],
                 capsize=5,
                 capthick=2)

    # Plot the mixed effects model
    min_x = min(mean_model_size) - 0.1 * min(mean_model_size)
    max_x = max(mean_model_size) + 0.1 * max(mean_model_size)
    x = np.linspace(min_x, max_x, 100)
    y = intercept + slope * x
    plt.plot(x, y, color='black')

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)

    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.tight_layout()

    # Save the figure
    if filename is not None:
        filename = f'{filename}.{filetype}'
        plt.savefig(filename, dpi=dpi if filetype != 'svg' else None)
        plt.close()
    else:
        return figure


def plot_sizes(sessions,
               empirical_sizes,
               model_sizes,
               fontsizes,
               labels,
               marker_color,
               figsize,
               filename=None,
               dpi=300,
               filetype='svg'):

    figure = plt.figure(figsize=figsize,
                        dpi=dpi if filetype != 'svg' else None)

    title, xlabel, ylabel, model_legend, empirical_legend = labels
    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize = fontsizes

    plt.plot(sessions,
             model_sizes,
             'o-',
             color='black',
             markerfacecolor=marker_color[0],
             label=model_legend)
    plt.plot(sessions,
             empirical_sizes,
             's-',
             color='black',
             markerfacecolor=marker_color[1],
             label=empirical_legend)

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize)

    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.tight_layout()

    # Save the figure
    if filename is not None:
        filename = f'{filename}.{filetype}'
        plt.savefig(filename, dpi=dpi if filetype != 'svg' else None)
        plt.close()


if __name__ == '__main__':

    # Load the figure parameters
    with open('config/plotting/figure_five.toml', 'rb') as f:
        figure_parameters = tomllib.load(f)

    # Convert the figure size to inches
    figsize = convert_size(*figure_parameters['general']['figure_size'])

    # Load the model fits
    with np.load('results/simulation/learning_simulation.npz') as model_fits:
        correlation_fits = model_fits['correlation_fits']
        jaccard_fits = model_fits['jaccard_fits']

    noise_ceiling_corrcoef = np.zeros((8, 2))
    noise_ceiling_jaccard = np.zeros((8, 2))

    for session in figure_parameters['general']['sessions']:
        individual_bats = np.load(
            f'results/empirical/session_{session}/individual_bats.npy')

        noise_ceiling_corrcoef[session - 1], noise_ceiling_jaccard[
            session - 1] = compute_noise_ceiling(individual_bats)

    # Panel A - correlation fits per session
    filename = os.path.join(BASE_PATH, 'panel_a')

    # Inverse Fisher-z transform the mean and confidence interval to get them back on the correlation scale
    mean_corr_fit, corr_error = compute_corr_mean_error(correlation_fits)

    # Create a bar plot
    fit_barplot(mean_corr_fit,
                corr_error,
                figure_parameters['general']['sessions'],
                noise_ceiling_corrcoef,
                figsize=figsize,
                labels=figure_parameters['panels'][0]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                capsize=figure_parameters['general']['capsize'],
                face_color=figure_parameters['general']['face_color'],
                filename=filename,
                dpi=300,
                filetype='svg')

    # Panel B - Jaccard fits per session
    filename = os.path.join(BASE_PATH, 'panel_b')

    # Compute the mean and standard error on the Jaccard scale
    mean_jaccard_fit = np.mean(jaccard_fits, axis=0)
    sem_jaccard_fit = np.std(jaccard_fits, axis=0) / np.sqrt(
        jaccard_fits.shape[0])

    # Create a bar plot
    fit_barplot(mean_jaccard_fit,
                sem_jaccard_fit,
                figure_parameters['general']['sessions'],
                noise_ceiling_jaccard,
                figsize=figsize,
                labels=figure_parameters['panels'][1]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                capsize=figure_parameters['general']['capsize'],
                face_color=figure_parameters['general']['face_color'],
                filename=filename,
                dpi=300,
                filetype='svg')

    # Panel C - Arnold tongue sizes per session
    filename = os.path.join(BASE_PATH, 'panel_c')

    # Load the data
    df = pd.read_csv('results/empirical/learning.csv')

    sessions = df['session'].unique() + 1
    # compute average model and empirical sizes for each session (averaged over subjects)
    model_sizes = df.groupby('session')['model_size'].mean()
    empirical_sizes = df.groupby('session')['empirical_size'].mean()

    # min-max normalize the sizes
    model_sizes = min_max_normalize(model_sizes)
    empirical_sizes = min_max_normalize(empirical_sizes)

    plot_sizes(sessions, empirical_sizes, model_sizes,
               figure_parameters['panels'][2]['fontsizes'],
               figure_parameters['panels'][2]['labels'],
               figure_parameters['panels'][2]['marker_color'], figsize,
               filename)

    # Panel D - model vs empirical Arnold tongue size
    filename = os.path.join(BASE_PATH, 'panel_d')

    # Get slope and intercept of the mixed effects model
    intercept, slope = get_intercept_and_slope(
        'results/statistics/mixed_effects_bat_size.nc')

    mean_model_size, _ = mean_and_sem(df, 'model_size')
    mean_empirical_size, sem_empirical_size = mean_and_sem(
        df, 'empirical_size')

    plot_model_vs_empirical(mean_model_size, mean_empirical_size,
                            sem_empirical_size, intercept, slope,
                            figure_parameters['panels'][3]['labels'],
                            figure_parameters['general']['fontsizes'],
                            figure_parameters['panels'][3]['marker_color'],
                            figsize, filename)
