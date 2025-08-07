"""
This script creates all panels of the fourth figure of the paper. 
"""

import os
import tomllib

import numpy as np
import matplotlib.pyplot as plt
from src.plot_utils import colored_heatmap, convert_size, plot_dAIC

BASE_PATH = 'results/figures/figure_four'

os.makedirs(BASE_PATH, exist_ok=True)

with open('config/plotting/figure_four.toml', 'rb') as f:
    figure_parameters = tomllib.load(f)

figsize = convert_size(*figure_parameters['general']['figure_size'])
cmap = figure_parameters['general']['colormap']

# Top row - behavioural Arnold tongue (raw) per session
for session in range(1, figure_parameters['data']['num_sessions'] + 1):
    filename = os.path.join(BASE_PATH, f'top_row_{session}')
    behavioural_arnold_tongue = np.load(
        f'results/empirical/session_{session}/average_bat.npy')

    title = f'Session {session}'
    labels = (title, *figure_parameters['data']['labels'])
    colored_heatmap(behavioural_arnold_tongue,
                    figsize=figsize,
                    labels=labels,
                    fontsizes=figure_parameters['general']['fontsizes'],
                    ticks=figure_parameters['data']['ticks'],
                    bounds=figure_parameters['data']['bounds'],
                    colormap=cmap,
                    filename=filename)

# Middle row - behavioural Arnold tongue (fitted) per session
for session in range(1, figure_parameters['data']['num_sessions'] + 1):
    filename = os.path.join(BASE_PATH, f'middle_row_{session}')
    fitted_arnold_tongue = np.load(
        f'results/empirical/session_{session}/continuous_bat.npy')

    psychometric_parameters = np.load(
        f'results/empirical/session_{session}/optimal_psychometric_parameters.npy'
    )

    title = f'Session {session}'
    labels = (title, *figure_parameters['data']['labels'])
    heatmap = colored_heatmap(
        fitted_arnold_tongue,
        figsize=figsize,
        labels=labels,
        fontsizes=figure_parameters['general']['fontsizes'],
        ticks=figure_parameters['data']['ticks'],
        bounds=figure_parameters['data']['bounds'],
        colormap=cmap)

    # Add threshold line (75% correct)
    contrast_heterogeneity = np.linspace(
        figure_parameters['data']['min_contrast_heterogeneity'],
        figure_parameters['data']['max_contrast_heterogeneity'],
        figure_parameters['data']['num_contrast_heterogeneity'])
    grid_coarseness = -(
        psychometric_parameters[1] * contrast_heterogeneity +
        psychometric_parameters[2]) / psychometric_parameters[0]

    ax = heatmap.gca()
    ax.plot(contrast_heterogeneity, grid_coarseness, 'k--')
    xticks, yticks = figure_parameters['data']['ticks']
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[-1], yticks[0])

    heatmap.tight_layout()
    filename = f'{filename}.svg'
    heatmap.savefig(filename)
    plt.close(heatmap)

# Bottom row - simulated Arnold tongue & transfer session results
simulated_arnold_tongue = np.load(
    'results/simulation/highres_arnold_tongues.npy').mean(axis=1)

for session in range(figure_parameters['model']['num_sessions']):
    filename = os.path.join(BASE_PATH, f'bottom_row_{session + 1}')

    title = f'Session {session + 1}'
    labels = (title, *figure_parameters['model']['labels'])
    colored_heatmap(simulated_arnold_tongue[session],
                    figsize=figsize,
                    labels=labels,
                    fontsizes=figure_parameters['general']['fontsizes'],
                    ticks=figure_parameters['model']['ticks'],
                    bounds=figure_parameters['model']['bounds'],
                    colormap=cmap,
                    filename=filename)
