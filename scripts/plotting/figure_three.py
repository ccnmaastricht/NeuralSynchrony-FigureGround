"""
This script creates all panels of the second figure of the paper. 
"""

import os
import tomllib

import numpy as np
from src.plot_utils import colored_heatmap, convert_size, plot_dAIC

BASE_PATH = 'results/figures/figure_three'

os.makedirs(BASE_PATH, exist_ok=True)

with open('config/plotting/figure_three.toml', 'rb') as f:
    figure_parameters = tomllib.load(f)

figsize = convert_size(*figure_parameters['general']['figure_size'])
cmap = figure_parameters['general']['colormap']

# top row - behavioural Arnold tongue (raw) per session
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

# middle row - behavioural Arnold tongue (fitted) per session
for session in range(1, figure_parameters['data']['num_sessions'] + 1):
    filename = os.path.join(BASE_PATH, f'middle_row_{session}')
    fitted_arnold_tongue = np.load(
        f'results/empirical/session_{session}/continuous_bat.npy')

    title = f'Session {session}'
    labels = (title, *figure_parameters['data']['labels'])
    colored_heatmap(fitted_arnold_tongue,
                    figsize=figsize,
                    labels=labels,
                    fontsizes=figure_parameters['general']['fontsizes'],
                    ticks=figure_parameters['data']['ticks'],
                    bounds=figure_parameters['data']['bounds'],
                    colormap=cmap,
                    filename=filename)

# bottom row - simulated Arnold tongue & transfer session results
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

# transfer session results
data = np.load('results/empirical/transfer_model_comparison.npz')
dAIC = data['delta_AIC']
session = np.arange(1, figure_parameters['data']['num_sessions'])

labels = figure_parameters['transfer']['labels']
fontsizes = figure_parameters['general']['fontsizes'][:3]
fill = (figure_parameters['transfer']['fill']['colors'],
        figure_parameters['transfer']['fill']['alpha'],
        figure_parameters['transfer']['fill']['bounds_x'],
        figure_parameters['transfer']['fill']['bounds_y'],
        figure_parameters['transfer']['fill']['labels'],
        figure_parameters['transfer']['fill']['label_x'],
        figure_parameters['transfer']['fill']['label_y'])

filename = os.path.join(BASE_PATH, 'bottom_row_transfer')
plot_dAIC(session,
          dAIC,
          figsize=figsize,
          labels=labels,
          fontsizes=fontsizes,
          line_color=figure_parameters['transfer']['line_color'],
          text_color=figure_parameters['transfer']['text_color'],
          fill=fill,
          filename=filename)
