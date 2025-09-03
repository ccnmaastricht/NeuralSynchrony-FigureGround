import os

session_ids = ["{}".format(i) for i in range(1, 10)]

rule all:
    input:
        ["data/Experiment.csv",
        "results/info/system.toml",
        "results/statistics/gee_full.pkl",
        "results/statistics/mixed_effects_bat_size.pkl",
         "results/empirical/learning.csv",
         "results/simulation/first_session_arnold_tongues.npy",
         "results/simulation/first_session_firing_rates.npy"
         "results/simulation/parameter_space_exploration.npz",
         "results/simulation/crossval_estimation.npz",
         "results/simulation/learning_simulation.npz",
         "results/simulation/highres_arnold_tongues.npy",
         "results/simulation/highres_firing_rates.npy",
         "results/figures/figure_two/panel_d.svg",
         "results/figures/figure_five/panel_d.svg"]

rule download_data:
    output:
        "data/Experiment.csv"
    shell:
        "zenodo_get https://doi.org/10.5281/zenodo.10817187 -o data"

rule run_system_info:
    output:
        "results/info/system.toml"
    shell:
        "python -m scripts.info.system"

rule run_gee_accuracy:
    input:
        "data/Experiment.csv"
    output:
        "results/statistics/gee_full.pkl"
    shell:
        "python -m scripts.statistics.gee_accuracy"

rule run_behavioral_arnold_tongue:
    input:
        "data/Experiment.csv"
    output:
        expand("results/empirical/session_{session}/optimal_psychometric_parameters.npy", session=session_ids) +
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/continuous_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/individual_bats.npy", session=session_ids)
    shell:
        "python -m scripts.analysis.behavioral_arnold_tongue"

rule run_first_session_simulations:
    output:
        ["results/simulation/first_session_arnold_tongues.npy",
        "results/simulation/first_session_firing_rates.npy"]
    shell:
        "python -m scripts.simulation.first_session_simulations"

rule run_parameter_exploration:
    input:
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids)
    output:
        "results/simulation/parameter_space_exploration.npz"
    shell:
        "python -m scripts.simulation.parameter_exploration"

rule run_crossval_estimation:
    input:
        expand("results/empirical/session_{session}/individual_bats.npy", session=session_ids)
    output:
        "results/simulation/crossval_estimation.npz"
    shell:
        "python -m scripts.simulation.crossval_estimation"

rule run_crossval_prediction:
    input:
        ["results/simulation/crossval_estimation.npz"] +
        expand("results/empirical/session_{session}/individual_bats.npy", session=session_ids)
    output:
        "results/simulation/learning_simulation.npz"
    shell:
        "python -m scripts.simulation.crossval_prediction"

rule run_high_resolution_simulations:
    input:
        "results/simulation/crossval_estimation.npz"
    output:
        "results/simulation/highres_arnold_tongues.npy"
    shell:
        "python -m scripts.simulation.high_resolution_simulations"

rule test_model_predictions:
    input:
        ["results/simulation/learning_simulation.npz"] + 
        expand("results/empirical/session_{session}/individual_bats.npy", session=session_ids)
    output:
        ["results/statistics/mixed_effects_bat_size.pkl",
        "results/empirical/learning.csv"]
    shell:
        "python -m scripts.statistics.quantitative_model_predictions"

rule create_figure_two:
    input:
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/continuous_bat.npy", session=session_ids) +
        ["results/simulation/highres_arnold_tongues.npy"]
    output:
        ["results/figures/figure_two/panel_a.svg",
        "results/figures/figure_two/panel_b.svg",
        "results/figures/figure_two/panel_c.svg",
        "results/figures/figure_two/panel_d.svg"]
    shell:
        "python -m scripts.plotting.figure_two"

rule create_figure_three:
    input:
        "results/simulation/parameter_space_exploration.npz"
    output:
        "results/figures/figure_three/figure_three.svg"
    shell:
        "python -m scripts.plotting.figure_three"

rule run_figure_four:
    input:
        "results/simulation/highres_arnold_tongues.npy" +
        expand("results/empirical/session_{session}/average_bat.npy", session=session_ids) +
        expand("results/empirical/session_{session}/continuous_bat.npy", session=session_ids)
    output:
        "results/figures/figure_four/bottom_row_transfer.svg"
    shell:
        "python -m scripts.plotting.figure_four"

rule run_figure_five:
    input:
        ["results/simulation/learning_simulation.npz",
        "results/empirical/learning.csv",
        "results/statistics/mixed_effects_bat_size.pkl"] +
        expand("results/empirical/session_{session}/individual_bats.npy", session=session_ids)
    output:
        ["results/figures/figure_five/panel_a.svg",
        "results/figures/figure_five/panel_b.svg",
        "results/figures/figure_five/panel_c.svg",
        "results/figures/figure_five/panel_d.svg"]
    shell:
        "python -m scripts.plotting.figure_five"

