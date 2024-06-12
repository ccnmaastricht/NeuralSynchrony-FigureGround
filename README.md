# Neural Synchrony for Figure Ground Segregation

Analysis and simulation code accompanying the article:

Karimian, A., Roberts, M., De Weerd, P., & Senden, M. (n.d.). Adaptive Synchronization of Neural Gamma Oscillations Mediates Figure Ground Perception in Texture Stimuli. *Manuscript in preparation*.

## Abstract
Gamma oscillations are ubiquitous in visual cortex, but their role in perception has been questioned because their frequencies are feature dependent. Drawing on the theory of weakly coupled oscillators (TWCO), we demonstrate that feature-dependence is crucial for gamma's functional role in perception. Two factors determine synchronization behaviour among coupled oscillators: dissimilarity of oscillation frequencies and coupling strength. In early visual cortex these factors relate to local features and physical distance between image elements, respectively. We manipulate these factors in a texture segregation experiment and show that performance follows TWCO predictions both qualitatively and quantitatively, as formalized in a computational model. Furthermore, learning-induced changes in the model's synchronization behaviour predict changes in participants' performance in a perceptual learning experiment. These results suggest that feature-dependence of gamma oscillations is critical for a synchronization-based mechanism of visual scene segmentation. Our findings offer insights into the functional role of gamma oscillations in the brain.

# Repository Overview

## Directory Structure
The project is organized into the following directories:
```bash
├── config
│   ├── analysis
│   ├── plotting
│   └── simulation
├── data
├── notebooks
├── results
├── scripts
│   ├── analysis
│   ├── info
│   ├── plotting
│   ├── simulation
│   └── statistics
└── src
```

## config
This directory contains TOML configuration files for the analysis, plotting, and simulation scripts. These files specify the parameters used by the scripts and allow for easy configuration of the project.

Note that for reasons of reproducibility, the random seed is fixed in the `simulation.toml` config file. This ensures that the results of the simulations can be reproduced exactly.

## data
This directory contains the data used in the project. The Snakemake workflow of the project automatically downloads CSV data files from Zenodo and places them in this directory. The data includes human psychophysics data on figure ground segregation in texture stimuli.

Karimian, M., Mark, R., De Weerd, P., & Senden, M. (2024). Human Psychophysics Dataset on Figure Ground Segregation in Texture Stimuli [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10817187

## notebooks
This directory contains Jupyter notebooks for exploring and reporting results as well as for generating additional figures and performing additional statistical analyses. These notebooks provide an interactive environment for analyzing data and visualizing results.

## results
This directory contains the results generated by the scripts. Results are organized into subdirectories following the same structure as the organization of scripts into subdirectories.

## scripts
This directory contains scripts for simulations, analysis, statistics, plotting, and querying system information. Each script is designed to perform a specific task within the Snakemake workflow. The scripts are organized into subdirectories based on their functionality.

## src
This directory contains Python source code for the project. The __init__.py file is used to mark the directory as a Python package. The source code includes classes and functions for model simulations, analyzing data, statistics, and generating figures.

# Workflow
The workflow of the project is defined in the Snakefile. The Snakefile specifies a set of rules that define how to generate the desired results from the input data. The workflow uses the Snakemake workflow management system to automatically execute the rules and generate the results.

The workflow consists of the following steps:

1. Download the data from Zenodo.
2. Run system information queries to gather information about the system used to run the workflow.
3. Perform statistical analyses on human psychophysics data using generalized estimating equations (GEE).
4. Pre-process human psychophysics data to obtain behavioural Arnold tongues and estimate optimal psychometric parameters for each session.
5. Evaluate assumption of local learning by analysing transfer session results.
7. Explore the parameter space of the model using parameter exploration.
8. Estimate the learning rate of the model using cross-validation.
9. Simulate the learning experiment using the estimated learning rate.
10. Test the model predictions against the human psychophysics data.
11. Generate figures to visualize the results.

## Running the Workflow

There are two ways to run the workflow:

1. **Locally:** To run the workflow locally, navigate to the root directory of the project and execute the following command:
```bash
snakemake --cores [number_of_cores]
```
Replace `[number_of_cores]` with the desired number of CPU cores to be used for the workflow. This command will execute the workflow and generate the results in the `results` directory.

Note that model simulations involve parallel processes implemented with the `multiprocessing` tools. The `config/simulation/simulation.toml` file specifies the number of CPU cores that should be used. If the machine you run this workflow on has fewer cores than specified in the config file, the workflow will still run. However, all of your machine's cores will be used, and the number of blocks of the simulations will be adjusted to be an integer multiple of the number of used cores. It is recommended to adjust the number of cores in the config file and in the `--cores` flag if you would like to use fewer cores.

For example, if you want to run the workflow using 10 cores, execute the following command:
```bash
snakemake --cores 10
```
This will run the workflow using 10 CPU cores and generate the results in the `results` directory.

2. **Docker container:** The workflow can also be run as a Docker container. To do this, build the Docker image using the following command:
```bash
docker build -t adaptive-synchronization .
```
Then, run the container using the following command:
```bash
docker run --rm -it -v $(pwd)/results:/workflow/results adaptive-synchronization snakemake --cores 30
```
This will run the workflow using 30 CPU cores and generate the results in the `results` directory.

Note that the `-v` flag maps the results directory from the host machine to the /workflow/results directory inside the container, allowing the workflow to write its results to the host machine.. The `--rm` flag removes the container after it has finished running.

The Dockerfile used to build the image is included in the root directory of the project. It is based on the latest Snakemake image and includes all the necessary dependencies for running the workflow. The `CMD` in the Dockerfile runs the Snakemake workflow with 30 cores by default. However, this can be overridden by specifying a different number of cores when running the container.
