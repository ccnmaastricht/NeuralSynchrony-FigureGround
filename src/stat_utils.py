import bambi as bmb
import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.special import expit
from prettytable import PrettyTable


def zscore_data(data, columns):
    """
    Computes the z-score for each column in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    columns : list
        The columns to compute the z-score for.

    Returns
    -------
    pandas.DataFrame
        The z-scored data.
    """
    zscored = data.copy()
    zscored[columns] = zscore(data[columns], axis=0)
    return zscored


def print_sample_info(metadata):
    """
    Prints information about the sample.

    Parameters
    ----------
    metadata : pandas.DataFrame
        The metadata of the sample.
    """
    num_samples = metadata.shape[0]
    num_females = metadata['Sex'].value_counts()[' F']
    mean_age = metadata['Age'].mean()
    std_age = metadata['Age'].std().__round__(3)

    print(
        f'{num_samples} particpants ({num_females} female, mean age = {mean_age}, standard deviation = {std_age})'
    )


def one_sided_posterior_prob(idata, predictor, direction='greater'):
    """Calculate the one-sided posterior probability that the samples are greater than zero."""

    beta = idata.posterior[predictor].values.flatten()
    if direction == 'greater':
        prob = np.mean(beta > 0)
    elif direction == 'less':
        prob = np.mean(beta < 0)
    else:
        raise ValueError("Direction must be 'greater' or 'less'.")
    return prob


def odds_ratio_summary(idata, predictor):
    """Calculate the odds ratio summary statistics for a given predictor."""
    beta = idata.posterior[predictor].values.flatten()
    or_mean = np.exp(beta).mean()
    or_low = np.percentile(np.exp(beta), 2.5)
    or_high = np.percentile(np.exp(beta), 97.5)
    return or_mean, or_low, or_high


def posterior_table(idata, predictors, directions):
    """Summarize the model by calculating posterior probabilities for each predictor."""
    table = PrettyTable()
    table.field_names = ["Predictor", "direction", "P"]

    for predictor, direction in zip(predictors, directions):
        prob = one_sided_posterior_prob(idata, predictor, direction)
        table.add_row([predictor, direction, f"{prob:.3f}"])

    return table


def OR_table(idata, predictors):
    """Summarize the model by calculating posterior probabilities and odds ratios for each predictor."""
    table = PrettyTable()
    table.field_names = ["Predictor", "Mean", "Lower (2.5%)", "Upper (97.5%)"]

    for predictor in predictors:
        or_mean, or_low, or_high = odds_ratio_summary(idata, predictor)
        table.add_row(
            [predictor, f"{or_mean:.3f}", f"{or_low:.3f}", f"{or_high:.3f}"])

    return table


def draw_posterior(name, idata, draw_index=None):
    """ 
    Extract posterior draws for a given variable from the inference data.

    Parameters
    ----------
    name : str
        Name of the variable to extract.
    idata : arviz.InferenceData
        Inference data object containing posterior samples.
    draw_index : int, optional
        Specific draw index to extract. If None, returns all draws.
    
    Returns
    -------
    np.ndarray
        Array of posterior draws for the specified variable.
    """
    data_frame = az.extract(idata, var_names=[name]).to_dataframe()
    values = data_frame.iloc[:, -1].to_numpy()
    return values if draw_index is None else float(values[draw_index])


def simulate_dataset_from_draw(idata,
                               df,
                               draw_index,
                               num_subjects=None,
                               rng=None):
    """
    Simulate a dataset from a specific posterior draw of a fitted model.

    Parameters
    ----------
    idata : arviz.InferenceData
        Inference data object containing posterior samples. 
    df : pandas.DataFrame
        Original dataframe with the structure of the data (including SubjectID and predictors).
    draw_index : int
        Index of the posterior draw to use for simulation.
    n_subjects : int, optional
        Number of subjects to include in the simulated dataset. If None, use all subjects.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new generator is created.
    
    Returns
    -------
    pandas.DataFrame
        Simulated dataset with the same structure as df, including a 'Correct' column with simulated outcomes.
    dict
        Dictionary containing the true beta coefficients used in the simulation.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Fixed effects at this posterior draw
    intercept = draw_posterior("Intercept", idata, draw_index)
    beta_contrast_heterogeneity = draw_posterior("ContrastHeterogeneity",
                                                 idata, draw_index)
    beta_grid_coarseness = draw_posterior("GridCoarseness", idata, draw_index)
    beta_interaction = draw_posterior("ContrastHeterogeneity:GridCoarseness",
                                      idata, draw_index)

    # Group-level SDs (diagonal approximation; extend to correlated REs if desired)
    sd_intercept = draw_posterior("1|SubjectID_sigma", idata, draw_index)
    sd_contrast_heterogeneity = draw_posterior(
        "ContrastHeterogeneity|SubjectID_sigma", idata, draw_index)
    sd_grid_coarseness = draw_posterior("GridCoarseness|SubjectID_sigma",
                                        idata, draw_index)
    sd_interaction = draw_posterior(
        "ContrastHeterogeneity:GridCoarseness|SubjectID_sigma", idata,
        draw_index)

    # Choose subjects (preserve trial structure per chosen subjects)
    all_subjects = df["SubjectID"].unique().tolist()
    if num_subjects is None or num_subjects >= len(all_subjects):
        subjects = all_subjects
    else:
        subjects = rng.choice(all_subjects, size=num_subjects,
                              replace=False).tolist()
    df_simulated = df[df["SubjectID"].isin(subjects)].copy()

    # Precompute predictors
    contrast_heterogeneity = df_simulated["ContrastHeterogeneity"].to_numpy()
    grid_coarseness = df_simulated["GridCoarseness"].to_numpy()
    interaction = contrast_heterogeneity * grid_coarseness

    # Draw random effects per subject
    random_effects = {
        subject:
        dict(
            intercept=rng.normal(0, sd_intercept),
            contrast_heterogeneity=rng.normal(0, sd_contrast_heterogeneity),
            grid_coarseness=rng.normal(0, sd_grid_coarseness),
            interaction=rng.normal(0, sd_interaction),
        )
        for subject in subjects
    }

    # Linear predictor and Bernoulli outcomes
    logits = np.empty(len(df_simulated), dtype=float)
    subs_vec = df_simulated["SubjectID"].to_numpy()
    for idx in range(len(df_simulated)):
        subject = subs_vec[idx]
        logits[idx] = (
            intercept + random_effects[subject]["intercept"] +
            (beta_contrast_heterogeneity +
             random_effects[subject]["contrast_heterogeneity"]) *
            contrast_heterogeneity[idx] +
            (beta_grid_coarseness + random_effects[subject]["grid_coarseness"])
            * grid_coarseness[idx] +
            (beta_interaction + random_effects[subject]["interaction"]) *
            interaction[idx])
    probabilities = expit(logits)
    df_simulated["Correct"] = (rng.uniform(size=len(df_simulated))
                               < probabilities).astype(int)
    return df_simulated, dict(
        contrast_heterogeneity=beta_contrast_heterogeneity,
        grid_coarseness=beta_grid_coarseness,
        interaction=beta_interaction)


def analyze_simulated(df_sim,
                      true_betas,
                      prob_thresh=0.95,
                      draws=1000,
                      tune=1000,
                      chains=2,
                      target_accept=0.85):
    # Fit lean model
    model = bmb.Model(
        "Correct ~ 1 + ContrastHeterogeneity * GridCoarseness + (1 + ContrastHeterogeneity * GridCoarseness | SubjectID)",
        data=df_sim,
        family="bernoulli")
    idata = model.fit(draws=draws,
                      tune=tune,
                      chains=chains,
                      target_accept=target_accept,
                      progressbar=False)

    # Pull posterior draws for fixed effects
    posteriors = az.extract(idata,
                            var_names=[
                                "ContrastHeterogeneity", "GridCoarseness",
                                "ContrastHeterogeneity:GridCoarseness"
                            ]).to_dataframe()

    out = {}
    for predictor, col in zip(
        ["contrast_heterogeneity", "grid_coarseness", "interaction"], [
            "ContrastHeterogeneity", "GridCoarseness",
            "ContrastHeterogeneity:GridCoarseness"
        ]):
        posterior_values = posteriors[col].to_numpy()
        true_beta = true_betas[predictor]

        # Directional probability in the true direction
        if true_beta >= 0:
            directional_probability = float((posterior_values > 0).mean())
        else:
            directional_probability = float((posterior_values < 0).mean())

        # Detection criterion: directional prob > 0.95
        detected = (directional_probability > prob_thresh)

        # Type-S error: detected but mean sign opposite to true sign
        mean_sign = np.sign(posterior_values.mean())
        true_sign = np.sign(true_beta) if true_beta != 0 else 0.0
        type_s = detected and (mean_sign != true_sign)

        # Type-M error (only if detected and true != 0): magnitude exaggeration ratio
        type_m = np.nan
        if detected and true_beta != 0:
            type_m = float(abs(posterior_values.mean()) / abs(true_beta))

        out[f"{predictor}_directional_probability"] = directional_probability
        out[f"{predictor}_detected"] = float(detected)
        out[f"{predictor}_typeS"] = float(type_s)
        out[f"{predictor}_typeM"] = type_m
        out[f"{predictor}_post_mean"] = float(posterior_values.mean())
    return out


def summarize_design_analysis(df_results):
    """
    Summarize the results of the design analysis by calculating mean and median statistics for each number of subjects.

    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing the results of the design analysis with columns for number of subjects and various statistics.
    
    Returns
    -------
    pandas.DataFrame
        Summary DataFrame with mean and median statistics for each number of subjects.
    """

    summary = df_results.groupby("num_subjects").agg({
        "contrast_heterogeneity_detected":
        "mean",
        "grid_coarseness_detected":
        "mean",
        "interaction_detected":
        "mean",
        "contrast_heterogeneity_typeS":
        "mean",
        "grid_coarseness_typeS":
        "mean",
        "interaction_typeS":
        "mean",
        "contrast_heterogeneity_typeM":
        "median",
        "grid_coarseness_typeM":
        "median",
        "interaction_typeM":
        "median",
        "contrast_heterogeneity_directional_probability":
        "mean",
        "grid_coarseness_directional_probability":
        "mean",
        "interaction_directional_probability":
        "mean",
    }).rename(
        columns={
            "contrast_heterogeneity_detected":
            "Detected CH",
            "grid_coarseness_detected":
            "Detected GC",
            "interaction_detected":
            "Detected Interaction",
            "contrast_heterogeneity_type S":
            "Type S error CH",
            "grid_coarseness_typeS":
            "Type S error GC",
            "interaction_typeS":
            "Type S error INT",
            "contrast_heterogeneity_typeM":
            "Type M error CH",
            "grid_coarseness_typeM":
            "Type M error GC",
            "interaction_typeM":
            "Type M error Interaction",
            "contrast_heterogeneity_directional_probability":
            "Probability (one-sided) CH",
            "grid_coarseness_directional_probability":
            "Probability (one-sided) GC",
            "interaction_directional_probability":
            "Probability (one-sided) Interaction",
        })
    return summary.round(2)
