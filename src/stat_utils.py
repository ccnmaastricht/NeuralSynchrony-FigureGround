import bambi as bmb
import arviz as az
from scipy.stats import zscore
from scipy.special import expit

from prettytable import PrettyTable
from statsmodels.stats.multitest import multipletests


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


def create_subject_index(df):
    """
    Create a subject index mapping from subject IDs to integer indices.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.

    Returns
    -------
    numpy.ndarray
        An array of subject indices.
    int
        The number of subjects.
    """
    unique_subjects = df["SubjectID"].unique()
    num_subjects = len(unique_subjects)
    id_to_index = {s: i for i, s in enumerate(unique_subjects)}
    subject_idx = df["SubjectID"].map(id_to_index).to_numpy()
    return subject_idx, num_subjects


def simulate_correct(df, beta_coefficient, predictor, intercept, sdev_beta,
                     sdev_intercept, subject_index, num_subjects, rng):
    """
    Simulates the "Correct" responses for a given set of parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    beta_coefficient : float
        The fixed effect coefficient for the predictor.
    predictor : pandas.Series
        The predictor variable.
    intercept : float
        The fixed effect intercept.
    sdev_beta : float
        The standard deviation of the random slopes.
    sdev_intercept : float
        The standard deviation of the random intercepts.
    subject_index : numpy.ndarray
        The subject index array.
    num_subjects : int
        The number of subjects.
    rng : np.random.Generator
        The random number generator.

    Returns
    -------
    pandas.DataFrame
        The simulated data with "Correct" responses.
    """

    # subject-level random effects
    random_intercepts = rng.normal(0.0, sdev_intercept,
                                   size=num_subjects)  # random intercepts
    random_slopes = rng.normal(
        0.0, sdev_beta,
        size=num_subjects)  # random slopes for effect of interest

    # linear predictor per trial
    logit = (intercept + random_intercepts[subject_index] +
             (beta_coefficient + random_slopes[subject_index]) * predictor)

    probability = expit(logit)
    y = rng.binomial(1, probability, size=len(probability))

    simulated_df = df.copy()
    simulated_df["Correct"] = y
    return simulated_df


def fit_and_decide(simulated_data,
                   formula,
                   effect_of_interest,
                   draws=800,
                   tune=800,
                   target_accept=0.9,
                   threshold=0.95):
    """
    Fit the model and decide if the effect of interest is "significant" (posterior probability > threshold)
    
    Parameters:
    ----------
    - simulated_data: The data to fit the model on.
    - formula: The formula to use for the model.
    - effect_of_interest: The effect to test for significance.
    - draws: The number of draws to use for fitting.
    - tune: The number of tuning steps to use.
    - target_accept: The target acceptance rate for the sampler.
    - threshold: The threshold for deciding if the effect is significant.

    Returns:
    -------
    - A boolean indicating if the effect of interest is significant.
    """
    model = bmb.Model(formula, data=simulated_data, family="bernoulli")
    idata = model.fit(draws=draws,
                      tune=tune,
                      target_accept=target_accept,
                      progressbar=False)

    beta = idata.posterior[effect_of_interest].values.flatten()
    return (beta > 0).mean() > threshold


def extract_pvals(results, cutoff=0.05, method='holm'):
    """
    Extracts p-values from Wald Chi-Square test and corrects for multiple comparisons.

    Parameters
    ----------
    results : statsmodels.genmod.generalized_estimating_equations.GEEResultsWrapper
        The results of the GEE analysis.
    cutoff : float, optional
        The cutoff for the p-value. The default is 0.05.
    method : str, optional
        The method for multiple testing correction. The default is 'holm'.

    Returns
    -------
    dict
        A dictionary mapping variable names to their corrected p-values.
    """

    family_of_tests = [
        name for name in results.model.exog_names if name != 'Intercept'
    ]

    pvals = []
    for var in family_of_tests:
        pvals.append(results.wald_test(var, scalar=True).pvalue)
    _, corrected = multipletests(pvals, alpha=cutoff, method=method)[:2]
    return dict(zip(family_of_tests, corrected))


def print_wald_chi_square(results):
    """
    Prints a table of Wald Chi-Square statistics for each variable in the model.

    Parameters
    ----------
    results : statsmodels.regression.linear_model.RegressionResultsWrapper
        The results of the GEE model.
    """

    corrected_pvals = extract_pvals(results)

    print('Wald Chi-Square:')
    table = PrettyTable()
    table.field_names = ['Variable', 'Chi-Square', 'p-value']
    for variable, pval in corrected_pvals.items():
        table.add_row([
            variable,
            results.wald_test(variable, scalar=True).statistic, pval
        ])
    print(table)


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
