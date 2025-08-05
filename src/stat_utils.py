from prettytable import PrettyTable
from statsmodels.stats.multitest import multipletests


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
