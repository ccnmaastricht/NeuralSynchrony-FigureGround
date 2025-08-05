"""
This script performs generalized estimating equations (GEE) analysis on the behavioral data.
Results are saved in results/statistics/.
"""
import pickle
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.anl_utils import load_data, get_session_data
from src.stat_utils import extract_pvals


def is_significant(corrected_pvals, variable, cutoff=0.05):
    """
    Check if a variable is significant.

    Parameters
    ----------
    results : statsmodels.genmod.generalized_estimating_equations.GEEResultsWrapper
        The results of the GEE analysis.
    variable : str
        The variable of interest.
    cutoff : float, optional
        The cutoff for the p-value. The default is 0.05.
        
    Returns
    -------
    significant : bool
        True if the variable is significant, False otherwise.
    """
    return corrected_pvals[variable] < cutoff


if __name__ == '__main__':
    # Load data
    data_path = 'data/Experiment.csv'
    try:
        data = load_data(data_path)
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        exit(1)

    # Ignore transfer (final) session
    data = data[data['SessionID'] != 9]

    # Define distribution and covariance structure for GEE
    family = sm.families.Binomial()
    covariance_structure = sm.cov_struct.Exchangeable()

    # Fit first session model
    first_session_data = get_session_data(data, 1)
    model = smf.gee(
        "Correct ~ ContrastHeterogeneity + GridCoarseness + ContrastHeterogeneity*GridCoarseness",
        "SubjectID",
        first_session_data,
        cov_struct=covariance_structure,
        family=family)
    results_first = model.fit()

    print(results_first.summary())

    # Save results of first model
    with open('results/statistics/gee_first.pkl', 'wb') as f:
        pickle.dump(results_first, f)

    # Fit full model
    model = smf.gee(
        "Correct ~ ContrastHeterogeneity + GridCoarseness + SessionID + SessionID*ContrastHeterogeneity + SessionID*GridCoarseness + ContrastHeterogeneity*GridCoarseness",
        "SubjectID",
        data,
        cov_struct=covariance_structure,
        family=family)
    results_full = model.fit()

    print(results_full.summary())

    # Save results of full model
    with open('results/statistics/gee_full.pkl', 'wb') as f:
        pickle.dump(results_full, f)

    corrected_pvals = extract_pvals(results_full)

    if is_significant(corrected_pvals, 'SessionID:ContrastHeterogeneity'):
        # Simple effects of contrast heterogeneity for each session
        for session in range(1, 9):
            session_data = get_session_data(data, session)
            model = smf.gee("Correct ~ ContrastHeterogeneity",
                            "SubjectID",
                            session_data,
                            cov_struct=covariance_structure,
                            family=family)
            results = model.fit()

            # Save results
            with open(
                    f'results/statistics/gee_contrast_heterogeneity_{session}.pkl',
                    'wb') as f:
                pickle.dump(results, f)

    if is_significant(corrected_pvals, 'SessionID:GridCoarseness'):
        # Simple effects of grid coarseness for each session
        for session in range(1, 9):
            session_data = get_session_data(data, session)
            model = smf.gee("Correct ~ GridCoarseness",
                            "SubjectID",
                            session_data,
                            cov_struct=covariance_structure,
                            family=family)
            results = model.fit()

            # Save results
            with open(f'results/statistics/gee_grid_coarseness_{session}.pkl',
                      'wb') as f:
                pickle.dump(results, f)
