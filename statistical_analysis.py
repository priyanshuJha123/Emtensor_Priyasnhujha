from scipy.stats import ttest_ind
from scipy.stats import f_oneway

def perform_ttest(group1, group2):
    """Perform a t-test between two groups."""
    return ttest_ind(group1, group2)

def perform_anova(*groups):
    """Perform ANOVA to compare multiple groups."""
    return f_oneway(*groups)
