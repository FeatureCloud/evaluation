import warnings

from classification.BRCA.centralized_brca import perform_centralized_analysis_brca
from classification.BRCA.individual_central_test_results import perform_individual_central_test_analysis_brca
from classification.BRCA.individual_local_test_results import perform_individual_local_test_analysis_brca
from classification.ILPD.baseline_ilpd import perform_centralized_analysis_ilpd
from classification.ILPD.individual_central_test_results import perform_individual_central_test_analysis_ilpd
from classification.ILPD.individual_local_test_results import perform_individual_local_test_analysis_ilpd
from classification.grouped_class_boxplots import perform_grouped_boxplot_analysis_classification
from regression.boston.centralized_boston import perform_centralized_analysis_boston
from regression.boston.individual_central_test import perform_individual_central_test_analysis_boston
from regression.boston.individual_local_test import perform_individual_local_test_analysis_boston
from regression.diabetes.centralized_diabetes import perform_centralized_analysis_diabetes
from regression.diabetes.individual_central_test import perform_individual_central_test_analysis_diabetes
from regression.diabetes.individual_local_test import perform_individual_local_test_analysis_diabetes
from regression.grouped_regr_boxplots import perform_grouped_boxplot_analysis_regression


def main():
    warnings.filterwarnings("ignore")
    print("Start analysis...")
    # Create classification results
    # Centralized
    perform_centralized_analysis_brca()
    perform_centralized_analysis_ilpd()
    # Individual, evaluated on central test data
    perform_individual_central_test_analysis_brca()
    perform_individual_central_test_analysis_ilpd()
    # Individual, evaluated on local test data
    perform_individual_local_test_analysis_brca()
    perform_individual_local_test_analysis_ilpd()

    # Create regression results
    # Centralized
    perform_centralized_analysis_boston()
    perform_centralized_analysis_diabetes()
    # Individual, evaluated on central test data
    perform_individual_central_test_analysis_boston()
    perform_individual_central_test_analysis_diabetes()
    # Individual, evaluated on local test data
    perform_individual_local_test_analysis_boston()
    perform_individual_local_test_analysis_diabetes()

    # Create final grouped boxplots (Paper plots)
    perform_grouped_boxplot_analysis_classification()
    perform_grouped_boxplot_analysis_regression()
    print("Analysis finished!")


if __name__ == "__main__":
    main()
