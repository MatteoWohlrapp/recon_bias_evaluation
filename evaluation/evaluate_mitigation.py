import pandas as pd
from plot_mitigation import plot_mitigation_combined, performance_evaluation_ucsf_table, performance_evaluation_chex_table


def evaluate_mitigation(config, results_dir, name):

    """fairness_config = config["fairness"]

    csv_standard_ucsf = None
    csv_eodd_ucsf = None
    csv_reweighted_ucsf = None
    csv_standard_chex = None
    csv_eodd_chex = None
    csv_reweighted_chex = None

    if "ucsf" in fairness_config:
        csv_standard_ucsf = pd.read_csv(fairness_config["ucsf"]["csv_standard"])
        csv_eodd_ucsf = pd.read_csv(fairness_config["ucsf"]["csv_eodd"])
        csv_reweighted_ucsf = pd.read_csv(fairness_config["ucsf"]["csv_reweighted"])
    if "chex" in fairness_config:
        csv_standard_chex = pd.read_csv(fairness_config["chex"]["csv_standard"])
        csv_eodd_chex = pd.read_csv(fairness_config["chex"]["csv_eodd"])
        csv_reweighted_chex = pd.read_csv(fairness_config["chex"]["csv_reweighted"])

    csv_standard = pd.concat([csv_standard_ucsf, csv_standard_chex])
    csv_eodd = pd.concat([csv_eodd_ucsf, csv_eodd_chex])
    csv_reweighted = pd.concat([csv_reweighted_ucsf, csv_reweighted_chex])

    plot_mitigation_combined(csv_standard, csv_reweighted, csv_eodd, results_dir, name)"""


    performance_config = config["performance"]

    csv_standard_ucsf = pd.read_csv(performance_config["ucsf"]["csv_standard"])
    csv_standard_ucsf = csv_standard_ucsf[csv_standard_ucsf["acceleration"] == 8]
    csv_eodd_ucsf = pd.read_csv(performance_config["ucsf"]["csv_eodd"])
    csv_reweighted_ucsf = pd.read_csv(performance_config["ucsf"]["csv_reweighted"])

    csv_standard_chex = pd.read_csv(performance_config["chex"]["csv_standard"])
    csv_standard_chex = csv_standard_chex[csv_standard_chex["photon_count"] == 10000]
    csv_eodd_chex = pd.read_csv(performance_config["chex"]["csv_eodd"])
    csv_reweighted_chex = pd.read_csv(performance_config["chex"]["csv_reweighted"]) 

    performance_evaluation_ucsf_table(results_dir, csv_standard_ucsf, csv_eodd_ucsf, csv_reweighted_ucsf, csv_eodd_ucsf.copy())
    performance_evaluation_chex_table(results_dir, csv_standard_chex, csv_eodd_chex, csv_reweighted_chex, csv_eodd_chex.copy())


    
    
