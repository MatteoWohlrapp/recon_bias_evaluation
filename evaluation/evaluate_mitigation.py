import pandas as pd
from plot_mitigation import plot_fairness_change_mitigation, table_performance_evaluation_ucsf_mitigation, table_performance_evaluation_chex_mitigation, table_fairness_evaluation_standard, plot_fairness_performance_tradeoff, table_fairness_evaluation_mitigation


def evaluate_mitigation(config, results_dir, name):

    """
    This function evaluates the mitigation and plots the results.
    For the GR, we have the following tables and plots: 
    - Tables: 
        - Summary from IDP: Additional bias introduced by the reconstruction in the standard model (for each sensitive attribute)
        - Performance: Change in performance of the standard model vs. the mitigated models (for each dataset x each model)
        - Fairness: Change in fairness of the mitigated models vs. the standard model (for each mitigation x each sensitive attribute)
    - Plots: 
        - Fairness change: Increase in fairness of the mitigated models vs. the standard model (average and each pathology)
        - Fairness performance tradeoff: Change in performance vs. fairness for each mitigation (for each mitigation x each sensitive attribute x fairness metrics)
    """

    fairness_config = config["fairness"]

    csv_standard_ucsf = None
    csv_reweighted_ucsf = None
    csv_eodd_ucsf = None
    csv_adv_ucsf = None

    csv_standard_chex = None
    csv_eodd_chex = None
    csv_reweighted_chex = None
    csv_adv_chex = None

    csv_standard_ucsf = pd.read_csv(fairness_config["ucsf"]["csv_standard"])
    csv_eodd_ucsf = pd.read_csv(fairness_config["ucsf"]["csv_eodd"])
    csv_reweighted_ucsf = pd.read_csv(fairness_config["ucsf"]["csv_reweighted"])
    csv_adv_ucsf = pd.read_csv(fairness_config["ucsf"]["csv_adv"])

    csv_standard_chex = pd.read_csv(fairness_config["chex"]["csv_standard"])
    csv_eodd_chex = pd.read_csv(fairness_config["chex"]["csv_eodd"])
    csv_reweighted_chex = pd.read_csv(fairness_config["chex"]["csv_reweighted"])
    csv_adv_chex = pd.read_csv(fairness_config["chex"]["csv_adv"])

    csv_standard = pd.concat([csv_standard_ucsf, csv_standard_chex])
    csv_eodd = pd.concat([csv_eodd_ucsf, csv_eodd_chex])
    csv_reweighted = pd.concat([csv_reweighted_ucsf, csv_reweighted_chex])
    csv_adv = pd.concat([csv_adv_ucsf, csv_adv_chex])

    plot_fairness_change_mitigation(csv_standard, csv_reweighted, csv_eodd, csv_adv, results_dir, name)  # Fairness change plot


    performance_config = config["performance"]

    csv_standard_ucsf_performance = pd.read_csv(performance_config["ucsf"]["csv_standard"])
    csv_standard_ucsf_performance = csv_standard_ucsf_performance[csv_standard_ucsf_performance["acceleration"] == 8]
    csv_reweighted_ucsf_performance = pd.read_csv(performance_config["ucsf"]["csv_reweighted"])
    csv_eodd_ucsf_performance = pd.read_csv(performance_config["ucsf"]["csv_eodd"])
    csv_adv_ucsf_performance = pd.read_csv(performance_config["ucsf"]["csv_adv"])

    csv_standard_chex_performance = pd.read_csv(performance_config["chex"]["csv_standard"])
    csv_standard_chex_performance = csv_standard_chex_performance[csv_standard_chex_performance["photon_count"] == 10000]
    csv_reweighted_chex_performance = pd.read_csv(performance_config["chex"]["csv_reweighted"]) 
    csv_eodd_chex_performance = pd.read_csv(performance_config["chex"]["csv_eodd"])
    csv_adv_chex_performance = pd.read_csv(performance_config["chex"]["csv_adv"])

    table_performance_evaluation_ucsf_mitigation(results_dir, csv_standard_ucsf_performance, csv_eodd_ucsf_performance, csv_reweighted_ucsf_performance, csv_adv_ucsf_performance) # Table performance
    table_performance_evaluation_chex_mitigation(results_dir, csv_standard_chex_performance, csv_eodd_chex_performance, csv_reweighted_chex_performance, csv_adv_chex_performance) # Table performance

    fairness_config = config["fairness"]
    csv_standard_chex_fairness = pd.read_csv(fairness_config["chex"]["csv_standard"])
    csv_standard_ucsf_fairness = pd.read_csv(fairness_config["ucsf"]["csv_standard"])

    combined_df = pd.concat([csv_standard_chex_fairness, csv_standard_ucsf_fairness])

    table_fairness_evaluation_standard(combined_df, results_dir) # Table summary of IDP (added bias in standard model)

    
    # --- Performance Dataframes ---
    # load UCSF
    ucsf_performance_standard_df = pd.read_csv(performance_config["ucsf"]["csv_standard"])
    ucsf_performance_standard_df = ucsf_performance_standard_df[ucsf_performance_standard_df["acceleration"] == 8]
    ucsf_performance_eodd_df = pd.read_csv(performance_config["ucsf"]["csv_eodd"])
    ucsf_performance_reweighted_df = pd.read_csv(performance_config["ucsf"]["csv_reweighted"])
    ucsf_performance_adv_df = pd.read_csv(performance_config["ucsf"]["csv_adv"]) 

    # Add columns for dataset and mitigation type
    ucsf_performance_standard_df["dataset"] = "ucsf"
    ucsf_performance_standard_df["mitigation"] = "standard"
    ucsf_performance_eodd_df["dataset"] = "ucsf"
    ucsf_performance_eodd_df["mitigation"] = "eodd"
    ucsf_performance_reweighted_df["dataset"] = "ucsf"
    ucsf_performance_reweighted_df["mitigation"] = "reweighted"
    ucsf_performance_adv_df["dataset"] = "ucsf"
    ucsf_performance_adv_df["mitigation"] = "adv"

    chex_performance_standard_df = pd.read_csv(performance_config["chex"]["csv_standard"])
    chex_performance_standard_df = chex_performance_standard_df[chex_performance_standard_df["photon_count"] == 10000]
    chex_performance_eodd_df = pd.read_csv(performance_config["chex"]["csv_eodd"])
    chex_performance_reweighted_df = pd.read_csv(performance_config["chex"]["csv_reweighted"])
    chex_performance_adv_df = pd.read_csv(performance_config["chex"]["csv_adv"]) 

    # Add columns for dataset and mitigation type
    chex_performance_standard_df["dataset"] = "chex"
    chex_performance_standard_df["mitigation"] = "standard"
    chex_performance_eodd_df["dataset"] = "chex"
    chex_performance_eodd_df["mitigation"] = "eodd"
    chex_performance_reweighted_df["dataset"] = "chex"
    chex_performance_reweighted_df["mitigation"] = "reweighted"
    chex_performance_adv_df["dataset"] = "chex"
    chex_performance_adv_df["mitigation"] = "adv"

    # Combine all performance dataframes
    performance_df = pd.concat([ucsf_performance_standard_df, ucsf_performance_eodd_df, ucsf_performance_reweighted_df, ucsf_performance_adv_df,
                                chex_performance_standard_df, chex_performance_eodd_df, chex_performance_reweighted_df, chex_performance_adv_df])


    # --- Fairness Dataframes ---
    ucsf_fairness_standard_df = pd.read_csv(fairness_config["ucsf"]["csv_standard"])
    ucsf_fairness_eodd_df = pd.read_csv(fairness_config["ucsf"]["csv_eodd"])
    ucsf_fairness_reweighted_df = pd.read_csv(fairness_config["ucsf"]["csv_reweighted"])
    ucsf_fairness_adv_df = pd.read_csv(fairness_config["ucsf"]["csv_adv"]) 

    # Add columns for dataset and mitigation type
    ucsf_fairness_standard_df["dataset"] = "ucsf"
    ucsf_fairness_standard_df["mitigation"] = "standard"
    ucsf_fairness_eodd_df["dataset"] = "ucsf"
    ucsf_fairness_eodd_df["mitigation"] = "eodd"
    ucsf_fairness_reweighted_df["dataset"] = "ucsf"
    ucsf_fairness_reweighted_df["mitigation"] = "reweighted"
    ucsf_fairness_adv_df["dataset"] = "ucsf"
    ucsf_fairness_adv_df["mitigation"] = "adv"

    chex_fairness_standard_df = pd.read_csv(fairness_config["chex"]["csv_standard"])
    chex_fairness_eodd_df = pd.read_csv(fairness_config["chex"]["csv_eodd"])
    chex_fairness_reweighted_df = pd.read_csv(fairness_config["chex"]["csv_reweighted"])
    chex_fairness_adv_df = pd.read_csv(fairness_config["chex"]["csv_adv"]) 

    # Add columns for dataset and mitigation type
    chex_fairness_standard_df["dataset"] = "chex"
    chex_fairness_standard_df["mitigation"] = "standard"
    chex_fairness_eodd_df["dataset"] = "chex"
    chex_fairness_eodd_df["mitigation"] = "eodd"
    chex_fairness_reweighted_df["dataset"] = "chex"
    chex_fairness_reweighted_df["mitigation"] = "reweighted"
    chex_fairness_adv_df["dataset"] = "chex"
    chex_fairness_adv_df["mitigation"] = "adv"

    # Combine all fairness dataframes
    fairness_df = pd.concat([ucsf_fairness_standard_df, ucsf_fairness_eodd_df, ucsf_fairness_reweighted_df, ucsf_fairness_adv_df,
                            chex_fairness_standard_df, chex_fairness_eodd_df, chex_fairness_reweighted_df, chex_fairness_adv_df])
    
    # --- Plot Fairness Performance Tradeoff ---
    plot_fairness_performance_tradeoff(performance_df, fairness_df, results_dir) # plot fairness performance tradeoff

    table_fairness_evaluation_mitigation(fairness_df, results_dir) # Table mitigation summary