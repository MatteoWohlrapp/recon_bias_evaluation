import pandas as pd
from plot_mitigation import plot_mitigation_combined, performance_evaluation_ucsf_table, performance_evaluation_chex_table, plot_combined_fairness_summary, plot_fairness_performance_tradeoff, plot_combined_mitigation_summary


def evaluate_mitigation(config, results_dir, name):

    fairness_config = config["fairness"]

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

    plot_mitigation_combined(csv_standard, csv_reweighted, csv_eodd, csv_eodd.copy(), results_dir, name)


    performance_config = config["performance"]

    csv_standard_ucsf_performance = pd.read_csv(performance_config["ucsf"]["csv_standard"])
    csv_standard_ucsf_performance = csv_standard_ucsf_performance[csv_standard_ucsf_performance["acceleration"] == 8]
    csv_eodd_ucsf_performance = pd.read_csv(performance_config["ucsf"]["csv_eodd"])
    csv_reweighted_ucsf_performance = pd.read_csv(performance_config["ucsf"]["csv_reweighted"])

    csv_standard_chex_performance = pd.read_csv(performance_config["chex"]["csv_standard"])
    csv_standard_chex_performance = csv_standard_chex_performance[csv_standard_chex_performance["photon_count"] == 10000]
    csv_eodd_chex_performance = pd.read_csv(performance_config["chex"]["csv_eodd"])
    csv_reweighted_chex_performance = pd.read_csv(performance_config["chex"]["csv_reweighted"]) 

    performance_evaluation_ucsf_table(results_dir, csv_standard_ucsf_performance, csv_eodd_ucsf_performance, csv_reweighted_ucsf_performance, csv_eodd_ucsf_performance.copy())
    performance_evaluation_chex_table(results_dir, csv_standard_chex_performance, csv_eodd_chex_performance, csv_reweighted_chex_performance, csv_eodd_chex_performance.copy())

    fairness_config = config["fairness"]
    csv_standard_chex_fairness = pd.read_csv(fairness_config["chex"]["csv_standard"])
    csv_standard_ucsf_fairness = pd.read_csv(fairness_config["ucsf"]["csv_standard"])

    combined_df = pd.concat([csv_standard_chex_fairness, csv_standard_ucsf_fairness])

    plot_combined_fairness_summary(combined_df, results_dir)

    
    # --- Performance Dataframes ---
    # load UCSF
    ucsf_performance_standard_df = pd.read_csv(performance_config["ucsf"]["csv_standard"])
    ucsf_performance_standard_df = ucsf_performance_standard_df[ucsf_performance_standard_df["acceleration"] == 8]
    ucsf_performance_eodd_df = pd.read_csv(performance_config["ucsf"]["csv_eodd"])
    ucsf_performance_reweighted_df = pd.read_csv(performance_config["ucsf"]["csv_reweighted"])
    ucsf_performance_adv_df = pd.read_csv(performance_config["ucsf"]["csv_eodd"]) # TODO: add adv

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
    chex_performance_adv_df = pd.read_csv(performance_config["chex"]["csv_eodd"]) # TODO: add adv

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
    ucsf_fairness_adv_df = pd.read_csv(fairness_config["ucsf"]["csv_eodd"]) # TODO: add adv

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
    chex_fairness_adv_df = pd.read_csv(fairness_config["chex"]["csv_eodd"]) # TODO: add adv

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
    plot_fairness_performance_tradeoff(performance_df, fairness_df, results_dir)

    plot_combined_mitigation_summary(fairness_df, results_dir)