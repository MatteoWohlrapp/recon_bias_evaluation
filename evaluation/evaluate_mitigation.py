import pandas as pd
from plot_mitigation import plot_mitigation_chex

def evaluate_mitigation(config, results_dir, name):
    csv_standard = config["csv_standard"]
    csv_mitigated = config["csv_mitigated"]

    standard_df = pd.read_csv(csv_standard)
    mitigated_df = pd.read_csv(csv_mitigated)

    if config["dataset"] == "chex":
        plot_mitigation_chex(standard_df, mitigated_df, results_dir, name)
    else:
        raise ValueError(f"Dataset {config['dataset']} not supported")
