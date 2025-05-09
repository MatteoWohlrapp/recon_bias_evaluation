import pandas as pd
from plot_combined import (
    plot_combined_summary,
    plot_combined_summary_v2,
    plot_combined_fairness_classifier,
    plot_combined_segmentation,
    plot_combined_fairness_classifier_with_baseline,
    plot_combined_segmentation_with_baseline,
    plot_combined_fairness_classifier_std_err_with_baseline,
    plot_combined_segmentation_std_err_with_baseline,
)


def evaluate_combined(config, results_dir, name):
    chex_df = pd.read_csv(config["chex_path"])
    ucsf_df = pd.read_csv(config["ucsf_path"])

    combined_df = pd.concat([chex_df, ucsf_df])

    plot_combined_summary(combined_df, results_dir)
    plot_combined_summary_v2(combined_df, results_dir)
    plot_combined_fairness_classifier(combined_df, results_dir)
    plot_combined_segmentation(combined_df, results_dir)

    plot_combined_fairness_classifier_with_baseline(combined_df, results_dir)
    plot_combined_segmentation_with_baseline(combined_df, results_dir)

    plot_combined_fairness_classifier_std_err_with_baseline(combined_df, results_dir)
    plot_combined_segmentation_std_err_with_baseline(combined_df, results_dir)
    chex_df.to_csv(results_dir / f"{name}_chex_results.csv", index=False)
    ucsf_df.to_csv(results_dir / f"{name}_ucsf_results.csv", index=False)
