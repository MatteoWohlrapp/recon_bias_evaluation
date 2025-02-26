import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

model_colors = {
    "unet": "#3089A2",
    "pix2pix": "#E6C321",
    "sde": "#EB0007",
}


def plot_combined_fairness_classifier(combined_results, results_dir):

    interpreter_configs = {
        "metrics": ["delta-EODD-bootstrapped", "delta-EOP-bootstrapped"],
        "x_labels": ["EODD", "EOP"],
    }

    interpreters = [
        "ec",
        "cardiomegaly",
        "lung-opacity",
        "lung-lesion",
        "edema",
        "consolidation",
        "pneumonia",
        "atelectasis",
        "pneumothorax",
        "pleural-effusion",
        "pleural-other",
        "fracture",
        "ttype",
        "ttype",
    ]

    models = ["unet", "pix2pix", "sde"]
    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    # Calculate averages across all interpreters
    plot_data = []

    for metric in interpreter_configs["metrics"]:
        # Group by model, attribute, and metric to calculate means
        metric_data = combined_results[
            (combined_results["metric"] == metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"].isin(models))
        ].copy()

        # Take absolute value before grouping
        metric_data["value"] = metric_data["value"].abs()
        avg_data = (
            metric_data.groupby(["model", "attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        avg_data["metric_type"] = interpreter_configs["x_labels"][
            interpreter_configs["metrics"].index(metric)
        ]

        plot_data.append(avg_data)

    plot_data = pd.concat(plot_data)

    # Calculate y limits
    y_min = plot_data["value"].min()
    y_max = plot_data["value"].max()

    y_min = np.floor(y_min / 0.05) * 0.05
    y_max = np.ceil(y_max / 0.05) * 0.05

    g = sns.FacetGrid(
        plot_data,
        col="attribute",
        col_order=["gender", "age", "ethnicity"],
        height=6,
        aspect=1.2,
        ylim=(y_min, y_max),
    )

    def plot_bars(data, **kwargs):
        ax = plt.gca()
        unique_metrics = data["metric_type"].unique()
        n_metrics = len(unique_metrics)
        n_models = len(model_labels)
        width = 0.15
        bar_gap = 0.005

        for y in np.arange(y_min, y_max + 0.05, 0.05):
            ax.axhline(
                y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0
            )

        if ax.get_position().x0 > 0.1:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))

        for i, (model, color) in enumerate(model_colors.items()):
            model_data = data[data["model"] == model]
            x_positions = np.arange(n_metrics) * 0.6 + (i - (n_models - 1) / 2) * (
                width + bar_gap
            )

            values = []
            for metric_type in unique_metrics:
                if metric_type == "EODD":
                    value = model_data[
                        model_data["metric"] == "delta-EODD-bootstrapped"
                    ]["value"].iloc[0]
                else:  # EOP
                    value = model_data[
                        model_data["metric"] == "delta-EOP-bootstrapped"
                    ]["value"].iloc[0]
                values.append(value)

            plt.bar(
                x_positions,
                values,
                width=width,
                color=color,
                label=model_labels[model],
                zorder=2,
            )

        plt.xticks(np.arange(n_metrics) * 0.6, unique_metrics)
        plt.setp(ax.get_xticklabels(), weight="bold")

    g.map_dataframe(plot_bars)

    col_names = {"gender": "Gender", "age": "Age", "ethnicity": "Race"}
    g.set_titles(template="{col_name}")

    # Add column titles and make them bold
    for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
        ax.set_title(title, fontweight="bold", pad=20, fontsize=20)

    # Add y-axis label to the leftmost plot with bold font
    g.axes[0, 0].set_ylabel("Additional Bias", fontweight="bold")

    # Remove legends from all subplots
    for ax in g.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Save the main plot
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_average_classifier.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

    # Create separate legend figure
    plt.figure(figsize=(8, 1))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=model_labels[model])
        for model, color in model_colors.items()
    ]

    # Create horizontal legend
    plt.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def plot_combined_segmentation(combined_results, results_dir):

    interpreter_configs = {
        "metrics": ["delta-SER-bootstrapped", "delta-delta-dice-bootstrapped"],
        "x_labels": ["SER", "Delta Dice"],
    }

    interpreters = [
        "dice",
    ]

    models = ["unet", "pix2pix", "sde"]
    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    # Calculate averages across all interpreters
    plot_data = []

    for metric in interpreter_configs["metrics"]:
        # Group by model, attribute, and metric to calculate means
        metric_data = combined_results[
            (combined_results["metric"] == metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"].isin(models))
        ].copy()

        # Take absolute value before grouping
        metric_data["value"] = metric_data["value"].abs()
        avg_data = (
            metric_data.groupby(["model", "attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        avg_data["metric_type"] = interpreter_configs["x_labels"][
            interpreter_configs["metrics"].index(metric)
        ]

        plot_data.append(avg_data)

    plot_data = pd.concat(plot_data)

    # Calculate y limits
    y_min = plot_data["value"].min()
    y_max = plot_data["value"].max()

    y_min = np.floor(y_min / 0.05) * 0.05
    y_max = np.ceil(y_max / 0.05) * 0.05

    g = sns.FacetGrid(
        plot_data,
        col="attribute",
        col_order=["gender", "age"],
        height=6,
        aspect=1.2,
        ylim=(y_min, y_max),
    )

    def plot_bars(data, **kwargs):
        ax = plt.gca()
        unique_metrics = data["metric_type"].unique()
        n_metrics = len(unique_metrics)
        n_models = len(model_labels)
        width = 0.15
        bar_gap = 0.005

        for y in np.arange(y_min, y_max + 0.05, 0.05):
            ax.axhline(
                y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0
            )

        if ax.get_position().x0 > 0.1:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))

        for i, (model, color) in enumerate(model_colors.items()):
            model_data = data[data["model"] == model]
            x_positions = np.arange(n_metrics) * 0.6 + (i - (n_models - 1) / 2) * (
                width + bar_gap
            )

            values = []
            for metric_type in unique_metrics:
                if metric_type == "SER":
                    value = model_data[
                        model_data["metric"] == "delta-SER-bootstrapped"
                    ]["value"].iloc[0]
                else:  # Delta Dice
                    value = model_data[
                        model_data["metric"] == "delta-delta-dice-bootstrapped"
                    ]["value"].iloc[0]
                values.append(value)

            plt.bar(
                x_positions,
                values,
                width=width,
                color=color,
                label=model_labels[model],
                zorder=2,
            )

        plt.xticks(np.arange(n_metrics) * 0.6, unique_metrics)
        plt.setp(ax.get_xticklabels(), weight="bold")

    g.map_dataframe(plot_bars)

    col_names = {"gender": "Gender", "age": "Age"}
    g.set_titles(template="{col_name}")

    # Add column titles and make them bold
    for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
        ax.set_title(title, fontweight="bold", pad=20, fontsize=20)

    # Add y-axis label to the leftmost plot with bold font
    g.axes[0, 0].set_ylabel("Additional Bias", fontweight="bold")

    # Remove legends from all subplots
    for ax in g.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Save the main plot
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_average_segmentation.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

    # Create separate legend figure
    plt.figure(figsize=(8, 1))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=model_labels[model])
        for model, color in model_colors.items()
    ]

    # Create horizontal legend
    plt.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def plot_combined_summary(combined_results, results_dir):

    latex_path = "latex/templates/combined_fairness_colors.tex"
    file_name = latex_path.split("/")[-1].split(".")[0]
    with open(latex_path, "r") as file:
        latex_content = file.read()

    new_latex_content = ""

    # Color definitions for positive and negative values
    colors = {
        "positive": {
            "significant": "E6C321",  # p < 0.05
            "marginal": "F1D892",  # 0.05 <= p < 0.1
        },
        "negative": {
            "significant": "3089A2",  # p < 0.05
            "marginal": "93C1C9",  # 0.05 <= p < 0.1
        },
    }

    interpreter_metrics = {
        "ec": ("age", "gender", "ethnicity"),
        "cardiomegaly": ("age", "gender", "ethnicity"),
        "lung-opacity": ("age", "gender", "ethnicity"),
        "lung-lesion": ("age", "gender", "ethnicity"),
        "edema": ("age", "gender", "ethnicity"),
        "consolidation": ("age", "gender", "ethnicity"),
        "pneumonia": ("age", "gender", "ethnicity"),
        "atelectasis": ("age", "gender", "ethnicity"),
        "pneumothorax": ("age", "gender", "ethnicity"),
        "pleural-effusion": ("age", "gender", "ethnicity"),
        "pleural-other": ("age", "gender", "ethnicity"),
        "fracture": ("age", "gender", "ethnicity"),
        "tgrade": ("age", "gender"),
        "ttype": ("age", "gender"),
    }
    interpreter_metric_labels = {
        "ec": "EC",
        "cardiomegaly": "Cardiomegaly",
        "lung-opacity": "Lung Opacity",
        "lung-lesion": "Lung Lesion",
        "edema": "Edema",
        "consolidation": "Consolidation",
        "pneumonia": "Pneumonia",
        "atelectasis": "Atelectasis",
        "pneumothorax": "Pneumothorax",
        "pleural-effusion": "Pleural Effusion",
        "pleural-other": "Pleural Other",
        "fracture": "Fracture",
        "tgrade": "Tumor Grade",
        "ttype": "Tumor Type",
    }

    models = ["baseline", "unet", "pix2pix", "sde"]
    metrics = ["delta-EODD", "delta-EOP"]

    segmentation_metrics = ["delta-SER", "delta-delta-dice"]

    attributes = ["gender", "age", "ethnicity"]
    attribute_mapping = {
        "gender": "gender",
        "age": "age",
        "ethnicity": "race",
    }

    attribute_contents = []

    for attribute in attributes:
        attribute_content = latex_content.replace(
            "-attribute-", attribute_mapping[attribute]
        )
        classifier_content = ""
        for interpreter, interpreter_attributes in interpreter_metrics.items():
            content = ""
            if attribute in interpreter_attributes:
                content += f"\\multicolumn{{1}}{{l|}}{{\\textbf{{{interpreter_metric_labels[interpreter]}}}}}"

                for model in models:
                    for metric in metrics:
                        if model == "baseline":
                            metric = metric.replace("delta-", "")

                        filtered_results = combined_results[
                            (combined_results["metric"] == metric)
                            & (combined_results["model"] == model)
                            & (combined_results["interpreter"] == interpreter)
                            & (combined_results["attribute"] == attribute)
                        ]
                        result = filtered_results["value"].iloc[0]
                        if len(filtered_results) == 0:
                            continue

                        if model == "baseline":
                            content += f"& {result:.3f}"
                        else:
                            filtered_p_values = combined_results[
                                (combined_results["metric"] == f"{metric}-p-value")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == interpreter)
                                & (combined_results["attribute"] == attribute)
                            ]

                            filtered_std_errors = combined_results[
                                (combined_results["metric"] == f"{metric}-std-err")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == interpreter)
                                & (combined_results["attribute"] == attribute)
                            ]

                            p_value = filtered_p_values["value"].iloc[0]
                            std_error = filtered_std_errors["value"].iloc[0]

                            color = None
                            if result > 0:
                                if p_value < 0.05:
                                    color = colors["positive"]["significant"]
                                elif p_value < 0.1:
                                    color = colors["positive"]["marginal"]
                            if result < 0:
                                if p_value < 0.05:
                                    color = colors["negative"]["significant"]
                                elif p_value < 0.1:
                                    color = colors["negative"]["marginal"]

                            if color is not None:
                                if abs(std_error) > abs(result):
                                    if result > 0:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{\\phantom{{-}}{result:.3f} $\\pm$ {std_error:.2f}}}}}"
                                    else:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{{result:.3f} $\\pm$ {std_error:.2f}}}}}"
                                else:
                                    if result > 0:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{\\phantom{{-}}{result:.3f} $\\pm$ {std_error:.2f}}}"
                                    else:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{{result:.3f} $\\pm$ {std_error:.2f}}}"
                            else:
                                if abs(std_error) > abs(result):
                                    if result > 0:
                                        content += f"& \\textbf{{\\phantom{{-}}{result:.3f} $\\pm$ {std_error:.2f}}}"
                                    else:
                                        content += f"& \\textbf{{{result:.3f} $\\pm$ {std_error:.2f}}}"
                                else:
                                    if result > 0:
                                        content += f"& \\phantom{{-}}{result:.3f} $\\pm$ {std_error:.2f}"
                                    else:
                                        content += (
                                            f"& {result:.3f} $\\pm$ {std_error:.2f}"
                                        )

                classifier_content += content + "\\\\ \n"

        attribute_content = attribute_content.replace(
            "-classifier-", classifier_content
        )

        if attribute in ["gender", "age"]:
            segmentation_content = """
                &                                   &                                            &                                            &                                                   &                                            &                                                   &                                            &                                                   \\\\ \\hline
            \\multicolumn{1}{l|}{} & \\multicolumn{1}{c}{\\textbf{SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta$ Dice}} & \\multicolumn{1}{c}{\\textbf{$\\Delta$ SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta \\Delta$ Dice}} & \\multicolumn{1}{c}{\\textbf{$\\Delta$ SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta \\Delta$ Dice}} & \\multicolumn{1}{c}{\\textbf{$\\Delta$ SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta \\Delta$ Dice}} \\\\ \\hline
                """
            content = ""
            content += f"\\multicolumn{{1}}{{l|}}{{\\textbf{{Segmentation}}}}"

            for model in models:
                for metric in segmentation_metrics:
                    if model == "baseline":
                        if metric == "delta-SER":
                            metric = "SER"
                        elif metric == "delta-delta-dice":
                            metric = "delta-dice"

                    filtered_results = combined_results[
                        (combined_results["metric"] == metric)
                        & (combined_results["model"] == model)
                        & (combined_results["interpreter"] == "dice")
                        & (combined_results["attribute"] == attribute)
                    ]
                    result = filtered_results["value"].iloc[0]
                    if len(filtered_results) == 0:
                        continue

                    if model == "baseline":
                        content += f"& {result:.3f}"
                    else:
                        filtered_p_values = combined_results[
                            (combined_results["metric"] == f"{metric}-p-value")
                            & (combined_results["model"] == model)
                            & (combined_results["interpreter"] == "dice")
                            & (combined_results["attribute"] == attribute)
                        ]

                        filtered_std_errors = combined_results[
                            (combined_results["metric"] == f"{metric}-std-err")
                            & (combined_results["model"] == model)
                            & (combined_results["interpreter"] == "dice")
                            & (combined_results["attribute"] == attribute)
                        ]

                        p_value = filtered_p_values["value"].iloc[0]
                        std_error = filtered_std_errors["value"].iloc[0]

                        color = None
                        if result > 0:
                            if p_value < 0.05:
                                color = colors["positive"]["significant"]
                            elif p_value < 0.1:
                                color = colors["positive"]["marginal"]
                        if result < 0:
                            if p_value < 0.05:
                                color = colors["negative"]["significant"]
                            elif p_value < 0.1:
                                color = colors["negative"]["marginal"]

                        if color is not None:
                            if abs(std_error) > abs(result):
                                if result > 0:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{\\phantom{{-}}{result:.3f} $\\pm$ {std_error:.2f}}}}}"
                                else:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{{result:.3f} $\\pm$ {std_error:.2f}}}}}"
                            else:
                                if result > 0:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{\\phantom{{-}}{result:.3f} $\\pm$ {std_error:.2f}}}"
                                else:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{{result:.3f} $\\pm$ {std_error:.2f}}}"
                        else:
                            if abs(std_error) > abs(result):
                                if result > 0:
                                    content += f"& \\textbf{{\\phantom{{-}}{result:.3f} $\\pm$ {std_error:.2f}}}"
                                else:
                                    content += f"& \\textbf{{{result:.3f} $\\pm$ {std_error:.2f}}}"
                            else:
                                if result > 0:
                                    content += f"& \\phantom{{-}}{result:.3f} $\\pm$ {std_error:.2f}"
                                else:
                                    content += f"& {result:.3f} $\\pm$ {std_error:.2f}"

            segmentation_content += "\n" + content + "\\\\"

            attribute_content += segmentation_content

        attribute_content += """    
        \end{tabular} 
        } % End of resizebox -legend-
        \end{table}"""

        legend = """\\\\[0.3cm]
            \\begin{tabular}{llllllll} 
            \\cellcolor[HTML]{E6C321} & $+$, $p < 0.05$ & \\cellcolor[HTML]{F1D892} &$+$, $0.05 \\leq p < 0.1$ & \\cellcolor[HTML]{3089A2} & $-$, $p < 0.05$  & \\cellcolor[HTML]{93C1C9} & $-$, $0.05 \\leq p < 0.1$ \\\\
            \\multicolumn{8}{l}{\\textbf{Bold} indicates standard error larger than absolute effect size}
            \\end{tabular}"""
        attribute_content = attribute_content.replace("-legend-", legend)

        attribute_contents.append(attribute_content)

    for content in attribute_contents:
        new_latex_content += content

    with open(os.path.join(results_dir, f"{file_name}.tex"), "w") as file:
        file.write(new_latex_content)

def plot_combined_fairness_classifier_with_baseline(combined_results, results_dir):
    interpreter_configs = {
        "metrics": ["delta-EODD-bootstrapped", "delta-EOP-bootstrapped"],
        "baseline_metrics": ["EODD-bootstrapped", "EOP-bootstrapped"],
        "x_labels": ["EODD", "EOP"],
    }

    interpreters = [
        "ec",
        "cardiomegaly",
        "lung-opacity",
        "lung-lesion",
        "edema",
        "consolidation",
        "pneumonia",
        "atelectasis",
        "pneumothorax",
        "pleural-effusion",
        "pleural-other",
        "fracture",
        "ttype",
        "ttype",
    ]

    models = ["unet", "pix2pix", "sde"]
    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    # Calculate averages across all interpreters
    plot_data = []
    baseline_data = []

    # Process delta metrics for models
    for metric in interpreter_configs["metrics"]:
        # Group by model, attribute, and metric to calculate means
        metric_data = combined_results[
            (combined_results["metric"] == metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"].isin(models))
        ].copy()

        # Take absolute value before grouping
        metric_data["value"] = metric_data["value"].abs()
        avg_data = (
            metric_data.groupby(["model", "attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        avg_data["metric_type"] = interpreter_configs["x_labels"][
            interpreter_configs["metrics"].index(metric)
        ]

        plot_data.append(avg_data)

    # Process baseline metrics
    for baseline_metric, x_label in zip(interpreter_configs["baseline_metrics"], interpreter_configs["x_labels"]):
        baseline_metric_data = combined_results[
            (combined_results["metric"] == baseline_metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"] == "baseline")
        ].copy()

        # Take absolute value before grouping
        baseline_metric_data["value"] = baseline_metric_data["value"].abs()
        baseline_avg_data = (
            baseline_metric_data.groupby(["attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        baseline_avg_data["metric_type"] = x_label
        
        baseline_data.append(baseline_avg_data)

    plot_data = pd.concat(plot_data)
    baseline_data = pd.concat(baseline_data)

    # Calculate y limits
    all_values = pd.concat([plot_data["value"], baseline_data["value"]])
    y_min = all_values.min()
    y_max = all_values.max()

    y_min = np.floor(y_min / 0.05) * 0.05
    y_max = np.ceil(y_max / 0.05) * 0.05

    g = sns.FacetGrid(
        plot_data,
        col="attribute",
        col_order=["gender", "age", "ethnicity"],
        height=6,
        aspect=1.2,
        ylim=(y_min, y_max),
    )

    def plot_bars_with_baseline(data, **kwargs):
        ax = plt.gca()
        unique_metrics = data["metric_type"].unique()
        n_metrics = len(unique_metrics)
        n_models = len(model_labels)
        width = 0.15
        bar_gap = 0.005

        for y in np.arange(y_min, y_max + 0.05, 0.05):
            ax.axhline(
                y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0
            )

        if ax.get_position().x0 > 0.1:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))

        # Get current attribute
        current_attribute = data["attribute"].iloc[0]
        
        # Plot bars for each model
        for i, (model, color) in enumerate(model_colors.items()):
            model_data = data[data["model"] == model]
            x_positions = np.arange(n_metrics) * 0.6 + (i - (n_models - 1) / 2) * (
                width + bar_gap
            )

            values = []
            for metric_type in unique_metrics:
                if metric_type == "EODD":
                    value = model_data[
                        model_data["metric"] == "delta-EODD-bootstrapped"
                    ]["value"].iloc[0]
                else:  # EOP
                    value = model_data[
                        model_data["metric"] == "delta-EOP-bootstrapped"
                    ]["value"].iloc[0]
                values.append(value)

            plt.bar(
                x_positions,
                values,
                width=width,
                color=color,
                label=model_labels[model],
                zorder=2,
            )
        
        # Plot baseline lines
        for j, metric_type in enumerate(unique_metrics):
            baseline_value = baseline_data[
                (baseline_data["attribute"] == current_attribute) & 
                (baseline_data["metric_type"] == metric_type)
            ]["value"].iloc[0]
            
            # Calculate x position for the line (centered on the metric position)
            x_start = j * 0.6 - 0.25
            x_end = j * 0.6 + 0.25
            
            # Plot the baseline line
            plt.plot(
                [x_start, x_end], 
                [baseline_value, baseline_value], 
                color="black", 
                linestyle="--", 
                linewidth=2, 
                zorder=3
            )

        plt.xticks(np.arange(n_metrics) * 0.6, unique_metrics)
        plt.setp(ax.get_xticklabels(), weight="bold")

    g.map_dataframe(plot_bars_with_baseline)

    col_names = {"gender": "Gender", "age": "Age", "ethnicity": "Race"}
    g.set_titles(template="{col_name}")

    # Add column titles and make them bold
    for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
        ax.set_title(title, fontweight="bold", pad=20, fontsize=20)

    # Add y-axis label to the leftmost plot with bold font
    g.axes[0, 0].set_ylabel("Bias Metric Value", fontweight="bold")

    # Remove legends from all subplots
    for ax in g.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Save the main plot
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_with_baseline_classifier.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

    # Create separate legend figure
    plt.figure(figsize=(10, 1.5))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements for models
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=model_labels[model])
        for model, color in model_colors.items()
    ]
    
    # Add baseline line to legend
    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Baseline")
    )

    # Create horizontal legend
    plt.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_with_baseline_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def plot_combined_segmentation_with_baseline(combined_results, results_dir):
    interpreter_configs = {
        "metrics": ["delta-SER-bootstrapped", "delta-delta-dice-bootstrapped"],
        "baseline_metrics": ["SER-bootstrapped", "delta-dice-bootstrapped"],
        "x_labels": ["SER", "Delta Dice"],
    }

    interpreters = [
        "dice",
    ]

    models = ["unet", "pix2pix", "sde"]
    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    # Calculate averages across all interpreters
    plot_data = []
    baseline_data = []

    # Process delta metrics for models
    for metric in interpreter_configs["metrics"]:
        # Group by model, attribute, and metric to calculate means
        metric_data = combined_results[
            (combined_results["metric"] == metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"].isin(models))
        ].copy()

        # Take absolute value before grouping
        metric_data["value"] = metric_data["value"].abs()
        avg_data = (
            metric_data.groupby(["model", "attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        avg_data["metric_type"] = interpreter_configs["x_labels"][
            interpreter_configs["metrics"].index(metric)
        ]

        plot_data.append(avg_data)

    # Process baseline metrics
    for baseline_metric, x_label in zip(interpreter_configs["baseline_metrics"], interpreter_configs["x_labels"]):
        baseline_metric_data = combined_results[
            (combined_results["metric"] == baseline_metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"] == "baseline")
        ].copy()

        # Take absolute value before grouping
        baseline_metric_data["value"] = baseline_metric_data["value"].abs()
        baseline_avg_data = (
            baseline_metric_data.groupby(["attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        baseline_avg_data["metric_type"] = x_label
        
        baseline_data.append(baseline_avg_data)

    plot_data = pd.concat(plot_data)
    baseline_data = pd.concat(baseline_data)

    # Calculate y limits
    all_values = pd.concat([plot_data["value"], baseline_data["value"]])
    y_min = all_values.min()
    y_max = all_values.max()

    y_min = np.floor(y_min / 0.05) * 0.05
    y_max = np.ceil(y_max / 0.05) * 0.05

    g = sns.FacetGrid(
        plot_data,
        col="attribute",
        col_order=["gender", "age"],
        height=6,
        aspect=1.2,
        ylim=(y_min, y_max),
    )

    def plot_bars_with_baseline(data, **kwargs):
        ax = plt.gca()
        unique_metrics = data["metric_type"].unique()
        n_metrics = len(unique_metrics)
        n_models = len(model_labels)
        width = 0.15
        bar_gap = 0.005

        for y in np.arange(y_min, y_max + 0.05, 0.05):
            ax.axhline(
                y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0
            )

        if ax.get_position().x0 > 0.1:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))

        # Get current attribute
        current_attribute = data["attribute"].iloc[0]
        
        # Plot bars for each model
        for i, (model, color) in enumerate(model_colors.items()):
            model_data = data[data["model"] == model]
            x_positions = np.arange(n_metrics) * 0.6 + (i - (n_models - 1) / 2) * (
                width + bar_gap
            )

            values = []
            for metric_type in unique_metrics:
                if metric_type == "SER":
                    value = model_data[
                        model_data["metric"] == "delta-SER-bootstrapped"
                    ]["value"].iloc[0]
                else:  # Delta Dice
                    value = model_data[
                        model_data["metric"] == "delta-delta-dice-bootstrapped"
                    ]["value"].iloc[0]
                values.append(value)

            plt.bar(
                x_positions,
                values,
                width=width,
                color=color,
                label=model_labels[model],
                zorder=2,
            )
        
        # Plot baseline lines
        for j, metric_type in enumerate(unique_metrics):
            baseline_value = baseline_data[
                (baseline_data["attribute"] == current_attribute) & 
                (baseline_data["metric_type"] == metric_type)
            ]["value"].iloc[0]
            
            # Calculate x position for the line (centered on the metric position)
            x_start = j * 0.6 - 0.25
            x_end = j * 0.6 + 0.25
            
            # Plot the baseline line
            plt.plot(
                [x_start, x_end], 
                [baseline_value, baseline_value], 
                color="black", 
                linestyle="--", 
                linewidth=2, 
                zorder=3
            )

        plt.xticks(np.arange(n_metrics) * 0.6, unique_metrics)
        plt.setp(ax.get_xticklabels(), weight="bold")

    g.map_dataframe(plot_bars_with_baseline)

    col_names = {"gender": "Gender", "age": "Age"}
    g.set_titles(template="{col_name}")

    # Add column titles and make them bold
    for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
        ax.set_title(title, fontweight="bold", pad=20, fontsize=20)

    # Add y-axis label to the leftmost plot with bold font
    g.axes[0, 0].set_ylabel("Bias Metric Value", fontweight="bold")

    # Remove legends from all subplots
    for ax in g.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Save the main plot
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_with_baseline_segmentation.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

    # Create separate legend figure
    plt.figure(figsize=(10, 1.5))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements for models
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=model_labels[model])
        for model, color in model_colors.items()
    ]
    
    # Add baseline line to legend
    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Baseline")
    )

    # Create horizontal legend
    plt.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_with_baseline_segmentation_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

def plot_combined_fairness_classifier_std_err_with_baseline(combined_results, results_dir):
    interpreter_configs = {
        "metrics": ["EODD-std-err-bootstrapped", "EOP-std-err-bootstrapped"],
        "x_labels": ["EODD Std Err", "EOP Std Err"],
    }

    interpreters = [
        "ec", "cardiomegaly", "lung-opacity", "lung-lesion", "edema",
        "consolidation", "pneumonia", "atelectasis", "pneumothorax",
        "pleural-effusion", "pleural-other", "fracture", "ttype", "tgrade",
    ]

    models = ["unet", "pix2pix", "sde"]
    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    # Calculate averages across all interpreters
    plot_data = []
    baseline_data = []

    # Process std-err metrics for models
    for metric in interpreter_configs["metrics"]:
        # Group by model, attribute, and metric to calculate means
        metric_data = combined_results[
            (combined_results["metric"] == metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"].isin(models))
        ].copy()

        # Take absolute value before grouping
        metric_data["value"] = metric_data["value"].abs()
        avg_data = (
            metric_data.groupby(["model", "attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        avg_data["metric_type"] = interpreter_configs["x_labels"][
            interpreter_configs["metrics"].index(metric)
        ]

        plot_data.append(avg_data)

    # Process baseline metrics
    for metric, x_label in zip(interpreter_configs["metrics"], interpreter_configs["x_labels"]):
        baseline_metric_data = combined_results[
            (combined_results["metric"] == metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"] == "baseline")
        ].copy()

        # Take absolute value before grouping
        baseline_metric_data["value"] = baseline_metric_data["value"].abs()
        baseline_avg_data = (
            baseline_metric_data.groupby(["attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        baseline_avg_data["metric_type"] = x_label
        
        baseline_data.append(baseline_avg_data)

    plot_data = pd.concat(plot_data)
    baseline_data = pd.concat(baseline_data)

    # Calculate y limits
    all_values = pd.concat([plot_data["value"], baseline_data["value"]])
    y_min = all_values.min()
    y_max = all_values.max()

    y_min = np.floor(y_min / 0.01) * 0.01
    y_max = np.ceil(y_max / 0.01) * 0.01

    g = sns.FacetGrid(
        plot_data,
        col="attribute",
        col_order=["gender", "age", "ethnicity"],
        height=6,
        aspect=1.2,
        ylim=(y_min, y_max),
    )

    def plot_bars_with_baseline(data, **kwargs):
        ax = plt.gca()
        unique_metrics = data["metric_type"].unique()
        n_metrics = len(unique_metrics)
        n_models = len(model_labels)
        width = 0.15
        bar_gap = 0.005

        # Add horizontal grid lines
        for y in np.arange(y_min, y_max + 0.01, 0.01):
            ax.axhline(
                y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0
            )

        if ax.get_position().x0 > 0.1:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.01))

        # Get current attribute
        current_attribute = data["attribute"].iloc[0]
        
        # Plot bars for each model
        for i, (model, color) in enumerate(model_colors.items()):
            model_data = data[data["model"] == model]
            x_positions = np.arange(n_metrics) * 0.6 + (i - (n_models - 1) / 2) * (
                width + bar_gap
            )

            values = []
            for metric_type in unique_metrics:
                if metric_type == "EODD Std Err":
                    metric_rows = model_data[
                        model_data["metric"] == "EODD-std-err-bootstrapped"
                    ]
                    if len(metric_rows) > 0:
                        value = metric_rows["value"].iloc[0]
                    else:
                        value = 0
                else:  # EOP Std Err
                    metric_rows = model_data[
                        model_data["metric"] == "EOP-std-err-bootstrapped"
                    ]
                    if len(metric_rows) > 0:
                        value = metric_rows["value"].iloc[0]
                    else:
                        value = 0
                values.append(value)

            plt.bar(
                x_positions,
                values,
                width=width,
                color=color,
                label=model_labels[model],
                zorder=2,
            )
        
        # Plot baseline lines
        for j, metric_type in enumerate(unique_metrics):
            baseline_rows = baseline_data[
                (baseline_data["attribute"] == current_attribute) & 
                (baseline_data["metric_type"] == metric_type)
            ]
            
            if len(baseline_rows) > 0:
                baseline_value = baseline_rows["value"].iloc[0]
                
                # Calculate x position for the line (centered on the metric position)
                x_start = j * 0.6 - 0.25
                x_end = j * 0.6 + 0.25
                
                # Plot the baseline line
                plt.plot(
                    [x_start, x_end], 
                    [baseline_value, baseline_value], 
                    color="black", 
                    linestyle="--", 
                    linewidth=2, 
                    zorder=3
                )

        plt.xticks(np.arange(n_metrics) * 0.6, unique_metrics)
        plt.setp(ax.get_xticklabels(), weight="bold")

    g.map_dataframe(plot_bars_with_baseline)

    col_names = {"gender": "Gender", "age": "Age", "ethnicity": "Race"}
    g.set_titles(template="{col_name}")

    # Add column titles and make them bold
    for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
        ax.set_title(title, fontweight="bold", pad=20, fontsize=20)

    # Add y-axis label to the leftmost plot with bold font
    g.axes[0, 0].set_ylabel("Standard Error", fontweight="bold")

    # Remove legends from all subplots
    for ax in g.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Save the main plot
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_std_err_classifier_with_baseline.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

    # Create separate legend figure
    plt.figure(figsize=(10, 1.5))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements for models
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=model_labels[model])
        for model, color in model_colors.items()
    ]
    
    # Add baseline line to legend
    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Baseline")
    )

    # Create horizontal legend
    plt.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_std_err_with_baseline_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

def plot_combined_segmentation_std_err_with_baseline(combined_results, results_dir):
    interpreter_configs = {
        "metrics": ["SER-std-err-bootstrapped", "delta-dice-std-err-bootstrapped"],
        "x_labels": ["SER Std Err", "Delta Dice Std Err"],
    }

    interpreters = [
        "dice",
    ]

    models = ["unet", "pix2pix", "sde"]
    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 24,
            "legend.title_fontsize": 24,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    # Calculate averages across all interpreters
    plot_data = []
    baseline_data = []

    # Process std-err metrics for models
    for metric in interpreter_configs["metrics"]:
        # Group by model, attribute, and metric to calculate means
        metric_data = combined_results[
            (combined_results["metric"] == metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"].isin(models))
        ].copy()

        # Take absolute value before grouping
        metric_data["value"] = metric_data["value"].abs()
        avg_data = (
            metric_data.groupby(["model", "attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        avg_data["metric_type"] = interpreter_configs["x_labels"][
            interpreter_configs["metrics"].index(metric)
        ]

        plot_data.append(avg_data)

    # Process baseline metrics
    for metric, x_label in zip(interpreter_configs["metrics"], interpreter_configs["x_labels"]):
        baseline_metric_data = combined_results[
            (combined_results["metric"] == metric)
            & (combined_results["interpreter"].isin(interpreters))
            & (combined_results["model"] == "baseline")
        ].copy()

        # Take absolute value before grouping
        baseline_metric_data["value"] = baseline_metric_data["value"].abs()
        baseline_avg_data = (
            baseline_metric_data.groupby(["attribute", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Set metric_type
        baseline_avg_data["metric_type"] = x_label
        
        baseline_data.append(baseline_avg_data)

    plot_data = pd.concat(plot_data)
    baseline_data = pd.concat(baseline_data)

    # Calculate y limits
    all_values = pd.concat([plot_data["value"], baseline_data["value"]])
    y_min = all_values.min()
    y_max = all_values.max()

    y_min = np.floor(y_min / 0.01) * 0.01
    y_max = np.ceil(y_max / 0.01) * 0.01

    g = sns.FacetGrid(
        plot_data,
        col="attribute",
        col_order=["gender", "age"],
        height=6,
        aspect=1.2,
        ylim=(y_min, y_max),
    )

    def plot_bars_with_baseline(data, **kwargs):
        ax = plt.gca()
        unique_metrics = data["metric_type"].unique()
        n_metrics = len(unique_metrics)
        n_models = len(model_labels)
        width = 0.15
        bar_gap = 0.005

        # Add horizontal grid lines
        for y in np.arange(y_min, y_max + 0.01, 0.01):
            ax.axhline(
                y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0
            )

        if ax.get_position().x0 > 0.1:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.01))

        # Get current attribute
        current_attribute = data["attribute"].iloc[0]
        
        # Plot bars for each model
        for i, (model, color) in enumerate(model_colors.items()):
            model_data = data[data["model"] == model]
            x_positions = np.arange(n_metrics) * 0.6 + (i - (n_models - 1) / 2) * (
                width + bar_gap
            )

            values = []
            for metric_type in unique_metrics:
                if metric_type == "SER Std Err":
                    metric_rows = model_data[
                        model_data["metric"] == "SER-std-err-bootstrapped"
                    ]
                    if len(metric_rows) > 0:
                        value = metric_rows["value"].iloc[0]
                    else:
                        value = 0
                else:  # Delta Dice Std Err
                    metric_rows = model_data[
                        model_data["metric"] == "delta-dice-std-err-bootstrapped"
                    ]
                    if len(metric_rows) > 0:
                        value = metric_rows["value"].iloc[0]
                    else:
                        value = 0
                values.append(value)

            plt.bar(
                x_positions,
                values,
                width=width,
                color=color,
                label=model_labels[model],
                zorder=2,
            )
        
        # Plot baseline lines
        for j, metric_type in enumerate(unique_metrics):
            baseline_rows = baseline_data[
                (baseline_data["attribute"] == current_attribute) & 
                (baseline_data["metric_type"] == metric_type)
            ]
            
            if len(baseline_rows) > 0:
                baseline_value = baseline_rows["value"].iloc[0]
                
                # Calculate x position for the line (centered on the metric position)
                x_start = j * 0.6 - 0.25
                x_end = j * 0.6 + 0.25
                
                # Plot the baseline line
                plt.plot(
                    [x_start, x_end], 
                    [baseline_value, baseline_value], 
                    color="black", 
                    linestyle="--", 
                    linewidth=2, 
                    zorder=3
                )

        plt.xticks(np.arange(n_metrics) * 0.6, unique_metrics)
        plt.setp(ax.get_xticklabels(), weight="bold")

    g.map_dataframe(plot_bars_with_baseline)

    col_names = {"gender": "Gender", "age": "Age"}
    g.set_titles(template="{col_name}")

    # Add column titles and make them bold
    for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
        ax.set_title(title, fontweight="bold", pad=20, fontsize=20)

    # Add y-axis label to the leftmost plot with bold font
    g.axes[0, 0].set_ylabel("Standard Error", fontweight="bold")

    # Remove legends from all subplots
    for ax in g.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Save the main plot
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_std_err_segmentation_with_baseline.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

    # Create separate legend figure
    plt.figure(figsize=(10, 1.5))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements for models
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=model_labels[model])
        for model, color in model_colors.items()
    ]
    
    # Add baseline line to legend
    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Baseline")
    )

    # Create horizontal legend
    plt.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "combined_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/combined_bias_std_err_segmentation_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def plot_combined_summary_v2(combined_results, results_dir):

    latex_path = "latex/templates/combined_fairness_colors_v2.tex"
    file_name = latex_path.split("/")[-1].split(".")[0]
    with open(latex_path, "r") as file:
        latex_content = file.read()

    new_latex_content = ""

    # Color definitions for positive and negative values
    colors = {
        "positive": {
            "significant": "E6C321",  # p < 0.05
            "marginal": "F1D892",  # 0.05 <= p < 0.1
        },
        "negative": {
            "significant": "3089A2",  # p < 0.05
            "marginal": "93C1C9",  # 0.05 <= p < 0.1
        },
    }

    interpreter_metrics = {
        "ec": ("age", "gender", "ethnicity"),
        "cardiomegaly": ("age", "gender", "ethnicity"),
        "lung-opacity": ("age", "gender", "ethnicity"),
        "lung-lesion": ("age", "gender", "ethnicity"),
        "edema": ("age", "gender", "ethnicity"),
        "consolidation": ("age", "gender", "ethnicity"),
        "pneumonia": ("age", "gender", "ethnicity"),
        "atelectasis": ("age", "gender", "ethnicity"),
        "pneumothorax": ("age", "gender", "ethnicity"),
        "pleural-effusion": ("age", "gender", "ethnicity"),
        "pleural-other": ("age", "gender", "ethnicity"),
        "fracture": ("age", "gender", "ethnicity"),
        "tgrade": ("age", "gender"),
        "ttype": ("age", "gender"),
    }
    interpreter_metric_labels = {
        "ec": "EC",
        "cardiomegaly": "Cardiomegaly",
        "lung-opacity": "Lung Opacity",
        "lung-lesion": "Lung Lesion",
        "edema": "Edema",
        "consolidation": "Consolidation",
        "pneumonia": "Pneumonia",
        "atelectasis": "Atelectasis",
        "pneumothorax": "Pneumothorax",
        "pleural-effusion": "Pleural Effusion",
        "pleural-other": "Pleural Other",
        "fracture": "Fracture",
        "tgrade": "Tumor Grade",
        "ttype": "Tumor Type",
    }

    models = ["baseline", "unet", "pix2pix", "sde"]
    metrics = ["EODD", "EOP"]

    segmentation_metrics = ["SER", "delta-dice"]

    attributes = ["gender", "age", "ethnicity"]
    attribute_mapping = {
        "gender": "gender",
        "age": "age",
        "ethnicity": "race",
    }

    attribute_contents = []

    for attribute in attributes:
        attribute_content = latex_content.replace(
            "-attribute-", attribute_mapping[attribute]
        )
        classifier_content = ""
        for interpreter, interpreter_attributes in interpreter_metrics.items():
            content = ""
            if attribute in interpreter_attributes:
                content += f"\\multicolumn{{1}}{{l|}}{{\\textbf{{{interpreter_metric_labels[interpreter]}}}}}"

                for model in models:
                    for metric in metrics:
                        filtered_results = combined_results[
                            (combined_results["metric"] == f"{metric}-bootstrapped")
                            & (combined_results["model"] == model)
                            & (combined_results["interpreter"] == interpreter)
                            & (combined_results["attribute"] == attribute)
                        ]
                        result = filtered_results["value"].iloc[0]

                        if len(filtered_results) == 0:
                            continue

                        if model == "baseline":
                            content += f"& {result:.3f}"
                        else:
                            filtered_p_values = combined_results[
                                (combined_results["metric"] == f"delta-{metric}-p-value")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == interpreter)
                                & (combined_results["attribute"] == attribute)
                            ]

                            filtered_std_errors = combined_results[
                                (combined_results["metric"] == f"delta-{metric}-std-err")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == interpreter)
                                & (combined_results["attribute"] == attribute)
                            ]
                            filtered_delta_result = combined_results[
                                (combined_results["metric"] == f"delta-{metric}-bootstrapped")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == interpreter)
                                & (combined_results["attribute"] == attribute)
                            ]
                            

                            p_value = filtered_p_values["value"].iloc[0]
                            std_error = filtered_std_errors["value"].iloc[0]
                            delta_result = filtered_delta_result["value"].iloc[0]

                            color = None
                            if delta_result > 0:
                                if p_value < 0.05:
                                    color = colors["positive"]["significant"]
                                elif p_value < 0.1:
                                    color = colors["positive"]["marginal"]
                            if delta_result < 0:
                                if p_value < 0.05:
                                    color = colors["negative"]["significant"]
                                elif p_value < 0.1:
                                    color = colors["negative"]["marginal"]

                            if color is not None:
                                if abs(std_error) > abs(result):
                                    if delta_result > 0:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{\\phantom{{-}}{result:.3f}}}}}"
                                    else:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{{result:.3f}}}}}"
                                else:
                                    if delta_result > 0:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{\\phantom{{-}}{result:.3f}}}"
                                    else:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{{result:.3f}}}"
                            else:
                                if abs(std_error) > abs(delta_result):
                                    if delta_result > 0:
                                        content += f"& \\textbf{{\\phantom{{-}}{result:.3f}}}"
                                    else:
                                        content += f"& \\textbf{{{result:.3f}}}"
                                else:
                                    if delta_result > 0:
                                        content += f"& \\phantom{{-}}{result:.3f}"
                                    else:
                                        content += f"& {result:.3f}"

                classifier_content += content + "\\\\ \n"

        attribute_content = attribute_content.replace(
            "-classifier-", classifier_content
        )

        if attribute in ["gender", "age"]:
            segmentation_content = """
                &                                   &                                            &                                            &                                                   &                                            &                                                   &                                            &                                                   \\\\ \\hline
            \\multicolumn{1}{l|}{} & \\multicolumn{1}{c}{\\textbf{SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta$ Dice}} & \\multicolumn{1}{c}{\\textbf{$\\Delta$ SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta \\Delta$ Dice}} & \\multicolumn{1}{c}{\\textbf{$\\Delta$ SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta \\Delta$ Dice}} & \\multicolumn{1}{c}{\\textbf{$\\Delta$ SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta \\Delta$ Dice}} \\\\ \\hline
                """
            content = ""
            content += f"\\multicolumn{{1}}{{l|}}{{\\textbf{{Segmentation}}}}"

            for model in models:
                for metric in segmentation_metrics:
                    filtered_results = combined_results[
                        (combined_results["metric"] == f"{metric}-bootstrapped")
                        & (combined_results["model"] == model)
                        & (combined_results["interpreter"] == "dice")
                        & (combined_results["attribute"] == attribute)
                    ]
                    result = filtered_results["value"].iloc[0]
                    if len(filtered_results) == 0:
                        continue

                    if model == "baseline":
                        content += f"& {result:.3f}"
                    else:
                        filtered_p_values = combined_results[
                            (combined_results["metric"] == f"delta-{metric}-p-value")
                            & (combined_results["model"] == model)
                            & (combined_results["interpreter"] == "dice")
                            & (combined_results["attribute"] == attribute)
                        ]

                        filtered_std_errors = combined_results[
                            (combined_results["metric"] == f"delta-{metric}-std-err")
                            & (combined_results["model"] == model)
                            & (combined_results["interpreter"] == "dice")
                            & (combined_results["attribute"] == attribute)
                        ]
                        filtered_delta_result = combined_results[
                            (combined_results["metric"] == f"delta-{metric}-bootstrapped")
                            & (combined_results["model"] == model)
                            & (combined_results["interpreter"] == "dice")
                            & (combined_results["attribute"] == attribute)
                        ]

                        p_value = filtered_p_values["value"].iloc[0]
                        std_error = filtered_std_errors["value"].iloc[0]
                        delta_result = filtered_delta_result["value"].iloc[0]

                        color = None
                        if delta_result > 0:
                            if p_value < 0.05:
                                color = colors["positive"]["significant"]
                            elif p_value < 0.1:
                                color = colors["positive"]["marginal"]
                        if delta_result < 0:
                            if p_value < 0.05:
                                color = colors["negative"]["significant"]
                            elif p_value < 0.1:
                                color = colors["negative"]["marginal"]

                        if color is not None:
                            if abs(std_error) > abs(delta_result):
                                if delta_result > 0:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{\\phantom{{-}}{result:.3f}}}}}"
                                else:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{{result:.3f}}}}}"
                            else:
                                if delta_result > 0:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{\\phantom{{-}}{result:.3f}}}"
                                else:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{{result:.3f}}}"
                        else:
                            if abs(std_error) > abs(delta_result):
                                if delta_result > 0:
                                    content += f"& \\textbf{{\\phantom{{-}}{result:.3f}}}"
                                else:
                                    content += f"& \\textbf{{{result:.3f}}}"
                            else:
                                if delta_result > 0:
                                    content += f"& \\phantom{{-}}{result:.3f}"
                                else:
                                    content += f"& {result:.3f}"

            segmentation_content += "\n" + content + "\\\\"

            attribute_content += segmentation_content

        attribute_content += """    
        \end{tabular} 
        } % End of resizebox -legend-
        \end{table}"""

        legend = """\\\\[0.3cm]
            \\begin{tabular}{llllllll} 
            \\cellcolor[HTML]{E6C321} & $+$, $p < 0.05$ & \\cellcolor[HTML]{F1D892} &$+$, $0.05 \\leq p < 0.1$ & \\cellcolor[HTML]{3089A2} & $-$, $p < 0.05$  & \\cellcolor[HTML]{93C1C9} & $-$, $0.05 \\leq p < 0.1$ \\\\
            \\multicolumn{8}{l}{\\textbf{Bold} indicates standard error larger than absolute effect size}
            \\end{tabular}"""
        attribute_content = attribute_content.replace("-legend-", legend)

        attribute_contents.append(attribute_content)

    for content in attribute_contents:
        new_latex_content += content

    with open(os.path.join(results_dir, f"{file_name}.tex"), "w") as file:
        file.write(new_latex_content)