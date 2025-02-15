import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.lines import Line2D
import numpy as np

model_colors = {
    "unet": "#3089A2",
    "pix2pix": "#E6C321",
    "sde": "#EB0007",
}


def plot_chex_performance(results, results_dir, name):

    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

    metrics = [
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
        "average",
    ]

    metric_labels = {
        "ec": "Enlarged Cardiomediastinum",
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
    }

    # Set font sizes and line widths
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

    for metric in metrics:
        # Create new figure for each pair
        plt.figure(figsize=(12, 6))

        # Add bold metric title if not average
        if metric != "average":
            plt.title(metric_labels[metric].title(), fontweight="bold", pad=20)

        # Filter data for the current metrics
        data1 = results[
            (results["metric"] == metric) & (results["model"] != "baseline")
        ].copy()
        if metric == "average":
            data2 = results[
                (results["metric"] == "psnr") & (results["model"] != "baseline")
            ].copy()
        else:
            data2 = results[
                (results["metric"] == f"psnr-{metric}")
                & (results["model"] != "baseline")
            ].copy()

        # Create first axis and plot
        ax1 = plt.gca()
        line1 = sns.lineplot(
            data=data1,
            x="photon_count",
            y="value",
            hue="model",
            palette=model_colors,
            marker="o",
            ax=ax1,
            linewidth=2.5,
        )

        # Create second y-axis and plot
        ax2 = ax1.twinx()
        line2 = sns.lineplot(
            data=data2,
            x="photon_count",
            y="value",
            hue="model",
            palette=model_colors,
            marker="s",
            linestyle="--",
            ax=ax2,
            linewidth=2.5,
        )

        # Set labels and title with bold font
        ax1.set_xlabel("Photon Count", fontweight="bold")
        ax1.set_ylabel("AUROC", fontweight="bold")
        ax2.set_ylabel("PSNR", fontweight="bold")

        # Make tick labels bold
        ax1.tick_params(axis="both", which="major")
        ax2.tick_params(axis="both", which="major")

        # Get lines and labels for both plots
        lines1, labels1 = ax1.get_legend_handles_labels()

        # Update custom lines to match new thickness
        custom_lines = [
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="-",
                marker="o",
                label="AUROC",
                linewidth=2.5,
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                marker="s",
                label="PSNR",
                linewidth=2.5,
                markersize=8,
            ),
        ]

        """ax1.legend(lines1[:3] + custom_lines,
                    [model_labels[model] for model in labels1[:3]] + ["AUROC", "PSNR"],
                    bbox_to_anchor=(1.15, 1))"""
        ax1.get_legend().remove()
        ax2.get_legend().remove()

        # Set x-axis to treat values as categorical
        available_photon_counts = sorted(data1["photon_count"].unique(), reverse=True)
        ax1.set_xticks(range(len(available_photon_counts)))
        ax1.set_xticklabels(available_photon_counts)

        # Update the data points to use categorical positions
        for line in ax1.lines:
            if len(line.get_xdata()) > 0:  # Check if line has data
                old_x = line.get_xdata()
                new_x = [list(available_photon_counts).index(x) for x in old_x]
                line.set_xdata(new_x)

        for line in ax2.lines:
            if len(line.get_xdata()) > 0:  # Check if line has data
                old_x = line.get_xdata()
                new_x = [list(available_photon_counts).index(x) for x in old_x]
                line.set_xdata(new_x)

        for y in [26, 28, 30, 32]:
            ax2.axhline(
                y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0
            )

        # Set x-axis limits to remove extra whitespace
        ax1.set_xlim(-0.2, len(available_photon_counts) - 0.8)

        # Calculate percentage drops for both metrics
        y1_max = data1["value"].max()
        y1_min = data1["value"].min()
        y2_max = data2["value"].max()
        y2_min = data2["value"].min()

        # Calculate the larger percentage drop
        drop1 = (y1_max - y1_min) / y1_max
        drop2 = (y2_max - y2_min) / y2_max
        max_drop = max(drop1, drop2)

        # Set limits to show the same percentage drop for both axes
        y1_bottom = y1_max * (1 - max_drop)
        y2_bottom = y2_max * (1 - max_drop)

        # Add small padding at top (5%)
        ax1.set_ylim(y1_bottom * 0.95, y1_max * 1.05)
        ax2.set_ylim(y2_bottom * 0.95, y2_max * 1.05)

        # Adjust figure size to accommodate legend
        plt.gcf().set_size_inches(12, 6)  # Wider figure to fit legend

        # Adjust layout to prevent legend overlap
        plt.tight_layout()

        # Save plots in both formats
        for fmt in ["eps", "pdf", "png"]:
            save_dir = os.path.join(results_dir, "chex_performance", fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{name}_performance_{metric}_psnr.{fmt}",
                bbox_inches="tight",
                dpi=300,
                format=fmt,
            )
        plt.close()

    # Create a separate legend figure
    plt.figure(figsize=(12, 1))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements
    model_lines = [
        Line2D([0], [0], color=color, label=model_labels[model])
        for model, color in model_colors.items()
    ]
    metric_lines = [
        Line2D([0], [0], color="gray", linestyle="-", marker="o", label="AUROC"),
        Line2D([0], [0], color="gray", linestyle="--", marker="s", label="PSNR"),
    ]

    # Create horizontal legend
    plt.legend(
        handles=model_lines + metric_lines,
        loc="center",
        ncol=len(model_lines) + len(metric_lines),
        bbox_to_anchor=(0.5, 0.5),
    )

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "chex_performance", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/{name}_performance_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def plot_chex_additional_bias(results, results_dir, name):
    interpreter_configs = {
        "metrics": ["delta-EODD", "delta-EOP"],
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
        "average",
    ]

    metric_labels = {
        "ec": "Enlarged Cardiomediastinum",
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
    }

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

    for interpreter in interpreters:
        data_task = results[results["interpreter"] == interpreter].copy()

        # Modify how we prepare plot_data to include std err metrics
        plot_data = []
        max_std_err = 0
        min_std_err = 0
        for metric in interpreter_configs["metrics"]:

            for _, row in data_task.iterrows():
                if row["metric"] == metric:
                    if row["value"] > 0:
                        for _, row_std_err in data_task.iterrows():
                            if row_std_err["metric"] == f"{metric}-std-err":
                                if row_std_err["value"] > max_std_err:
                                    max_std_err = row_std_err["value"]
                    else:
                        for _, row_std_err in data_task.iterrows():
                            if row_std_err["metric"] == f"{metric}-std-err":
                                if row_std_err["value"] > min_std_err:
                                    min_std_err = row_std_err["value"]

            metric_data = data_task[
                data_task["metric"].isin([metric, f"{metric}-std-err"])
            ].copy()

            # Set metric_type for both the main metric and its std-err
            metric_data["metric_type"] = interpreter_configs["x_labels"][
                interpreter_configs["metrics"].index(metric)
            ]

            plot_data.append(metric_data)

        plot_data = pd.concat(plot_data)

        # Calculate y limits with more padding for error bars
        y_min = plot_data["value"].min()
        y_max = plot_data["value"].max()

        # Add padding for error bars by including the standard errors
        """std_err_max = plot_data[plot_data["metric"].str.contains("std-err", na=False)][
            "value"
        ].max()
        y_min = np.floor((y_min - std_err_max) / 0.05) * 0.05  # Add padding below
        #y_max = np.ceil((y_max + std_err_max) / 0.05) * 0.05  # Add padding above
        y_max = y_max + std_err_max"""

        y_min = np.floor((y_min - min_std_err) / 0.05) * 0.05
        y_max = np.ceil((y_max + max_std_err) / 0.05) * 0.05

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
            n_models = len(model_colors)
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
                std_errs = []
                for metric_type in unique_metrics:
                    if metric_type == "EODD":
                        value = model_data[model_data["metric"] == "delta-EODD"][
                            "value"
                        ].iloc[0]
                        std_err = model_data[
                            model_data["metric"] == "delta-EODD-std-err"
                        ]["value"].iloc[0]
                    else:  # EOP
                        value = model_data[model_data["metric"] == "delta-EOP"][
                            "value"
                        ].iloc[0]
                        std_err = model_data[
                            model_data["metric"] == "delta-EOP-std-err"
                        ]["value"].iloc[0]
                    values.append(value)
                    std_errs.append(std_err)

                plt.bar(
                    x_positions,
                    values,
                    width=width,
                    color=color,
                    label=model_labels[model],
                    zorder=2,
                    yerr=std_errs,
                    capsize=5,
                    error_kw={"elinewidth": 1.5, "capthick": 1.5},
                )

            plt.xticks(np.arange(n_metrics) * 0.6, unique_metrics)
            plt.setp(ax.get_xticklabels(), weight="bold")

        g.map_dataframe(plot_bars)

        col_names = {"gender": "Gender", "age": "Age", "ethnicity": "Race"}
        g.set_titles(template="{col_name}")

        # Add column titles and make them bold, with smaller font size
        for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
            ax.set_title(
                title, fontweight="bold", pad=20, fontsize=20
            )  # Match x-label size

        # Add single interpreter title if not average
        if interpreter != "average":
            g.fig.suptitle(
                metric_labels.get(interpreter, interpreter.replace("-", " ").title()),
                fontweight="bold",
                fontsize=28,
                y=1.1,
            )

        # Add y-axis label to the leftmost plot with bold font
        g.axes[0, 0].set_ylabel("Additional Bias", fontweight="bold")

        # Only show legend for average plot
        """if interpreter == "average":
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:"""
        for ax in g.axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Save the plots
        for fmt in ["eps", "pdf", "png"]:
            save_dir = os.path.join(results_dir, "chex_bias", fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{name}_bias_{interpreter}.{fmt}",
                bbox_inches="tight",
                dpi=300,
                format=fmt,
            )
        plt.close()

    # Create a separate legend figure
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
        save_dir = os.path.join(results_dir, "chex_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/{name}_bias_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def plot_chex_additonal_bias_summary(results, results_dir, name):
    latex_path = "latex/templates/chex_fairness_colors.tex"

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

    models = ["unet", "pix2pix", "sde"]
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
        "average",
    ]
    metrics = ["delta-EODD", "delta-EOP"]
    metrics_labels = {
        "delta-EODD": ("EODD", ""),
        "delta-EOP": ("EOP", ""),
        "EODD-percent": ("EODD", "in percent "),
        "EOP-percent": ("EOP", "in percent "),
    }
    sensitive_attributes = ["gender", "age", "ethnicity"]

    # Read the LaTeX file
    file_name = latex_path.split("/")[-1].split(".")[0]
    with open(latex_path, "r") as file:
        latex_content = file.read()

    new_latex_content = ""

    for metric in metrics:
        current_content = latex_content
        current_content = current_content.replace("-metric-", metrics_labels[metric][0])
        current_content = current_content.replace(
            "-description-", metrics_labels[metric][1]
        )
        for model in models:
            for interpreter in interpreters:
                for sensitive_attribute in sensitive_attributes:
                    # Create the key to replace in the latex template
                    key = f"{interpreter}-{model}-{sensitive_attribute}"

                    # Filter results
                    filtered_results = results[
                        (results["metric"] == metric)
                        & (results["model"] == model)
                        & (results["interpreter"] == interpreter)
                        & (results["attribute"] == sensitive_attribute)
                    ]

                    filtered_p_values = results[
                        (results["metric"] == f"{metric}-p-value")
                        & (results["model"] == model)
                        & (results["interpreter"] == interpreter)
                        & (results["attribute"] == sensitive_attribute)
                    ]

                    if len(filtered_p_values) == 0 or len(filtered_results) == 0:
                        # Replace with empty cell if no data
                        # current_content = current_content.replace(key, "-")
                        continue

                    p_value = filtered_p_values["value"].iloc[0]
                    result = filtered_results["value"].iloc[0]

                    # Determine color based on result value and p-value
                    color = None
                    if result > 0:
                        if p_value < 0.05:
                            color = colors["positive"]["significant"]
                        elif p_value < 0.1:
                            color = colors["positive"]["marginal"]
                    else:
                        if p_value < 0.05:
                            color = colors["negative"]["significant"]
                        elif p_value < 0.1:
                            color = colors["negative"]["marginal"]

                    # Format the value
                    formatted_value = f"{result:.3f}"

                    # Add color if significant
                    if color:
                        formatted_value = (
                            f"\\cellcolor[HTML]{{{color}}}{formatted_value}"
                        )

                    # Replace the key in the latex content
                    current_content = current_content.replace(key, formatted_value)

        legend = """\\\\[0.3cm]
            \\begin{tabular}{llllllll} 
            \\cellcolor[HTML]{E6C321} & $+$, $p < 0.05$ & \\cellcolor[HTML]{F1D892} &$+$, $0.05 \\leq p < 0.1$ & \\cellcolor[HTML]{3089A2} & $-$, $p < 0.05$  & \\cellcolor[HTML]{93C1C9} & $-$, $0.05 \\leq p < 0.1$ \\
            \\end{tabular}"""
        current_content = current_content.replace("-legend-", legend)
        new_latex_content += current_content

    # Write to latex
    with open(os.path.join(results_dir, f"{file_name}.tex"), "w") as file:
        file.write(new_latex_content)
