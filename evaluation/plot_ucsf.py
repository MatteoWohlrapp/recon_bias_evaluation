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


def plot_ucsf_performance(results, results_dir, name):

    metric_pairs = [
        ("tgrade", "psnr"),
        ("ttype", "psnr"),
        ("dice", "psnr"),
        ("auroc-avg", "psnr"),
    ]

    metric_labels = {
        "psnr": "PSNR",
        "tgrade": "AUROC",
        "ttype": "AUROC",
        "dice": "Dice",
        "auroc-avg": "AUROC",
    }

    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

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

    for metric1, metric2 in metric_pairs:
        # Create new figure for each pair
        plt.figure(figsize=(12, 6))

        # Special handling for averaged AUROC
        if metric1 == "auroc-avg":
            # Calculate average of tgrade and ttype for each model and acceleration
            data_tgrade = results[
                (results["metric"] == "tgrade") & (results["model"] != "baseline")
            ].copy()
            data_ttype = results[
                (results["metric"] == "ttype") & (results["model"] != "baseline")
            ].copy()

            # Merge on model and acceleration to properly align values
            data1 = pd.merge(
                data_tgrade[["model", "acceleration", "value"]],
                data_ttype[["model", "acceleration", "value"]],
                on=["model", "acceleration"],
                suffixes=("_tgrade", "_ttype"),
            )
            data1["value"] = (data1["value_tgrade"] + data1["value_ttype"]) / 2
            data1["metric"] = "auroc-avg"
        else:
            # Original data filtering
            data1 = results[
                (results["metric"] == metric1) & (results["model"] != "baseline")
            ].copy()

        data2 = results[
            (results["metric"] == metric2) & (results["model"] != "baseline")
        ].copy()

        # Create first axis
        ax1 = plt.gca()

        line1 = sns.lineplot(
            data=data1,
            x="acceleration",
            y="value",
            hue="model",
            palette=model_colors,
            marker="o",
            ax=ax1,
            linewidth=2.5,
        )

        # Create second y-axis and add reference lines
        ax2 = ax1.twinx()
        line2 = sns.lineplot(
            data=data2,
            x="acceleration",
            y="value",
            hue="model",
            palette=model_colors,
            marker="s",
            linestyle="--",
            ax=ax2,
            linewidth=2.5,
        )

        # Set labels and title with bold font
        ax1.set_xlabel("Acceleration Factor", fontweight="bold")
        ax1.set_ylabel(metric_labels[metric1], fontweight="bold")
        ax2.set_ylabel(metric_labels[metric2], fontweight="bold")

        # Make tick labels bold
        ax1.tick_params(axis="both", which="major")
        ax2.tick_params(axis="both", which="major")

        # Remove legends from plot
        ax1.get_legend().remove()
        ax2.get_legend().remove()

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
                label=metric_labels[metric1],
                linewidth=2.5,
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                marker="s",
                label=metric_labels[metric2],
                linewidth=2.5,
                markersize=8,
            ),
        ]

        # Set x-axis to treat values as categorical
        available_accelerations = sorted(data1["acceleration"].unique())
        ax1.set_xticks(range(len(available_accelerations)))
        ax1.set_xticklabels(available_accelerations)

        # Update the data points to use categorical positions
        for line in ax1.lines:
            if len(line.get_xdata()) > 0:  # Check if line has data
                old_x = line.get_xdata()
                new_x = [list(available_accelerations).index(x) for x in old_x]
                line.set_xdata(new_x)

        for line in ax2.lines:
            if len(line.get_xdata()) > 0:  # Check if line has data
                old_x = line.get_xdata()
                new_x = [list(available_accelerations).index(x) for x in old_x]
                line.set_xdata(new_x)

        for y in [32, 34, 36, 38, 40, 42, 44]:
            ax2.axhline(
                y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0
            )

        # Set x-axis limits to remove extra whitespace
        ax1.set_xlim(-0.2, len(available_accelerations) - 0.8)

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
            save_dir = os.path.join(results_dir, "ucsf_performance", fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{name}_performance_{metric1}_{metric2}.{fmt}",
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
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="-",
            marker="o",
            label=metric_labels[metric1],
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            marker="s",
            label=metric_labels[metric2],
        ),
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
        save_dir = os.path.join(results_dir, "ucsf_performance", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/{name}_performance_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def plot_ucsf_additional_bias(results, results_dir, name):

    interpreter_configs = {
        "tgrade": {"metrics": ["delta-EODD", "delta-EOP"], "x_labels": ["EODD", "EOP"]},
        "ttype": {"metrics": ["delta-EODD", "delta-EOP"], "x_labels": ["EODD", "EOP"]},
        "dice": {
            "metrics": ["delta-SER", "delta-delta-dice"],
            "x_labels": ["SER", "Dice"],
        },
    }

    title_map = {
        "tgrade": "Tumor Grade",
        "ttype": "Tumor Type",
        "dice": "Segmentation",
    }
    # (Optional) Labels for display in the plot legends or axis labels.
    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

    # Update matplotlib rcParams to match chex style
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

    for interpreter, config in interpreter_configs.items():
        # Filter data for current interpreter
        data_task = results[results["interpreter"] == interpreter].copy()

        # Reshape data for plotting
        plot_data = []
        max_std_err = 0
        min_std_err = 0
        for metric in config["metrics"]:
            # Check for max/min std errors
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
            metric_data["metric_type"] = config["x_labels"][
                config["metrics"].index(metric)
            ]
            plot_data.append(metric_data)

        plot_data = pd.concat(plot_data)

        # Calculate y limits with padding for error bars
        y_min = plot_data["value"].min()
        y_max = plot_data["value"].max()

        # Add padding for error bars
        y_min = np.floor((y_min - min_std_err) / 0.05) * 0.05
        y_max = np.ceil((y_max + max_std_err) / 0.05) * 0.05

        # Create the plot with fixed y limits
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
            n_models = len(model_colors)
            width = 0.15
            bar_gap = 0.005

            # Add horizontal reference lines using global limits
            for y in np.arange(y_min, y_max + 0.05, 0.05):
                ax.axhline(
                    y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0
                )

            # Handle y-axis appearance based on facet position
            if ax.get_position().x0 > 0.1:  # Second facet
                ax.spines["left"].set_visible(False)
                ax.yaxis.set_visible(False)
            else:  # First facet
                # Set y-ticks at every 0.05 interval
                ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))

            for i, (model, color) in enumerate(model_colors.items()):
                model_data = data[data["model"] == model]
                x_positions = np.arange(n_metrics) * 0.6 + (i - (n_models - 1) / 2) * (
                    width + bar_gap
                )

                values = []
                std_errs = []
                for metric_type in unique_metrics:
                    metric_name = next(
                        m
                        for m in config["metrics"]
                        if config["x_labels"][config["metrics"].index(m)] == metric_type
                    )
                    value = model_data[model_data["metric"] == metric_name][
                        "value"
                    ].iloc[0]
                    std_err = model_data[
                        model_data["metric"] == f"{metric_name}-std-err"
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

        # Update title style
        col_names = {"gender": "Gender", "age": "Age"}
        g.set_titles(template="{col_name}")

        # Add column titles with bold font and smaller size
        for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
            ax.set_title(title, fontweight="bold", pad=20, fontsize=20)

        # Add interpreter title
        g.fig.suptitle(
            title_map[interpreter],
            fontweight="bold",
            fontsize=28,
            y=1.1,
        )

        # Make y-axis label bold
        g.axes[0, 0].set_ylabel("Additional Bias", fontweight="bold")

        # Remove legend from main plots
        for ax in g.axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Save the plots
        for fmt in ["eps", "pdf", "png"]:
            save_dir = os.path.join(results_dir, "ucsf_bias", fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{name}_bias_{interpreter}.{fmt}",
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
        save_dir = os.path.join(results_dir, "ucsf_bias", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/{name}_bias_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()


def plot_ucsf_additional_bias_summary_segmentation(results, results_dir, name):
    latex_path = "latex/templates/ucsf_fairness_colors_segmentation.tex"

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
        "dice",
    ]
    metrics = ["delta-SER", "delta-delta-dice"]
    metrics_labels = {
        "delta-SER": ("SER", ""),
        "delta-delta-dice": ("$\\Delta$ dice", ""),
    }
    sensitive_attributes = ["gender", "age"]

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


def plot_ucsf_additional_bias_summary_classifier(results, results_dir, name):
    latex_path = "latex/templates/ucsf_fairness_colors_classifier.tex"

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
        "tgrade",
        "ttype",
    ]
    metrics = ["delta-EODD", "delta-EOP"]
    metrics_labels = {
        "delta-EODD": ("EODD", ""),
        "delta-EOP": ("EOP", ""),
    }
    sensitive_attributes = ["gender", "age"]

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
