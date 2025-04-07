import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Original colors from the dict
colors = {
    "pix2pix": {
        "significant": "#E6C321",  # Yellow - original
        "marginal": "#F1D892",     # Light yellow - mitigated
    },
    "unet": {
        "significant": "#3089A2",   # Blue - original
        "marginal": "#93C1C9",      # Light blue - mitigated
    },
    "sde": {
        "significant": "#EB0007",   # Red - original
        "marginal": "#FF9999",      # Light red - mitigated
    }
}

def create_latex_grid(run_name, results_dir):
    """Create a LaTeX figure with a grid of all plots."""
    
    # Define the order of plots (with average at the end)
    interpreters = [
        "atelectasis", "cardiomegaly",
        "consolidation", "edema",
        "ec", "fracture",
        "lung-lesion", "lung-opacity",
        "pleural-effusion", "pleural-other",
        "pneumonia", "pneumothorax",
        "average"
    ]

    latex_content = []
    latex_content.append(r"\begin{figure}[htbp]")
    latex_content.append(r"\caption{" + f"{run_name}" + "}")
    latex_content.append(r"    \centering")
    
    # Add legend
    latex_content.append(r"    \begin{subfigure}[t]{0.4\textwidth}")
    latex_content.append(r"        \centering")
    latex_content.append(r"        \includegraphics[width=\linewidth]{" + f"GR/fig/{run_name}/mitigation_comparison_legend.pdf" + "}")
    latex_content.append(r"    \end{subfigure}")
    latex_content.append("")
    
    # Start grid
    latex_content.append(r"    \vspace{0.5cm}")
    latex_content.append(r"    \begin{tabular}{c@{\hspace{0.5cm}}c}")
    
    # Add all plots except average in a 2x6 grid
    for i in range(0, len(interpreters)-1, 2):  # -1 to exclude average for now
        interpreter1 = interpreters[i]
        interpreter2 = interpreters[i+1]
        
        # Add first plot
        latex_content.append(r"        \begin{subfigure}[t]{0.48\textwidth}")
        latex_content.append(r"            \centering")
        latex_content.append(r"            \includegraphics[width=\linewidth]{" + f"GR/fig/{run_name}/mitigation_comparison_{interpreter1}.pdf" + "}")
        latex_content.append(r"        \end{subfigure}")
        latex_content.append(r"        &")
        
        # Add second plot
        latex_content.append(r"        \begin{subfigure}[t]{0.48\textwidth}")
        latex_content.append(r"            \centering")
        latex_content.append(r"            \includegraphics[width=\linewidth]{" + f"GR/fig/{run_name}/mitigation_comparison_{interpreter2}.pdf" + "}")
        latex_content.append(r"        \end{subfigure}")
        
        # Add line break unless it's the last pair
        if i < len(interpreters)-3:  # -3 because we're excluding average
            latex_content.append(r"        \\[0.3cm]")
        latex_content.append("")

    latex_content.append(r"    \end{tabular}")
    latex_content.append("")
    
    # Add average plot centered below the grid
    latex_content.append(r"    \vspace{0.5cm}")
    latex_content.append(r"    \begin{subfigure}[t]{0.48\textwidth}")
    latex_content.append(r"        \centering")
    latex_content.append(r"        \includegraphics[width=\linewidth]{" + f"GR/fig/{run_name}/mitigation_comparison_average.pdf" + "}")
    latex_content.append(r"    \end{subfigure}")
    latex_content.append("")
    
    # Add caption and label
    latex_content.append(r"\end{figure}")
    
    # Write to file
    with open(os.path.join(results_dir, run_name, "mitigation_grid.tex"), "w") as f:
        f.write("\n".join(latex_content))

def plot_fairness_change_mitigation(original_df, reweighted_df, eodd_df, adv_df, results_dir, run_name):
    """
    Create a faceted bar plot showing the difference between different mitigation approaches vs. original
    for EODD and EOP metrics across different models and attributes.
    
    Args:
        original_df: DataFrame with original results
        reweighted_df: DataFrame with reweighted results
        eodd_df: DataFrame with EODD mitigation results
        adv_df: DataFrame with adversarial mitigation results
        results_dir: Directory to save the plots
        run_name: Name of the run for organizing output files
    """
    # Define metrics to compare
    metrics = ["EODD-bootstrapped", "EOP-bootstrapped"]
    x_labels = ["EODD", "EOP"]
    
    # Define interpreters for ChexPert dataset
    interpreters = [
        "ec", "cardiomegaly", "lung-opacity", "lung-lesion", "edema",
        "consolidation", "pneumonia", "atelectasis", "pneumothorax",
        "pleural-effusion", "pleural-other", "fracture", "average", "ttype", "tgrade"
    ]
    
    # Define labels for interpreters
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
        "average": "Average",
        "tgrade": "Tumor Grade",
        "ttype": "Tumor Type",
    }
    
    # Define model labels
    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}
    
    # Updated color scheme with three mitigation types per model
    colors = {
        "pix2pix": {
            "reweighted": "#F1D892",  # Light yellow
            "eodd": "#E6C321",        # Yellow
            "adv": "#B39700"          # Dark yellow/gold
        },
        "unet": {
            "reweighted": "#93C1C9",  # Light blue
            "eodd": "#3089A2",        # Blue
            "adv": "#0A5C7A"          # Dark blue
        },
        "sde": {
            "reweighted": "#FF9999",  # Light red
            "eodd": "#EB0007",        # Red
            "adv": "#B30005"          # Dark red
        }
    }
    
    # Set plotting parameters
    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 24,
        "legend.title_fontsize": 24,
    })
    
    for interpreter in interpreters:
        if not interpreter in original_df["interpreter"].unique():
            continue
        # Filter data for current interpreter
        orig_data = original_df[original_df["interpreter"] == interpreter].copy()
        reweight_data = reweighted_df[reweighted_df["interpreter"] == interpreter].copy()
        eodd_data = eodd_df[eodd_df["interpreter"] == interpreter].copy()
        adv_data = adv_df[adv_df["interpreter"] == interpreter].copy()
        
        if len(orig_data) == 0 or len(reweight_data) == 0 or len(eodd_data) == 0 or len(adv_data) == 0:
            continue  # Skip if no data for this interpreter
        
        # Create combined data for facet grid
        plot_data = pd.DataFrame()
        
        for metric in metrics:
            # Get original data
            orig_metric = orig_data[orig_data["metric"] == metric].copy()
            
            # Get reweighted data and calculate difference
            reweight_metric = reweight_data[reweight_data["metric"] == metric].copy()
            reweight_metric["type"] = "reweighted"
            reweight_metric["diff_value"] = reweight_metric["value"] - orig_metric["value"].values
            
            # Get EODD data and calculate difference
            eodd_metric = eodd_data[eodd_data["metric"] == metric].copy()
            eodd_metric["type"] = "eodd"
            eodd_metric["diff_value"] = eodd_metric["value"] - orig_metric["value"].values
            
            # Get adversarial data and calculate difference
            adv_metric = adv_data[adv_data["metric"] == metric].copy()
            adv_metric["type"] = "adv"
            adv_metric["diff_value"] = adv_metric["value"] - orig_metric["value"].values
            
            # Add to plot data
            plot_data = pd.concat([plot_data, reweight_metric, eodd_metric, adv_metric])
        
        # Add metric_type for easier plotting
        plot_data["metric_type"] = plot_data["metric"].apply(
            lambda x: "EODD" if x == "EODD-bootstrapped" else "EOP"
        )
        
        # Calculate y limits with padding
        y_min = plot_data["diff_value"].min()
        y_max = plot_data["diff_value"].max()
        
        # Round to nearest 0.05
        y_min = np.floor(y_min / 0.05) * 0.05
        y_max = np.ceil(y_max / 0.05) * 0.05
        
        # For tgrade and ttype, only show age and gender attributes
        if interpreter in ["tgrade", "ttype"]:
            plot_data = plot_data[plot_data["attribute"].isin(["gender", "age"])]
            col_order = ["gender", "age"]
        else:
            col_order = ["gender", "age", "ethnicity"]
        
        # Create facet grid
        g = sns.FacetGrid(
            plot_data,
            col="attribute",
            col_order=col_order,
            height=6,
            aspect=1.2,
            ylim=(y_min, y_max),
        )
        
        def plot_bars(data, **kwargs):
            ax = plt.gca()
            unique_metrics = ["EODD", "EOP"]
            n_metrics = len(unique_metrics)
            n_models = len(model_labels)
            width = 0.08  # Narrower bars to accommodate more
            bar_gap = 0.005
            group_width = width * 3 + bar_gap * 2  # Width of all 3 bars for one model
            
            # Add horizontal grid lines
            for y in np.arange(y_min, y_max + 0.05, 0.05):
                ax.axhline(y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0)
            
            # Handle y-axis visibility
            if ax.get_position().x0 > 0.1:
                ax.spines["left"].set_visible(False)
                ax.yaxis.set_visible(False)
            else:
                ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))
            
            # Add zero line with higher visibility
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.5, zorder=1)
            
            # Plot bars for each model
            for i, model in enumerate(model_labels.keys()):
                model_data = data[data["model"] == model]
                if len(model_data) == 0:
                    continue
                
                # Base position for this model's group of bars
                x_base = np.arange(n_metrics) * 0.9 + (i - (n_models - 1) / 2) * (group_width + bar_gap)
                
                # Get values for each mitigation type
                for j, mitigation_type in enumerate(["reweighted", "eodd", "adv"]):
                    values = []
                    for metric_type in unique_metrics:
                        mask = (model_data["type"] == mitigation_type) & (model_data["metric_type"] == metric_type)
                        value = model_data[mask]["diff_value"].iloc[0] if len(model_data[mask]) > 0 else 0
                        values.append(value)
                    
                    # Position for this specific bar within the model's group
                    x_positions = x_base + j * (width + bar_gap)
                    
                    # Get color for this model and mitigation type
                    bar_color = colors[model][mitigation_type]
                    
                    # Plot bars
                    plt.bar(x_positions, values, width=width, color=bar_color,
                           label=f"{model_labels[model]} {mitigation_type.capitalize()}", zorder=2)
            
            plt.xticks(np.arange(n_metrics) * 0.9, unique_metrics)
            plt.setp(ax.get_xticklabels(), weight="bold")
        
        g.map_dataframe(plot_bars)
        
        # Set titles and labels
        col_names = {"gender": "Gender", "age": "Age", "ethnicity": "Race"}
        g.set_titles(template="{col_name}")
        
        for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
            ax.set_title(title, fontweight="bold", pad=20, fontsize=20)
        
        # Add interpreter title
        g.fig.suptitle(metric_labels[interpreter], fontweight="bold", fontsize=28, y=1.1)
        
        # Add y-axis label to the leftmost plot
        g.axes[0, 0].set_ylabel("Bias Change", fontweight="bold")
        
        # Remove individual legends
        for ax in g.axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        
        # Save the plots
        for fmt in ["eps", "pdf", "png"]:
            save_dir = os.path.join(results_dir, f"plot_fairness_change_mitigation", fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/mitigation_comparison_{interpreter}.{fmt}",
                bbox_inches="tight",
                dpi=300,
                format=fmt,
            )
        plt.close()
    
    # Create separate legend figure
    plt.figure(figsize=(14, 2.5))
    ax = plt.gca()
    ax.set_axis_off()
    
    # Define gray tones for mitigation types (from light to dark)
    mitigation_colors = {
        "reweighted": "#CCCCCC",  # Light gray
        "eodd": "#888888",        # Medium gray
        "adv": "#444444"          # Dark gray
    }

    # Create legend elements
    legend_elements = []

    # Order the models as requested: unet, pix2pix, sde
    ordered_models = ["unet", "pix2pix", "sde"]

    # First add model elements with their middle color
    for model in ordered_models:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color=colors[model]["eodd"], 
                         label=f"{model_labels[model]}")
        )

    # Add a small space in legend
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', label=""))

    # Then add mitigation types with gray tones (from light to dark)
    for mitigation_type, label in [
        ("reweighted", "Reweighted"),  # Light
        ("eodd", "EODD"),              # Medium
        ("adv", "Adversarial")         # Dark
    ]:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color=mitigation_colors[mitigation_type], 
                         label=label)
        )

    # Create legend with all elements in a single row
    plt.legend(handles=legend_elements, loc="center", ncol=len(legend_elements),
              bbox_to_anchor=(0.5, 0.5))

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, f"plot_fairness_change_mitigation", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/mitigation_comparison_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

    # Create the LaTeX grid
    create_latex_grid("plot_fairness_change_mitigation", results_dir)

def table_performance_evaluation_ucsf_mitigation(results_dir, csv_standard, csv_eodd, csv_reweighted, csv_adv):
    # create new df object empty 
    results = pd.DataFrame()

    # Safely add rows from each dataframe
    def add_df_rows(df, mitigation_type):
        df_copy = df.copy()
        df_copy["mitigation"] = mitigation_type
        return df_copy
    
    # Combine all the dataframes
    dfs_to_combine = [
        add_df_rows(csv_standard, "standard"),
        add_df_rows(csv_eodd, "eodd"),
        add_df_rows(csv_reweighted, "re"),
        add_df_rows(csv_adv, "adv")
    ]
    
    # Concatenate all dataframes at once
    results = pd.concat(dfs_to_combine, ignore_index=True)
    
    latex_path = "latex/templates/ucsf_performance_mitigation.tex"

    # Color definitions for positive and negative values
    colors = {
        "negative": {
            "significant": "E6C321",  # change > 5%
            "marginal": "F1D892",  # 5% <= change
        },
        "positive": {
            "significant": "3089A2",  # change > 5%
            "marginal": "93C1C9",  # 5% <= change
        },
    }

    models = ["baseline", "unet", "pix2pix", "sde"]

    metrics = [
        "tgrade",
        "ttype",
        "dice",
        "psnr",
        "lpips"
    ]

    mitigations = ["eodd", "re", "adv", "standard"]

    # Read the LaTeX file
    file_name = latex_path.split("/")[-1].split(".")[0]
    with open(latex_path, "r") as file:
        latex_content = file.read()

    for metric in metrics:
        for model in models:
            if model == "baseline":
                key = f"{model}-{metric}"

                filtered_results = results[
                        (results["metric"] == metric)
                        & (results["model"] == model)
                    ]

                if len(filtered_results) == 0:
                    continue

                result = filtered_results["value"].iloc[0]

                latex_content = latex_content.replace(key, f"{result:.3f}")

            else:
                for mitigation in mitigations:
                    key = f"{model}-{metric}-{mitigation}"

                    filtered_results = results[
                        (results["metric"] == metric)
                        & (results["model"] == model)
                        & (results["mitigation"] == mitigation)
                    ]

                    filter_model = model

                    baseline_filtered = results[
                        (results["metric"] == metric)
                        & (results["model"] == filter_model)
                        & (results["mitigation"] == "standard")
                    ]

                    # Skip if either filtered results or baseline results don't exist
                    if len(filtered_results) == 0 or len(baseline_filtered) == 0:
                        # Replace with empty cell if no data
                        # current_content = current_content.replace(key, "-")
                        print(key)
                        continue
                    
                    if len(filtered_results) > 1:
                        raise ValueError(f"Multiple results found for {key}")

                    baseline_result = baseline_filtered["value"].iloc[0] if mitigation != "standard" else 0
                    result = filtered_results["value"].iloc[0]

                    percent_change = (result - baseline_result) / baseline_result if baseline_result != 0 else 0

                    if metric == "lpips": 
                        percent_change = -percent_change

                    # Determine color based on percent change
                    color = None
                    if percent_change > 0:
                        if percent_change > 0.1:
                            color = colors["positive"]["significant"]
                        elif percent_change > 0.05:
                            color = colors["positive"]["marginal"]
                    else:
                        if percent_change < -0.1:
                            color = colors["negative"]["significant"]
                        elif percent_change < -0.05:
                            color = colors["negative"]["marginal"]

                    # Format the value
                    formatted_value = f"{result:.3f}"

                    # Add color if significant
                    if color:
                        formatted_value = (
                            f"\\cellcolor[HTML]{{{color}}}{formatted_value}"
                        )

                    latex_content = latex_content.replace(key, formatted_value)

    # Write to latex
    with open(os.path.join(results_dir, f"table_performance_evaluation_ucsf_mitigation.tex"), "w") as file:
        file.write(latex_content)

def table_performance_evaluation_chex_mitigation(results_dir, csv_standard, csv_eodd, csv_reweighted, csv_adv):
    # create new df object empty 
    results = pd.DataFrame()

    # Safely add rows from each dataframe
    def add_df_rows(df, mitigation_type):
        df_copy = df.copy()
        df_copy["mitigation"] = mitigation_type
        return df_copy
    
    # Combine all the dataframes
    dfs_to_combine = [
        add_df_rows(csv_standard, "standard"),
        add_df_rows(csv_eodd, "eodd"),
        add_df_rows(csv_reweighted, "re"),
        add_df_rows(csv_adv, "adv")
    ]
    
    # Concatenate all dataframes at once
    results = pd.concat(dfs_to_combine, ignore_index=True)
    
    latex_path = "latex/templates/chex_performance_mitigation.tex"

    # Color definitions for positive and negative values
    colors = {
        "negative": {
            "significant": "E6C321",  # change > 5%
            "marginal": "F1D892",  # 5% <= change
        },
        "positive": {
            "significant": "3089A2",  # change > 5%
            "marginal": "93C1C9",  # 5% <= change
        },
    }

    models = ["baseline", "unet", "pix2pix", "sde"]

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
        "psnr",
        "lpips"
    ]

    mitigations = ["eodd", "re", "adv", "standard"]

    # Read the LaTeX file
    file_name = latex_path.split("/")[-1].split(".")[0]
    with open(latex_path, "r") as file:
        latex_content = file.read()

    for metric in metrics:
        for model in models:
            if model == "baseline":
                key = f"{model}-{metric}"

                filtered_results = results[
                        (results["metric"] == metric)
                        & (results["model"] == model)
                    ]

                if len(filtered_results) == 0:
                    continue

                result = filtered_results["value"].iloc[0]

                latex_content = latex_content.replace(key, f"{result:.3f}")

            else:
                for mitigation in mitigations:
                    key = f"{model}-{metric}-{mitigation}"

                    filtered_results = results[
                        (results["metric"] == metric)
                        & (results["model"] == model)
                        & (results["mitigation"] == mitigation)
                    ]

                    filter_model = model

                    baseline_filtered = results[
                        (results["metric"] == metric)
                        & (results["model"] == filter_model)
                        & (results["mitigation"] == "standard")
                    ]

                    # Skip if either filtered results or baseline results don't exist
                    if len(filtered_results) == 0 or len(baseline_filtered) == 0:
                        # Replace with empty cell if no data
                        # current_content = current_content.replace(key, "-")
                        print(key)
                        continue
                    
                    if len(filtered_results) > 1:
                        raise ValueError(f"Multiple results found for {key}")

                    baseline_result = baseline_filtered["value"].iloc[0] if mitigation != "standard" else 0
                    result = filtered_results["value"].iloc[0]

                    percent_change = (result - baseline_result) / baseline_result if baseline_result != 0 else 0

                    if metric == "lpips": 
                        percent_change = -percent_change

                    # Determine color based on percent change
                    color = None
                    if percent_change > 0:
                        if percent_change > 0.1:
                            color = colors["positive"]["significant"]
                        elif percent_change > 0.05:
                            color = colors["positive"]["marginal"]
                    else:
                        if percent_change < -0.1:
                            color = colors["negative"]["significant"]
                        elif percent_change < -0.05:
                            color = colors["negative"]["marginal"]

                    # Format the value
                    formatted_value = f"{result:.3f}"

                    # Add color if significant
                    if color:
                        formatted_value = (
                            f"\\cellcolor[HTML]{{{color}}}{formatted_value}"
                        )

                    latex_content = latex_content.replace(key, formatted_value)

    # Write to latex
    with open(os.path.join(results_dir, f"table_performance_evaluation_chex_mitigation.tex"), "w") as file:
        file.write(latex_content)

def table_fairness_evaluation_standard(combined_results, results_dir):

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
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{\\phantom{{-}}{result:.3f}}}}}"
                                    else:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{{result:.3f}}}}}"
                                else:
                                    if result > 0:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{\\phantom{{-}}{result:.3f}}}"
                                    else:
                                        content += f"& \\cellcolor[HTML]{{{color}}}{{{result:.3f}}}"
                            else:
                                if abs(std_error) > abs(result):
                                    if result > 0:
                                        content += f"& \\textbf{{\\phantom{{-}}{result:.3f}}}"
                                    else:
                                        content += f"& \\textbf{{{result:.3f}}}"
                                else:
                                    if result > 0:
                                        content += f"& \\phantom{{-}}{result:.3f}"
                                    else:
                                        content += (
                                            f"& {result:.3f}"
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
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{\\phantom{{-}}{result:.3f}}}}}"
                                else:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{\\textbf{{{result:.3f}}}}}"
                            else:
                                if result > 0:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{\\phantom{{-}}{result:.3f}}}"
                                else:
                                    content += f"& \\cellcolor[HTML]{{{color}}}{{{result:.3f}}}"
                        else:
                            if abs(std_error) > abs(result):
                                if result > 0:
                                    content += f"& \\textbf{{\\phantom{{-}}{result:.3f}}}"
                                else:
                                    content += f"& \\textbf{{{result:.3f}}}"
                            else:
                                if result > 0:
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

    with open(os.path.join(results_dir, f"table_fairness_evaluation_standard.tex"), "w") as file:
        file.write(new_latex_content)

def create_shared_legend(results_dir):
    """
    Create a shared legend figure for the fairness performance scatter plots.
    
    Args:
        results_dir: Directory to save the legend
    """
    # Model and dataset name mappings
    model_map = {
        "unet": "U-Net",
        "pix2pix": "Pix2Pix",
        "sde": "SDE"
    }

    dataset_map = {
        "chex": "CheXpert",
        "ucsf": "UCSF"
    }
    
    # Define colors for models
    model_colors = {
        "pix2pix": "#E6C321",  # Yellow
        "unet": "#3089A2",     # Blue
        "sde": "#EB0007"       # Red
    }
    
    # Define dataset styles (outline or filled)
    dataset_markers = {
        "chex": "s", 
        "ucsf": "o" 
    }

    # Create figure for legend
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_axis_off()  # Hide axes
    
    # Create legend elements
    legend_elements = []
    
    # Model colors
    for model in sorted(model_colors.keys()):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=model_colors[model], markersize=15, 
                      label=f"Model: {model_map.get(model, model)}")
        )
    
    # Dataset styles
    for dataset in sorted(dataset_markers.keys()):
        legend_elements.append(
            plt.Line2D([0], [0], marker=dataset_markers[dataset], color='gray', 
                      markersize=15, label=f"Dataset: {dataset_map.get(dataset, dataset)}")
        )
    
    # Example size indicators
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=6, label="Fairness: Low")
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, label="Fairness: Medium")
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=15, label="Fairness: High")
    )
    
    # Create the legend
    plt.legend(handles=legend_elements, loc='center', frameon=True, 
              fancybox=True, shadow=True, ncol=1, fontsize=18)
    
    # Save the legend
    for fmt in ["eps", "pdf", "png"]:
        os.makedirs(os.path.join(results_dir, fmt), exist_ok=True)
        plt.savefig(
            os.path.join(results_dir, fmt, f"fairness_performance_legend.{fmt}"),
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    
    plt.close()

def fairness_performance_scatter(df, results_dir, vertical_lines, mitigation_type="reweighting", fairness_metric="EODD", attribute="age"):
    """
    Create a scatter plot showing the trade-off between fairness and performance changes.
    
    Args:
        df: DataFrame with the following columns:
            - metric: Metric (string)
            - fairness_percent: Percentage change in fairness
            - fairness_val: Absolute fairness value
            - performance_percent: Percentage change in performance
            - dataset: Dataset type (string)
            - model: Model name (string)
        vertical_lines: DataFrame with columns:
            - performance_percent: x-position of vertical line
            - model: Model name for color
            - dataset: Dataset name for line style
        results_dir: Directory to save the plots
    """
    
    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18,
    })

    model_colors = {
        "pix2pix": "#E6C321",
        "unet": "#3089A2",
        "sde": "#EB0007"
    }

    dataset_markers = {
        "chex": "s",
        "ucsf": "o"
    }

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
        "average": "Average",
        "tgrade": "Tumor Grade",
        "ttype": "Tumor Type",
    }
    # Define line styles for datasets - using markers in the lines to match scatter plot
    dataset_lines = {
        "chex": {
            "linestyle": "-",       # solid line
            "marker": "s",          # square marker
            "markevery": [0],       # place marker at the start (bottom)
            "markersize": 10,       # larger marker size
            "fillstyle": "none"     # hollow markers
        },
        "ucsf": {
            "linestyle": "-",       # solid line
            "marker": "o",          # circle marker
            "markevery": [0],       # place marker at the start (bottom)
            "markersize": 10,       # larger marker size
            "fillstyle": "none"     # hollow markers
        }
    }

    plt.figure(figsize=(12, 10))

    # First plot vertical lines (so they appear behind the scatter points)
    for _, line in vertical_lines.iterrows():
        color = model_colors.get(line['model'], "#777777")
        line_style = dataset_lines.get(line['dataset'], {})
        
        # Plot vertical line from bottom to top of plot
        plt.axvline(x=line['performance_percent'], 
                   color=color, 
                   linestyle=line_style.get('linestyle', '-'),
                   linewidth=2.0,    # Make lines thicker
                   alpha=0.4,        # Slightly more visible
                   zorder=1,         # Ensure lines are behind points
                   marker=line_style.get('marker', 'o'),
                   markevery=line_style.get('markevery', [0]),
                   markersize=line_style.get('markersize', 10),
                   fillstyle=line_style.get('fillstyle', 'none'),
                   markeredgecolor=color)

    # Rest of the plotting code...
    min_fairness = df['fairness_val'].min()
    max_fairness = df['fairness_val'].max()

    def normalize_size(val):
        if max_fairness == min_fairness:
            return 150
        return 50 + 250 * (val - min_fairness) / (max_fairness - min_fairness)

    # Calculate statistics to identify outliers
    fairness_mean = df['fairness_percent'].mean()
    fairness_std = df['fairness_percent'].std()
    performance_mean = df['performance_percent'].mean()
    performance_std = df['performance_percent'].std()
    
    # Define outlier thresholds (points beyond 2 standard deviations)
    fairness_threshold = 1.5
    performance_threshold = 1.5

    # Collect all outliers to identify the most extreme ones
    outliers = []
    
    # Plot scatter points with higher zorder to ensure they're on top of lines
    for idx, row in df.iterrows():
        model = row['model']
        dataset = row['dataset']
        metric = row['metric']
        fairness_pct = row['fairness_percent']
        performance_pct = row['performance_percent']
        fairness_val = row['fairness_val']

        color = model_colors.get(model, "#777777")
        marker = dataset_markers.get(dataset, "D")
        size = normalize_size(fairness_val)

        plt.scatter(performance_pct, fairness_pct, s=size, marker=marker, 
                   color=color, zorder=2)  # Ensure points are above lines
        
        # Check if this point is an outlier
        fairness_z = abs(fairness_pct - fairness_mean) / fairness_std if fairness_std > 0 else 0
        performance_z = abs(performance_pct - performance_mean) / performance_std if performance_std > 0 else 0
        
        if fairness_z > fairness_threshold or performance_z > performance_threshold:
            outlier_score = fairness_z + performance_z
            outliers.append((outlier_score, idx, fairness_pct, performance_pct, metric, model, dataset))

    # Sort outliers and add labels (with highest zorder to be on top)
    outliers.sort(reverse=True)
    top_outliers = outliers[:min(5, len(outliers))]
    
    for _, idx, fairness_pct, performance_pct, metric, model, dataset in top_outliers:
        label = metric_labels.get(metric, metric)
        
        plt.annotate(label, 
                    xy=(performance_pct, fairness_pct),
                    xytext=(7, 7), 
                    textcoords='offset points',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                    zorder=3)  # Ensure labels are on top

    # Add reference lines
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.6, zorder=0)
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.6, zorder=0)

    plt.xlabel("Performance Change (%)", fontweight="bold")
    plt.ylabel("Fairness Change (%)", fontweight="bold")

    plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

    # Scaling Y-axis to 10 times the X-axis
    xlim = plt.xlim()
    ylim = plt.ylim()
    x_range = xlim[1] - xlim[0]
    y_mid = (ylim[0] + ylim[1]) / 2
    plt.ylim(y_mid - 5 * x_range, y_mid + 5 * x_range)

    # Add the 10× label at the corner of the y-axis
    plt.text(0.02, 0.98, "10×", transform=plt.gca().transAxes, 
             fontsize=18, fontweight='bold', ha='left', va='top')

    plt.tight_layout()

    for fmt in ["eps", "pdf", "png"]:
        os.makedirs(os.path.join(results_dir, fmt), exist_ok=True)
        plt.savefig(
            os.path.join(results_dir, fmt, f"fairness_performance_tradeoff_{mitigation_type}_{fairness_metric}_{attribute}.{fmt}"),
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )

    plt.close()


def _get_performance_change(df, model, metric, dataset, mitigation): 
    mitigation_df = df[
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["metric"] == metric)
        & (df["mitigation"] == mitigation)
    ]

    standard_df = df[
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["metric"] == metric)
        & (df["mitigation"] == "standard")
    ]

    if len(mitigation_df) == 0 or len(standard_df) == 0:
        #print(f"No data found for model: {model}, dataset: {dataset}, metric: {metric}, mitigation: {mitigation}")
        return None

    mitigation_value = mitigation_df["value"].iloc[0]
    standard_value = standard_df["value"].iloc[0]

    # return the percentage change
    return (mitigation_value - standard_value) / standard_value * 100
    

def _get_fairness_change(df, model, interpreter, attribute, metric, dataset, mitigation): 
    mitigation_df = df[
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["metric"] == metric)
        & (df["mitigation"] == mitigation)
        & (df["attribute"] == attribute)
        & (df["interpreter"] == interpreter)
    ]

    standard_df = df[
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["metric"] == metric)
        & (df["mitigation"] == "standard")
        & (df["attribute"] == attribute)
        & (df["interpreter"] == interpreter)
    ]
    if len(mitigation_df) == 0 or len(standard_df) == 0:
        #print(f"No data found for model: {model}, dataset: {dataset}, metric: {metric}, mitigation: {mitigation}, attribute: {attribute}, interpreter: {interpreter}")
        return None, None

    mitigation_value = mitigation_df["value"].iloc[0]
    standard_value = standard_df["value"].iloc[0]

    return (mitigation_value - standard_value) / standard_value * 100, mitigation_value

def plot_fairness_performance_tradeoff(performance_df, fairness_df, results_dir):
    """
    Plot the trade-off between fairness and performance changes.
    
    Args:
        performance_df: DataFrame with performance data
        fairness_df: DataFrame with fairness data
        results_dir: Directory to save the plots
    """
    performance_df_og = performance_df.copy()
    fairness_df_og = fairness_df.copy()
    results_dir = os.path.join(results_dir, "plot_fairness_performance_tradeoff")
    os.makedirs(results_dir, exist_ok=True)

    # Create a shared legend first
    create_shared_legend(results_dir)

    fairness_metrics = ["EODD-bootstrapped", "EOP-bootstrapped"]
    interpreters = [
        "ec", "cardiomegaly", "lung-opacity", "lung-lesion", "edema",
        "consolidation", "pneumonia", "atelectasis", "pneumothorax",
        "pleural-effusion", "pleural-other", "fracture", "ttype", "tgrade"
    ]
    attributes = [
        "gender", "age", "ethnicity"
    ]

    for mitigation in performance_df["mitigation"].unique():
        if mitigation == "standard": 
            continue

        for fairness_metric in fairness_metrics:
            performance_df = performance_df_og.copy()

            for attribute in attributes:
                # at the end of this loop we call the scatter plot function
                fairness_df = fairness_df_og.copy()

                # Create a list to collect rows instead of using append
                rows_list = []
                vertical_lines = []

                for model in performance_df["model"].unique():
                    if model == "baseline":
                        continue
                    for dataset in performance_df["dataset"].unique():

                        pnsr_performance_change = _get_performance_change(performance_df, model, "psnr", dataset, mitigation)
                        vertical_lines.append({
                            "performance_percent": pnsr_performance_change,
                            "model": model,
                            "dataset": dataset,
                        })

                        for interpreter in interpreters:
                            performance_change = _get_performance_change(performance_df, model, interpreter, dataset, mitigation)
                            fairness_change, fairness_val = _get_fairness_change(fairness_df, model, interpreter, attribute, fairness_metric, dataset, mitigation)

                            if performance_change is None or fairness_change is None:
                                continue

                            # Add to rows list instead of appending
                            rows_list.append({
                                "metric": interpreter,
                                "fairness_percent": fairness_change,
                                "fairness_val": fairness_val,
                                "performance_percent": performance_change,
                                "dataset": dataset,
                                "attribute": attribute,
                                "model": model
                            })

                # Create the DataFrame from the list of rows
                df = pd.DataFrame(rows_list)
                vertical_lines = pd.DataFrame(vertical_lines)



                # Only create plot if we have data
                if not df.empty:
                    fairness_performance_scatter(df, results_dir, vertical_lines, mitigation, fairness_metric, attribute)

    # After all plots are generated, create the LaTeX grid
    create_fairness_performance_latex_grid(results_dir)

def create_fairness_performance_latex_grid(results_dir):
    """
    Create a LaTeX figure with a grid of fairness-performance plots organized by
    mitigation type and attribute, allowing it to split across pages.
    
    Args:
        results_dir: Directory containing the plots and where to save the LaTeX file
    """
    # Define the structure of plots to include
    mitigations = ["reweighted", "eodd", "adv"]
    mitigation_labels = {
        "reweighted": "Reweighting",
        "eodd": "Equalized Odds",
        "adv": "Adversarial"
    }
    
    fairness_metrics = ["EODD", "EOP"]
    attributes = ["age", "gender", "ethnicity"]
    
    latex_content = []
    
    # Start with document content instead of a figure environment
    # This allows splitting across pages
    
    # Loop through each mitigation type
    for mitigation in mitigations:
        # Add a subsection for this mitigation type        
        # Create a 2x3 grid for each mitigation type (2 metrics × 3 attributes)
        for metric in fairness_metrics:
            # Start a figure for this row
            latex_content.append(r"\begin{figure}[ht]")
            latex_content.append(r"\centering")
            
            # Add plots for each attribute side by side with tabular
            latex_content.append(r"\begin{tabular}{c@{\hspace{0.2cm}}c@{\hspace{0.2cm}}c}")  # Reduced spacing
            
            for i, attribute in enumerate(attributes):
                plot_path = f"GR/fig/fairness_performance_tradeoff/fairness_performance_tradeoff_{mitigation}_{metric}-bootstrapped_{attribute}.pdf"
                
                # Add plot
                latex_content.append(r"\begin{subfigure}[t]{0.3\textwidth}")
                latex_content.append(r"    \centering")
                latex_content.append(r"    \includegraphics[width=\linewidth]{" + plot_path + "}")
                latex_content.append(r"    \caption{" + f"{attribute.capitalize()}" + "}")
                latex_content.append(r"\end{subfigure}")
                
                # Add separator between columns (except for the last column)
                if i < len(attributes) - 1:
                    latex_content.append(r"&")
            
            # End tabular
            latex_content.append(r"\end{tabular}")
            
            # Add caption for this row
            latex_content.append(r"\caption{" + f"{mitigation_labels[mitigation]} Mitigation: {metric} Metric across Attributes" + "}")
            latex_content.append(r"\end{figure}")
            latex_content.append("")
        
        # Add minimal space between mitigation types
        latex_content.append(r"\vspace{0.2cm}")  # Greatly reduced spacing
        latex_content.append("")
    
    # Add the legend as a separate figure at the end
    latex_content.append(r"\begin{figure}[ht]")
    latex_content.append(r"\centering")
    latex_content.append(r"\includegraphics[width=0.7\textwidth]{GR/fig/fairness_performance_tradeoff/fairness_performance_legend.pdf}")
    latex_content.append(r"\caption{Legend for Fairness-Performance Trade-off Plots}")
    latex_content.append(r"\label{fig:fairness-performance-legend}")
    latex_content.append(r"\end{figure}")
    
    # Write to file
    os.makedirs(os.path.join(results_dir, "latex"), exist_ok=True)
    with open(os.path.join(results_dir, "latex", "fairness_performance_grid.tex"), "w") as f:
        f.write("\n".join(latex_content))
    
    print(f"LaTeX grid saved to {os.path.join(results_dir, 'latex', 'fairness_performance_grid.tex')}")

def table_fairness_evaluation_mitigation(combined_results, results_dir):

    latex_path = "latex/templates/combined_fairness_colors_mitigation.tex"
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

    models = ["unet", "pix2pix", "sde"]
    metrics = ["EODD", "EOP"]

    segmentation_metrics = ["SER", "delta-dice"]

    attributes = ["gender", "age", "ethnicity"]
    attribute_mapping = {
        "gender": "gender",
        "age": "age",
        "ethnicity": "race",
    }
    mitigations = ["reweighted", "eodd", "adv"]

    mitigation_mapping = {
        "reweighted": "Reweighting",
        "eodd": "Equalized Odds",
        "adv": "Adversarial",
    }

    attribute_contents = []

    for mitigation in mitigations:
        mitigation_content = latex_content.replace(
            "-mitigation-", mitigation_mapping[mitigation]
        )
        for attribute in attributes:
            attribute_content = mitigation_content.replace(
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
                                & (combined_results["mitigation"] == mitigation)
                            ]
                            result = filtered_results["value"].iloc[0]

                            if len(filtered_results) == 0:
                                continue


                            filtered_p_values = combined_results[
                                (combined_results["metric"] == f"delta-{metric}-p-value")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == interpreter)
                                & (combined_results["attribute"] == attribute)
                                & (combined_results["mitigation"] == mitigation)
                            ]

                            filtered_std_errors = combined_results[
                                (combined_results["metric"] == f"delta-{metric}-std-err")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == interpreter)
                                & (combined_results["attribute"] == attribute)
                                & (combined_results["mitigation"] == mitigation)
                            ]
                            filtered_delta_result = combined_results[
                                (combined_results["metric"] == f"delta-{metric}-bootstrapped")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == interpreter)
                                & (combined_results["attribute"] == attribute)
                                & (combined_results["mitigation"] == mitigation)    
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
                                content += f"& \\cellcolor[HTML]{{{color}}}{{{result:.3f}$\\pm${std_error:.3f}}}"
                            else:
                                content += f"& {result:.3f}$\\pm${std_error:.3f}"

                    classifier_content += content + "\\\\ \n"

            attribute_content = attribute_content.replace(
                "-classifier-", classifier_content
            )

            if attribute in ["gender", "age"]:
                segmentation_content = """
                                                             &                                            &                                                   &                                            &                                                   &                                            &                                                   \\\\ \\hline
                \\multicolumn{1}{l|}{} & \\multicolumn{1}{c}{\\textbf{SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta$ Dice}} & \\multicolumn{1}{c}{\\textbf{SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta$ Dice}} & \\multicolumn{1}{c}{\\textbf{$SER}}  & \\multicolumn{1}{c}{\\textbf{$\\Delta$ Dice}} \\\\ \\hline
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
                            & (combined_results["mitigation"] == mitigation)
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
                                & (combined_results["mitigation"] == mitigation)
                            ]

                            filtered_std_errors = combined_results[
                                (combined_results["metric"] == f"delta-{metric}-std-err")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == "dice")
                                & (combined_results["attribute"] == attribute)
                                & (combined_results["mitigation"] == mitigation)
                            ]
                            filtered_delta_result = combined_results[
                                (combined_results["metric"] == f"delta-{metric}-bootstrapped")
                                & (combined_results["model"] == model)
                                & (combined_results["interpreter"] == "dice")
                                & (combined_results["attribute"] == attribute)
                                & (combined_results["mitigation"] == mitigation)
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
                                content += f"& \\cellcolor[HTML]{{{color}}}{{{result:.3f}$\\pm${std_error:.3f}}}" 
                            else:

                                content += f"& {result:.3f}$\\pm${std_error:.3f}"

                segmentation_content += "\n" + content + "\\\\"

                attribute_content += segmentation_content

            attribute_content += """    
            \end{tabular} 
            } 
            \end{table}"""

            legend = """\\\\[0.3cm]
                \\begin{tabular}{llllllll} 
                \\cellcolor[HTML]{E6C321} & $+$, $p < 0.05$ & \\cellcolor[HTML]{F1D892} &$+$, $0.05 \\leq p < 0.1$ & \\cellcolor[HTML]{3089A2} & $-$, $p < 0.05$  & \\cellcolor[HTML]{93C1C9} & $-$, $0.05 \\leq p < 0.1$ \\\\
                \\end{tabular}"""
            attribute_content = attribute_content.replace("-legend-", legend)

            attribute_contents.append(attribute_content)

    for content in attribute_contents:
        new_latex_content += content

    with open(os.path.join(results_dir, f"table_fairness_evaluation_mitigation.tex"), "w") as file:
        file.write(new_latex_content)