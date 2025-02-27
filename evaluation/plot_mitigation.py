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

def plot_mitigation_chex(original_df, mitigated_df, results_dir, run_name):
    interpreter_configs_original = {
        "metrics": ["delta-EODD-bootstrapped", "delta-EOP-bootstrapped"],
        "x_labels": ["EODD", "EOP"],
    }

    interpreters = [
        "ec", "cardiomegaly", "lung-opacity", "lung-lesion", "edema",
        "consolidation", "pneumonia", "atelectasis", "pneumothorax",
        "pleural-effusion", "pleural-other", "fracture", "average",
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
        "average": "Average",
    }

    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}

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
        # Prepare data for current interpreter
        orig_data_task = original_df[original_df["interpreter"] == interpreter].copy()
        mitig_data_task = mitigated_df[mitigated_df["interpreter"] == interpreter].copy()
        
        # Create combined data for facet grid
        plot_data = pd.DataFrame()
        for metric in interpreter_configs_original["metrics"]:
            orig_metric = orig_data_task[orig_data_task["metric"] == metric].copy()
            orig_metric["type"] = "original"
            mitig_metric = mitig_data_task[mitig_data_task["metric"] == metric].copy()
            mitig_metric["type"] = "mitigated"
            
            plot_data = pd.concat([plot_data, orig_metric, mitig_metric])
        
        # Calculate y limits with padding
        y_min = plot_data["value"].min()
        y_max = plot_data["value"].max()
        
        # Round to nearest 0.05
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
            unique_metrics = interpreter_configs_original["x_labels"]
            n_metrics = len(unique_metrics)
            n_models = len(model_labels)
            width = 0.12
            bar_gap = 0.005
            
            # Add horizontal grid lines
            for y in np.arange(y_min, y_max + 0.05, 0.05):
                ax.axhline(y=y, color="gray", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0)
            
            # Handle y-axis visibility
            if ax.get_position().x0 > 0.1:
                ax.spines["left"].set_visible(False)
                ax.yaxis.set_visible(False)
            else:
                ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))
            
            # Plot bars for each model
            for i, model in enumerate(model_labels.keys()):
                x_positions = np.arange(n_metrics) * 0.8 + (i - (n_models - 1) / 2) * (width * 2 + bar_gap)
                
                # Get data for current model
                model_data = data[data["model"] == model]
                
                # Plot original values
                orig_values = []
                for metric in interpreter_configs_original["metrics"]:
                    mask = (model_data["type"] == "original") & (model_data["metric"] == metric)
                    value = model_data[mask]["value"].iloc[0] if len(model_data[mask]) > 0 else 0
                    orig_values.append(value)
                
                # Plot mitigated values
                mitig_values = []
                for metric in interpreter_configs_original["metrics"]:
                    mask = (model_data["type"] == "mitigated") & (model_data["metric"] == metric)
                    value = model_data[mask]["value"].iloc[0] if len(model_data[mask]) > 0 else 0
                    mitig_values.append(value)
                
                # Determine colors based on model
                if model == "unet":
                    orig_color = colors["unet"]["significant"]
                    mitig_color = colors["unet"]["marginal"]
                elif model == "pix2pix":
                    orig_color = colors["pix2pix"]["significant"]
                    mitig_color = colors["pix2pix"]["marginal"]
                else:  # sde
                    orig_color = colors["sde"]["significant"]
                    mitig_color = colors["sde"]["marginal"]

                # Plot original bars
                plt.bar(x_positions, orig_values, width=width, color=orig_color, 
                       label=f"{model_labels[model]} Original", zorder=2)
                # Plot mitigated bars
                plt.bar(x_positions + width, mitig_values, width=width, color=mitig_color,
                       label=f"{model_labels[model]} Mitigated", zorder=2)

            plt.xticks(np.arange(n_metrics) * 0.8, unique_metrics)
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
        g.axes[0, 0].set_ylabel("Additional Bias", fontweight="bold")

        # Remove individual legends
        for ax in g.axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Save the plots
        for fmt in ["eps", "pdf", "png"]:
            save_dir = os.path.join(results_dir, "chex_mitigation", fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/mitigation_bias_{interpreter}.{fmt}",
                bbox_inches="tight",
                dpi=300,
                format=fmt,
            )
        plt.close()

    # Create separate legend figure
    plt.figure(figsize=(12, 1))
    ax = plt.gca()
    ax.set_axis_off()

    # Create legend elements for all combinations
    legend_elements = []
    for model in model_labels.keys():
        if model == "unet":
            orig_color = colors["unet"]["significant"]
            mitig_color = colors["unet"]["marginal"]
        elif model == "pix2pix":
            orig_color = colors["pix2pix"]["significant"]
            mitig_color = colors["pix2pix"]["marginal"]
        else:  # sde
            orig_color = colors["sde"]["significant"]
            mitig_color = colors["sde"]["marginal"]
            
        legend_elements.extend([
            plt.Rectangle((0, 0), 1, 1, color=orig_color, label=f"{model_labels[model]} Original"),
            plt.Rectangle((0, 0), 1, 1, color=mitig_color, label=f"{model_labels[model]} Mitigated")
        ])

    # Create horizontal legend
    plt.legend(handles=legend_elements, loc="center", ncol=len(legend_elements),
              bbox_to_anchor=(0.5, 0.5))

    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, "chex_mitigation", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/mitigation_bias_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

    # After all plots are created, generate the LaTeX grid
    create_latex_grid(run_name, results_dir)

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
    latex_content.append(r"        \includegraphics[width=\linewidth]{" + f"fig/{run_name}/mitigation_bias_legend.pdf" + "}")
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
        latex_content.append(r"        \begin{subfigure}[t]{0.45\textwidth}")
        latex_content.append(r"            \centering")
        latex_content.append(r"            \includegraphics[width=\linewidth]{" + f"fig/{run_name}/mitigation_bias_{interpreter1}.pdf" + "}")
        latex_content.append(r"        \end{subfigure}")
        latex_content.append(r"        &")
        
        # Add second plot
        latex_content.append(r"        \begin{subfigure}[t]{0.45\textwidth}")
        latex_content.append(r"            \centering")
        latex_content.append(r"            \includegraphics[width=\linewidth]{" + f"fig/{run_name}/mitigation_bias_{interpreter2}.pdf" + "}")
        latex_content.append(r"        \end{subfigure}")
        
        # Add line break unless it's the last pair
        if i < len(interpreters)-3:  # -3 because we're excluding average
            latex_content.append(r"        \\[0.3cm]")
        latex_content.append("")

    latex_content.append(r"    \end{tabular}")
    latex_content.append("")
    
    # Add average plot centered below the grid
    latex_content.append(r"    \vspace{0.5cm}")
    latex_content.append(r"    \begin{subfigure}[t]{0.45\textwidth}")
    latex_content.append(r"        \centering")
    latex_content.append(r"        \includegraphics[width=\linewidth]{" + f"fig/{run_name}/mitigation_bias_average.pdf" + "}")
    latex_content.append(r"    \end{subfigure}")
    latex_content.append("")
    
    # Add caption and label
    latex_content.append(r"\end{figure}")
    
    # Write to file
    os.makedirs(os.path.join(results_dir, "latex"), exist_ok=True)
    with open(os.path.join(results_dir, "latex", "mitigation_grid.tex"), "w") as f:
        f.write("\n".join(latex_content))

def plot_combined_mitigation(original_df, reweighted_df, mitigated_df, results_dir, run_name):
    """
    Create a faceted bar plot showing the difference between reweighted vs. original and mitigated vs. original
    for EODD and EOP metrics across different models and attributes.
    
    Args:
        original_df: DataFrame with original results
        reweighted_df: DataFrame with reweighted results
        mitigated_df: DataFrame with mitigated results
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
        "pleural-effusion", "pleural-other", "fracture", "average",
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
    }
    
    # Define model labels
    model_labels = {"unet": "U-Net", "pix2pix": "Pix2Pix", "sde": "SDE"}
    
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
        # Filter data for current interpreter
        orig_data = original_df[original_df["interpreter"] == interpreter].copy()
        reweight_data = reweighted_df[reweighted_df["interpreter"] == interpreter].copy()
        mitig_data = mitigated_df[mitigated_df["interpreter"] == interpreter].copy()
        
        if len(orig_data) == 0 or len(reweight_data) == 0 or len(mitig_data) == 0:
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
            
            # Get mitigated data and calculate difference
            mitig_metric = mitig_data[mitig_data["metric"] == metric].copy()
            mitig_metric["type"] = "mitigated"
            mitig_metric["diff_value"] = mitig_metric["value"] - orig_metric["value"].values
            
            # Add to plot data
            plot_data = pd.concat([plot_data, reweight_metric, mitig_metric])
        
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
        
        # Create facet grid
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
            unique_metrics = ["EODD", "EOP"]
            n_metrics = len(unique_metrics)
            n_models = len(model_labels)
            width = 0.12
            bar_gap = 0.005
            
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
                
                x_positions = np.arange(n_metrics) * 0.8 + (i - (n_models - 1) / 2) * (width * 2 + bar_gap)
                
                # Get reweighted values
                reweight_values = []
                for metric_type in unique_metrics:
                    mask = (model_data["type"] == "reweighted") & (model_data["metric_type"] == metric_type)
                    value = model_data[mask]["diff_value"].iloc[0] if len(model_data[mask]) > 0 else 0
                    reweight_values.append(value)
                
                # Get mitigated values
                mitig_values = []
                for metric_type in unique_metrics:
                    mask = (model_data["type"] == "mitigated") & (model_data["metric_type"] == metric_type)
                    value = model_data[mask]["diff_value"].iloc[0] if len(model_data[mask]) > 0 else 0
                    mitig_values.append(value)
                
                # Determine colors based on model
                if model == "unet":
                    reweight_color = colors["unet"]["marginal"]
                    mitig_color = colors["unet"]["significant"]
                elif model == "pix2pix":
                    reweight_color = colors["pix2pix"]["marginal"]
                    mitig_color = colors["pix2pix"]["significant"]
                else:  # sde
                    reweight_color = colors["sde"]["marginal"]
                    mitig_color = colors["sde"]["significant"]
                
                # Plot reweighted bars
                plt.bar(x_positions, reweight_values, width=width, color=reweight_color,
                       label=f"{model_labels[model]} Reweighted", zorder=2)
                
                # Plot mitigated bars
                plt.bar(x_positions + width, mitig_values, width=width, color=mitig_color,
                       label=f"{model_labels[model]} Mitigated", zorder=2)
            
            plt.xticks(np.arange(n_metrics) * 0.8, unique_metrics)
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
        g.axes[0, 0].set_ylabel("Bias Reduction", fontweight="bold")
        
        # Remove individual legends
        for ax in g.axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        
        # Save the plots
        for fmt in ["eps", "pdf", "png"]:
            save_dir = os.path.join(results_dir, f"{run_name}_mitigation_comparison", fmt)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/mitigation_comparison_{interpreter}.{fmt}",
                bbox_inches="tight",
                dpi=300,
                format=fmt,
            )
        plt.close()
    
    # Create separate legend figure
    plt.figure(figsize=(12, 1.5))
    ax = plt.gca()
    ax.set_axis_off()
    
    # Define reweighted and mitigated colors for legend
    reweighted_color = "#CCCCCC"  # Light gray
    mitigated_color = "#666666"   # Dark gray
    
    # Create legend elements
    legend_elements = []
    
    # Add model colors
    for model in model_labels.keys():
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color=colors[model]["significant"], label=f"{model_labels[model]}")
        )
    
    # Add reweighted and mitigated indicators
    legend_elements.extend([
        plt.Rectangle((0, 0), 1, 1, color=reweighted_color, label="Reweighted"),
        plt.Rectangle((0, 0), 1, 1, color=mitigated_color, label="Mitigated")
    ])
    
    # Create horizontal legend
    plt.legend(handles=legend_elements, loc="center", ncol=len(legend_elements),
              bbox_to_anchor=(0.5, 0.5))
    
    # Save legend
    for fmt in ["eps", "pdf", "png"]:
        save_dir = os.path.join(results_dir, f"{run_name}_mitigation_comparison", fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/mitigation_comparison_legend.{fmt}",
            bbox_inches="tight",
            dpi=300,
            format=fmt,
        )
    plt.close()

    create_latex_grid(run_name, results_dir)