import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.lines import Line2D
import numpy as np

model_colors = {
    'unet': '#3089A2', 
    'pix2pix': '#E6C321', 
    'sde': '#EB0007', 
}

def plot_ucsf_performance(results, results_dir, name):
    # Define metric pairs and their labels

    # create new directory for the plots
    os.makedirs(results_dir / 'performance_plots', exist_ok=True)

    metric_pairs = [('tgrade', 'psnr'), ('ttype', 'psnr'), ('dice', 'psnr')]

    metric_labels = {
        'psnr': 'PSNR',
        'tgrade': 'AUROC',
        'ttype': 'AUROC',
        'dice': 'Dice'
    }

    model_labels = {
        'unet': 'U-Net',
        'pix2pix': 'Pix2Pix',
        'sde': 'SDE'
    }
    
    # Set font sizes and line widths
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    })
    
    for metric1, metric2 in metric_pairs:
        # Create new figure for each pair
        plt.figure(figsize=(10, 6))
        
        # Filter data for the current metrics
        data1 = results[(results['metric'] == metric1) & (results['model'] != 'baseline')].copy()
        data2 = results[(results['metric'] == metric2) & (results['model'] != 'baseline')].copy()
        
        # Create first axis
        ax1 = plt.gca()
        
        line1 = sns.lineplot(
            data=data1,
            x='acceleration',
            y='value',
            hue='model',
            palette=model_colors,
            marker='o',
            ax=ax1,
            linewidth=2.5
        )
        
        # Create second y-axis and add reference lines
        ax2 = ax1.twinx()            
        line2 = sns.lineplot(
            data=data2,
            x='acceleration',
            y='value',
            hue='model',
            palette=model_colors,
            marker='s',
            linestyle='--',
            ax=ax2,
            linewidth=2.5
        )
        
        # Set labels and title
        ax1.set_xlabel('Acceleration Factor')
        ax1.set_ylabel(metric_labels[metric1])
        ax2.set_ylabel(metric_labels[metric2])
        
        # Get lines and labels for both plots
        lines1, labels1 = ax1.get_legend_handles_labels()
        
        # Update custom lines to match new thickness
        custom_lines = [
            Line2D([0], [0], color='gray', linestyle='-', marker='o', 
                  label=metric_labels[metric1], linewidth=2.5, markersize=8),
            Line2D([0], [0], color='gray', linestyle='--', marker='s', 
                  label=metric_labels[metric2], linewidth=2.5, markersize=8)
        ]
        
        # Create combined legend with both models and line styles
        ax1.legend(lines1[:3] + custom_lines,
                  [model_labels[model] for model in labels1[:3]]+ [f'{metric_labels[metric1]}'] + [f'{metric_labels[metric2]}'],
                  bbox_to_anchor=(1.3, 1))
        ax2.get_legend().remove()
        
        # Set x-axis to treat values as categorical
        available_accelerations = sorted(data1['acceleration'].unique())
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
            ax2.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.2, zorder=0)
        
        # Set x-axis limits to remove extra whitespace
        ax1.set_xlim(-0.2, len(available_accelerations) - 0.8)
        
        # Calculate percentage drops for both metrics
        y1_max = data1['value'].max()
        y1_min = data1['value'].min()
        y2_max = data2['value'].max()
        y2_min = data2['value'].min()
        
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
        plt.gcf().set_size_inches(10, 6)  # Wider figure to fit legend
        
        # Adjust layout to prevent legend overlap
        plt.tight_layout()
        
        # Save plots in both formats
        for fmt in ['eps', 'pdf', 'png']:
            plt.savefig(
                f'{results_dir}/performance_plots/{name}_performance_{metric1}_{metric2}.{fmt}', 
                bbox_inches='tight', 
                dpi=300,
                format=fmt
            )
        plt.close()

def plot_ucsf_additional_bias(results, results_dir, name):
    # Create new directory for bias plots
    bias_dir = os.path.join(results_dir, 'bias_plots')
    os.makedirs(bias_dir, exist_ok=True)
    
    interpreter_configs = {
        'tgrade': {
            'metrics': ['delta-EODD', 'delta-EOP'],
            'x_labels': ['EODD', 'EOP']
        },
        'ttype': {
            'metrics': ['delta-EODD', 'delta-EOP'],
            'x_labels': ['EODD', 'EOP']
        },
        'dice': {
            'metrics': ['delta-SER', 'delta-delta-dice'],
            'x_labels': ['SER', 'Dice']
        }
    }
    
    # (Optional) Labels for display in the plot legends or axis labels.
    model_labels = {
        'unet': 'U-Net',
        'pix2pix': 'Pix2Pix',
        'sde': 'SDE'
    }
    
    # Update matplotlib rcParams to match desired style
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    })
    
    for interpreter, config in interpreter_configs.items():
        # Filter data for current interpreter
        data_task = results[results['interpreter'] == interpreter].copy()
        
        # Reshape data for plotting
        plot_data = []
        for metric in config['metrics']:
            metric_data = data_task[data_task['metric'] == metric].copy()
            metric_data['metric_type'] = config['x_labels'][config['metrics'].index(metric)]
            plot_data.append(metric_data)
        
        plot_data = pd.concat(plot_data)
        
        # Calculate global y limits before creating the plot
        y_min = plot_data['value'].min()
        y_max = plot_data['value'].max()
        
        # Round to next 0.05 interval in each direction
        y_min = np.floor(y_min / 0.05) * 0.05
        y_max = np.ceil(y_max / 0.05) * 0.05
        
        # Create the plot with fixed y limits
        g = sns.FacetGrid(
            plot_data,
            col="attribute",
            col_order=["gender", "age"],
            height=6,
            aspect=1.2,
            ylim=(y_min, y_max)
        )
        
        def plot_bars(data, **kwargs):
            ax = plt.gca()
            unique_metrics = data['metric_type'].unique()
            n_metrics = len(unique_metrics)
            n_models = len(model_colors)
            width = 0.15
            bar_gap = 0.005
            
            # Add horizontal reference lines using global limits
            for y in np.arange(y_min, y_max + 0.05, 0.05):
                ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.2, zorder=0)
            
            # Handle y-axis appearance based on facet position
            if ax.get_position().x0 > 0.1:  # Second facet
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_visible(False)
            else:  # First facet
                # Set y-ticks at every 0.05 interval
                ax.set_yticks(np.arange(y_min, y_max + 0.05, 0.05))
            
            for i, (model, color) in enumerate(model_colors.items()):
                model_data = data[data['model'] == model]
                x_positions = np.arange(n_metrics) * 0.6 + (i - (n_models-1)/2) * (width + bar_gap)
                
                values = [model_data[model_data['metric_type'] == metric]['value'].iloc[0] 
                         for metric in unique_metrics]
                
                plt.bar(x_positions, 
                       values,
                       width=width,
                       color=color,
                       label=model_labels[model],
                       zorder=2)
                
            plt.xticks(np.arange(n_metrics) * 0.6, unique_metrics)
            
        g.map_dataframe(plot_bars)
        
        # Customize the plot

        col_names = {"gender": "Gender", "age": "Age"}
        g.set_titles(template="{col_name}".format)
        for ax, title in zip(g.axes.flat, [col_names[col] for col in g.col_names]):
            ax.set_title(title)
        
        # Add legend
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the plots
        for fmt in ['eps', 'pdf', 'png']:
            plt.savefig(
                f'{results_dir}/bias_plots/{name}_bias_{interpreter}.{fmt}',
                bbox_inches='tight',
                dpi=300,
                format=fmt
            )
        plt.close()