import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import os

class MetricsVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize(self, metrics_history):
        self._plot_line_charts(metrics_history)
        self._plot_heatmap(metrics_history)
        self._plot_boxplots(metrics_history)
        self._plot_scatter(metrics_history)
        self._plot_bar_charts(metrics_history)
        self._plot_network(metrics_history)

    def _plot_line_charts(self, metrics_history):
        for metric_name, values in metrics_history.items():
            if isinstance(values[0], (int, float)):
                plt.figure(figsize=(10, 6))
                plt.plot(values, marker='o')
                plt.title(f'{metric_name} over Generations')
                plt.xlabel('Generation')
                plt.ylabel(metric_name)
                plt.savefig(os.path.join(self.output_dir, f'{metric_name}_line_plot.png'))
                plt.close()

    def _plot_heatmap(self, metrics_history):
        # Assuming 'behavioral_space' is a 2D histogram of solution distribution
        if 'behavioral_space' in metrics_history:
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics_history['behavioral_space'][-1], cmap='viridis')
            plt.title('Distribution of Solutions in Behavioral Space')
            plt.savefig(os.path.join(self.output_dir, 'behavioral_space_heatmap.png'))
            plt.close()

    def _plot_boxplots(self, metrics_history):
        # Plotting boxplots for metrics that are lists of values
        for metric_name, values in metrics_history.items():
            if isinstance(values[0], list):
                plt.figure(figsize=(12, 6))
                plt.boxplot(values)
                plt.title(f'{metric_name} Distribution over Generations')
                plt.xlabel('Generation')
                plt.ylabel(metric_name)
                plt.savefig(os.path.join(self.output_dir, f'{metric_name}_boxplot.png'))
                plt.close()

    def _plot_scatter(self, metrics_history):
        # Scatter plot of behavioral diversity vs genotypic diversity
        if 'behavioral_diversity' in metrics_history and 'genotypic_diversity' in metrics_history:
            plt.figure(figsize=(8, 8))
            plt.scatter(metrics_history['behavioral_diversity'], metrics_history['genotypic_diversity'])
            plt.xlabel('Behavioral Diversity')
            plt.ylabel('Genotypic Diversity')
            plt.title('Behavioral vs Genotypic Diversity')
            plt.savefig(os.path.join(self.output_dir, 'diversity_scatter.png'))
            plt.close()

    def _plot_bar_charts(self, metrics_history):
        # Bar chart of cluster sizes for the last generation
        if 'cluster_sizes' in metrics_history:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(metrics_history['cluster_sizes'][-1])), metrics_history['cluster_sizes'][-1])
            plt.title('Cluster Sizes in Last Generation')
            plt.xlabel('Cluster')
            plt.ylabel('Size')
            plt.savefig(os.path.join(self.output_dir, 'cluster_sizes_bar.png'))
            plt.close()

    def _plot_network(self, metrics_history):
        # Network graph of solution connectivity in behavioral space
        if 'solution_connectivity' in metrics_history:
            G = nx.Graph(metrics_history['solution_connectivity'][-1])
            plt.figure(figsize=(12, 12))
            nx.draw(G, with_labels=False, node_size=30)
            plt.title('Solution Connectivity in Behavioral Space')
            plt.savefig(os.path.join(self.output_dir, 'solution_network.png'))
            plt.close()