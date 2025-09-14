"""
Visualization Utilities for Ancient Greek NMT

This module provides comprehensive visualization tools for understanding
model behavior, training progress, and translation quality.

Key Visualizations:
1. Attention heatmaps - Show word alignments
2. Training curves - Monitor learning progress
3. Error analysis - Identify common mistakes
4. Performance metrics - Compare models
5. Linguistic analysis - Understand language patterns
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import torch
from pathlib import Path


class AttentionVisualizer:
    """
    Visualize attention patterns in translation models.

    Attention patterns show which source words the model focuses on
    when generating each target word. This helps understand:
    - Word alignments between languages
    - Model's understanding of grammar
    - Potential translation errors
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize attention visualizer.

        Parameters:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        source_tokens: List[str],
        target_tokens: List[str],
        title: str = "Attention Weights",
        save_path: Optional[str] = None
    ):
        """
        Create an attention heatmap showing word alignments.

        Parameters:
            attention_weights: Matrix of attention scores (source x target)
            source_tokens: List of source language tokens
            target_tokens: List of target language tokens
            title: Plot title
            save_path: Optional path to save the figure

        The heatmap shows:
        - Darker colors = stronger attention
        - Each cell shows attention weight between word pairs
        - Patterns reveal translation alignments
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap
        sns.heatmap(
            attention_weights,
            xticklabels=target_tokens,
            yticklabels=source_tokens,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            fmt='.2f',
            annot=True,
            square=True,
            linewidths=0.5,
            ax=ax
        )

        # Customize appearance
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Target (English)', fontsize=12)
        ax.set_ylabel('Source (Greek)', fontsize=12)

        # Rotate labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {save_path}")

        plt.show()

    def plot_attention_evolution(
        self,
        attention_history: List[np.ndarray],
        layer_names: Optional[List[str]] = None
    ):
        """
        Show how attention patterns evolve across model layers.

        Different layers capture different types of relationships:
        - Early layers: Surface features (word similarity)
        - Middle layers: Syntactic patterns (grammar)
        - Late layers: Semantic relationships (meaning)
        """
        n_layers = len(attention_history)

        if layer_names is None:
            layer_names = [f"Layer {i+1}" for i in range(n_layers)]

        # Create subplot grid
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten() if n_layers > 1 else [axes]

        for i, (attn, name) in enumerate(zip(attention_history, layer_names)):
            if i < len(axes):
                im = axes[i].imshow(attn, cmap='Blues', aspect='auto')
                axes[i].set_title(name, fontsize=11)
                axes[i].set_xlabel('Target Position')
                axes[i].set_ylabel('Source Position')
                plt.colorbar(im, ax=axes[i], fraction=0.046)

        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Attention Pattern Evolution Across Layers',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


class TrainingVisualizer:
    """
    Visualize training progress and metrics.

    Helps identify:
    - Overfitting (diverging train/val curves)
    - Learning rate issues (plateaus or instability)
    - Convergence patterns
    """

    def __init__(self):
        """Initialize training visualizer."""
        self.style_config = {
            'train': {'color': 'blue', 'linestyle': '-', 'label': 'Training'},
            'val': {'color': 'orange', 'linestyle': '--', 'label': 'Validation'},
            'test': {'color': 'green', 'linestyle': ':', 'label': 'Test'}
        }

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        metrics: List[str] = ['loss', 'bleu'],
        save_path: Optional[str] = None
    ):
        """
        Plot training curves for multiple metrics.

        Parameters:
            history: Dictionary with metric histories
            metrics: List of metrics to plot
            save_path: Optional path to save figure

        Interpretation:
        - Loss should decrease over time
        - BLEU should increase over time
        - Gap between train/val indicates generalization
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            # Plot each split if available
            for split in ['train', 'val', 'test']:
                key = f"{split}_{metric}"
                if key in history:
                    epochs = range(1, len(history[key]) + 1)
                    ax.plot(epochs, history[key],
                           **self.style_config[split],
                           linewidth=2)

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.upper()} During Training', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_learning_rate_schedule(
        self,
        learning_rates: List[float],
        steps: Optional[List[int]] = None
    ):
        """
        Visualize learning rate schedule.

        Common patterns:
        - Warmup: Gradual increase at start
        - Decay: Decrease over time
        - Plateaus: Constant periods
        - Restarts: Cyclic patterns
        """
        if steps is None:
            steps = list(range(len(learning_rates)))

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(steps, learning_rates, 'g-', linewidth=2)
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add annotations for key points
        if len(learning_rates) > 0:
            max_lr = max(learning_rates)
            max_idx = learning_rates.index(max_lr)
            ax.annotate(f'Peak: {max_lr:.2e}',
                       xy=(steps[max_idx], max_lr),
                       xytext=(10, 10),
                       textcoords='offset points',
                       ha='left',
                       bbox=dict(boxstyle='round,pad=0.5',
                                fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->',
                                      connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        plt.show()


class ErrorAnalyzer:
    """
    Analyze and visualize translation errors.

    Helps identify:
    - Common error patterns
    - Problematic source constructions
    - Model weaknesses
    """

    def __init__(self):
        """Initialize error analyzer."""
        self.error_categories = [
            'word_order',
            'vocabulary',
            'grammar',
            'omission',
            'addition',
            'untranslated'
        ]

    def plot_error_distribution(
        self,
        error_counts: Dict[str, int],
        total_samples: int
    ):
        """
        Visualize distribution of error types.

        Parameters:
            error_counts: Dictionary of error type counts
            total_samples: Total number of samples analyzed
        """
        # Calculate percentages
        error_data = []
        for category in self.error_categories:
            count = error_counts.get(category, 0)
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            error_data.append({
                'Category': category.replace('_', ' ').title(),
                'Count': count,
                'Percentage': percentage
            })

        df = pd.DataFrame(error_data)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
        bars = ax1.bar(df['Category'], df['Count'], color=colors)
        ax1.set_xlabel('Error Type', fontsize=12)
        ax1.set_ylabel('Number of Errors', fontsize=12)
        ax1.set_title('Error Count by Type', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

        # Pie chart
        explode = [0.1 if p > 20 else 0 for p in df['Percentage']]
        ax2.pie(df['Count'], labels=df['Category'], autopct='%1.1f%%',
               colors=colors, explode=explode, startangle=90)
        ax2.set_title('Error Distribution', fontsize=13, fontweight='bold')

        plt.suptitle(f'Translation Error Analysis (n={total_samples})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_error_heatmap_by_length(
        self,
        errors_by_length: Dict[int, Dict[str, int]]
    ):
        """
        Show how error rates vary with sentence length.

        Longer sentences typically have:
        - More word order errors (flexible Greek syntax)
        - More omissions (information loss)
        - Lower overall accuracy
        """
        # Prepare data matrix
        lengths = sorted(errors_by_length.keys())
        categories = self.error_categories

        matrix = np.zeros((len(categories), len(lengths)))
        for i, cat in enumerate(categories):
            for j, length in enumerate(lengths):
                matrix[i, j] = errors_by_length[length].get(cat, 0)

        # Normalize by column (percentage per length)
        matrix_norm = matrix / (matrix.sum(axis=0) + 1e-10) * 100

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(matrix_norm,
                   xticklabels=[f"{l} words" for l in lengths],
                   yticklabels=[c.replace('_', ' ').title() for c in categories],
                   cmap='RdYlBu_r',
                   fmt='.1f',
                   annot=True,
                   cbar_kws={'label': 'Error Rate (%)'},
                   ax=ax)

        ax.set_title('Error Rates by Sentence Length',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Sentence Length', fontsize=12)
        ax.set_ylabel('Error Type', fontsize=12)

        plt.tight_layout()
        plt.show()


class PerformanceComparator:
    """
    Compare performance across models, datasets, or configurations.
    """

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['bleu', 'chrf', 'meteor']
    ):
        """
        Compare multiple models across different metrics.

        Parameters:
            results: Dictionary of model_name -> metric_scores
            metrics: List of metrics to compare
        """
        models = list(results.keys())
        n_models = len(models)
        n_metrics = len(metrics)

        # Prepare data
        data = np.zeros((n_models, n_metrics))
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics):
                data[i, j] = results[model].get(metric, 0)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(n_models)
        width = 0.8 / n_metrics

        colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

        for j, metric in enumerate(metrics):
            offset = (j - n_metrics/2) * width + width/2
            bars = ax.bar(x + offset, data[:, j], width,
                         label=metric.upper(), color=colors[j])

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    def plot_speed_quality_tradeoff(
        self,
        models_data: List[Dict[str, Any]]
    ):
        """
        Visualize the trade-off between inference speed and quality.

        Parameters:
            models_data: List of dicts with 'name', 'speed', 'quality', 'size'
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract data
        names = [m['name'] for m in models_data]
        speeds = [m['speed'] for m in models_data]
        qualities = [m['quality'] for m in models_data]
        sizes = [m.get('size', 100) for m in models_data]

        # Create scatter plot
        scatter = ax.scatter(speeds, qualities, s=sizes, alpha=0.6,
                           c=range(len(names)), cmap='viridis',
                           edgecolors='black', linewidth=2)

        # Add labels
        for i, name in enumerate(names):
            ax.annotate(name, (speeds[i], qualities[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10)

        ax.set_xlabel('Inference Time (ms)', fontsize=12)
        ax.set_ylabel('Translation Quality (BLEU)', fontsize=12)
        ax.set_title('Speed vs Quality Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add size legend
        handles, labels = ax.get_legend_handles_labels()
        legend1 = ax.legend(handles, labels, loc='upper right', title='Models')

        # Add optimal frontier line (example)
        ax.plot([min(speeds), max(speeds)],
               [max(qualities), min(qualities)],
               'r--', alpha=0.5, label='Pareto Frontier')

        plt.tight_layout()
        plt.show()