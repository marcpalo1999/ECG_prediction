from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
import joblib
from pathlib import Path
from functools import wraps
import time
from datetime import datetime
import multiprocessing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import joblib
import json
from pathlib import Path
import os

n_workers = max(1, multiprocessing.cpu_count() - 1)
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {func.__name__} took {(end-start)/60:.2f} minutes")
        return result
    return wrapper

class RandomForestAnalyzer:
    """Class to handle Random Forest classification analysis."""

    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def preprocess(self, X, y):
        X_processed = X.copy()
        for column in X_processed.columns:
            X_processed[column] = pd.to_numeric(X_processed[column], errors='coerce')
        X_processed = X_processed.fillna(X_processed.median())
        
        # Use original y values for mapping
        unique_labels = np.unique(y)
        encoding_dict = {label: i for i, label in enumerate(unique_labels)}
        
        # Encode y
        y_processed = pd.Categorical(y).codes

        return X_processed, y_processed, encoding_dict
    
    @timing_decorator
    def analyze_oob(self, X, y):

        """Analyze model using Out-of-Bag error estimation."""
        # Preprocess data
        X_processed, y_processed, _ = self.preprocess(X,y)

        
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            oob_score=True, 
            random_state=self.random_state
        )
        rf.fit(X_processed, y_processed)
        
        results = {
            'train_score': rf.score(X_processed, y_processed),
            'oob_score': rf.oob_score_,
        }
        results['gap'] = results['train_score'] - results['oob_score']
        
        return results
    
    @timing_decorator
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation analysis with multiclass support."""
        # Preprocess data
        X_processed = X.copy()
        for column in X_processed.columns:
            X_processed[column] = pd.to_numeric(X_processed[column], errors='coerce')
        X_processed = X_processed.fillna(X_processed.median())
        y_processed = pd.factorize(y)[0]
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=self.n_estimators, 
                random_state=self.random_state
            ))
        ])
        
        scores = cross_validate(
            pipeline, 
            X_processed, 
            y_processed, 
            cv=cv,
            scoring={
                'accuracy': 'accuracy',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro',
                'f1_macro': 'f1_macro',
                'balanced_accuracy': 'balanced_accuracy'
            },
            return_train_score=True
        )
        
        return scores

    @timing_decorator
    def analyze_feature_importance(self, X, y, n_iterations=10):
        """Analyze feature importance stability."""
        # Preprocess data
        X_processed, y_processed, _ = self.preprocess(X,y)

        
        importances = []
        
        for _ in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(len(X_processed), len(X_processed), replace=True)
            X_boot = X_processed.iloc[indices]
            y_boot = y_processed[indices]
            
            # Fit model and get importance
            rf = self.get_fitted_model(X_boot, y_boot)

            importances.append(rf.feature_importances_)
        
        importances = np.array(importances)
        mean_imp = np.mean(importances, axis=0)
        std_imp = np.std(importances, axis=0)
        cv = std_imp / mean_imp
        
        # Calculate ranking stability
        rankings = np.argsort(-importances, axis=1)
        rank_stability = np.mean([
            kendalltau(rankings[0], rankings[i])[0] 
            for i in range(1, n_iterations)
        ])
        
        return {
            'mean_importance': mean_imp,
            'std_importance': std_imp,
            'cv': cv,
            'rank_stability': rank_stability,
            'features': X_processed.columns,
            'all_importances': importances
        }
    
    @timing_decorator
    def get_fitted_model(self, X, y):
        """Get a fitted RF model."""

        
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs = n_workers 
        )
        model.fit(X, y)
        
        return model

    def save_model(self, model, X_train, output_dir=None, timestamp=None):
        """
        Save Random Forest model with configuration and metadata
        
        Args:
            model: Trained RandomForest model
            X_train: Training feature matrix
            output_dir: Optional output directory
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir is None:
            output_dir = f"Results/RF_model/results_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'final_rf_model.joblib')
        joblib.dump(model, model_path)
        
        # Save configuration
        config = {
            'model_type': 'RandomForest',
            'n_estimators': model.n_estimators,
            'random_state': self.random_state,
            'feature_names': list(X_train.columns)
        }
        
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Model saved in: {output_dir}")
        return output_dir
            
    def load_model(self, model_path):
        """
        Load RF model and restore parameters.
        
        Args:
            model_path (str): Path to saved model file
        
        Returns:
            tuple: Loaded model and list of feature names
        """
        # Load joblib file
        model_data = joblib.load(model_path)
        
        # Restore model parameters
        model = model_data['model']
        features = model_data['features']
        
        # Update analyzer attributes
        self.n_estimators = model.n_estimators
        self.random_state = model.random_state
        
        return model, features


class ModelVisualizer:
    """Class to handle visualization of model results."""

    def __init__(self, figsize=(10, 6), style='default'):
        self.figsize = figsize
        plt.style.use(style)
        self.colors = {
            'train': 'lightblue',
            'test': 'lightgreen',
            'bar': 'lightblue',
            'stable': 'green',
            'medium': 'yellow',
            'unstable': 'red'
        }

    def plot_cv_results(self, scores, multiclass=True):
        """
        Enhanced cross-validation results plotting for both binary and multiclass scenarios
        
        Args:
            scores (dict): Cross-validation scores from sklearn
            multiclass (bool): Whether to use multiclass-specific visualization
        """
        # Basic metrics to plot
        if multiclass:
            # Multiclass-specific metrics
            metrics = [
                'test_accuracy', 
                'test_precision_macro', 
                'test_recall_macro', 
                'test_f1_macro', 
                'test_balanced_accuracy'
            ]
        else:
            # Default binary classification metrics
            metrics = [
                'test_accuracy', 
                'test_precision', 
                'test_recall', 
                'test_f1'
            ]
        
        plt.figure(figsize=(12, 6))
        plt.boxplot(
            [scores[metric] for metric in metrics], 
            labels=[m.replace('test_','') for m in metrics]
        )
        plt.title('Performance Metrics Across Cross-Validation Folds')
        plt.ylabel('Score')
        plt.xlabel('Metrics')
        plt.ylim(0, 1)  # Standard score range
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


        def plot_oob_analysis(self, results):
            plt.figure(figsize=self.figsize)
            bars = plt.bar(
                ['Training Score', 'OOB Score'], 
                [results['train_score'], results['oob_score']],
                color=[self.colors['train'], self.colors['test']]
            )
            
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2., 
                    height,
                    f'{height:.3f}',
                    ha='center', 
                    va='bottom'
                )
            
            plt.title('Random Forest Performance: Training vs OOB')
            plt.ylabel('Score')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.show()

    def plot_multiclass_confusion_matrix(self,y_true, y_pred, class_names, normalise = None):

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fmt = 'd'
        if normalise == 0:  # normalize columns
            cm = cm / cm.sum(axis=0, keepdims=True)
            fmt = '.2f'

        elif normalise == 1:  # normalize rows  
            cm = cm / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Function to plot multiclass ROC
    
    def plot_multiclass_roc(self, y_true, y_score, classes, save_path=None):
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        y_true_bin = label_binarize(y_true, classes=range(len(classes)))
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i, color in zip(range(len(classes)), colors):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=color, 
                    label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curves')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        return roc_auc

    def plot_feature_importance(self, importance_results):
        """Plot feature importance with stability indicators."""
        plt.figure(figsize=(self.figsize[0]+2, self.figsize[1]))
        
        # Sort by importance
        idx = np.argsort(importance_results['mean_importance'])
        features = importance_results['features'][idx]
        means = importance_results['mean_importance'][idx]
        stds = importance_results['std_importance'][idx]
        
        # Plot importance bars with error bars
        plt.barh(range(len(features)), means, 
                xerr=stds, 
                capsize=5,
                color=self.colors['bar'],
                alpha=0.8)
        
        # Add stability indicators
        for i, cv in enumerate(importance_results['cv'][idx]):
            color = (self.colors['stable'] if cv < 0.25 
                    else self.colors['medium'] if cv < 0.5 
                    else self.colors['unstable'])
            plt.plot(means[i], i, 'o', color=color, markeredgecolor='black')
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance (Rank Stability: {importance_results["rank_stability"]:.2f})')
        
        # Add stability legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor=self.colors['stable'],
                    label='High Stability (CV < 0.25)', 
                    markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor=self.colors['medium'],
                    label='Medium Stability (CV < 0.5)', 
                    markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor=self.colors['unstable'],
                    label='Low Stability (CV ≥ 0.5)', 
                    markeredgecolor='black')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        plt.show()



class BoundaryVisualizer:
    """Class to handle decision boundary visualization."""

    def __init__(self, figsize=(10, 10)):
        self.figsize = figsize
        self.colors = ['#FF9999', '#66B2FF']
        # self.cmap = ListedColormap(self.colors)
        self.cmap = plt.cm.get_cmap('viridis')
        
    def plot_boundaries_2d(self, X, y, model, method='pca', mesh_step=0.1):
        """Plot decision boundaries after dimensionality reduction."""
        # Scale data before reduction
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(X_scaled)
        else:
            reducer = TSNE(n_components=2, random_state=42, n_iter=250)
            X_reduced = reducer.fit_transform(X_scaled)
        
        # Create mesh grid (with fewer points)
        x_min, x_max = X_reduced[:, 0].min() - 0.5, X_reduced[:, 0].max() + 0.5
        y_min, y_max = X_reduced[:, 1].min() - 0.5, X_reduced[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 50),
            np.linspace(y_min, y_max, 50)
        )
        
        # Get predictions for mesh grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Transform grid points back to original space
        if method == 'pca':
            grid_points_original = scaler.inverse_transform(reducer.inverse_transform(grid_points))
        else:
            grid_points_original = grid_points
            
        # Predict in batches
        batch_size = 1000
        Z = np.zeros(grid_points_original.shape[0])
        for i in range(0, len(grid_points_original), batch_size):
            batch = grid_points_original[i:i + batch_size]
            Z[i:i + batch_size] = model.predict(batch)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=self.figsize)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=self.cmap)
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                            c=y, cmap=self.cmap, 
                            alpha=0.6)
        
        plt.title(f'Decision Boundaries ({method.upper()})')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        plt.legend(handles=scatter.legend_elements()[0], 
                    labels=['Ill', 'Healthy'],
                    title='Class')
        
        plt.tight_layout()
        plt.show()
        
        return X_reduced
    

    def plot_multiclass_boundaries(self, X, y, model):
        # Dimensionality reduction
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
        
        # Unique classes
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Create mesh grid
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        # Reconstruct original feature space
        grid_reduced = np.c_[xx.ravel(), yy.ravel()]
        grid_original = reducer.inverse_transform(grid_reduced)
        
        # Predict probabilities for each class
        Z = model.predict_proba(grid_original)
        
        # Plotting
        plt.figure(figsize=(12, 10))
        
        # Create heatmap of class probabilities
        for i in range(n_classes):
            plt.contourf(
                xx, yy, Z[:, i].reshape(xx.shape), 
                levels=np.linspace(0, 1, 10),
                cmap=plt.cm.get_cmap('viridis'),
                alpha=0.3
            )
        
        # Scatter plot of original points
        scatter = plt.scatter(
            X_reduced[:, 0], 
            X_reduced[:, 1], 
            c=y, 
            cmap=plt.cm.get_cmap('viridis'),
            edgecolor='black'
        )
        
        plt.colorbar(scatter, label='Class')
        plt.title('Multiclass Decision Boundaries')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()
