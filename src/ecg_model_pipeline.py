from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import multiprocessing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
from matplotlib.lines import Line2D
from functools import wraps
import time
import shap


n_workers = max(1, multiprocessing.cpu_count() - 1)


def timing_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {func.__name__} took {(end-start)/60:.2f} minutes")
        return result
    return wrapper


class ECGModelTrainer:
    """Train and evaluate Random Forest models for ECG classification"""
    
    def __init__(self, n_estimators=100, random_state=42, feature_scaling=True, class_weight='balanced'):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_scaling = feature_scaling
        self.class_weight = class_weight
        self.model = None
        self.preprocessing_params = {}
        self.label_mapping = {}

        
    def create_pipeline(self):
        """Create sklearn pipeline with preprocessing and model"""
        steps = []
        
        if self.feature_scaling:
            steps.append(('scaler', StandardScaler()))
            
        steps.append(('classifier', RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=n_workers
        )))
        
        return Pipeline(steps)
    
    def preprocess(self, X, y, cardiac_pathology_groups=None):
        """
        Preprocess data for model training
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target labels (cardiac conditions)
        cardiac_pathology_groups : dict, optional
            Dictionary of cardiac pathology groups with severity scores
            
        Returns:
        --------
        X_processed : DataFrame
            Processed feature matrix
        y_processed : ndarray
            Processed target labels (encoded)
        label_mapping : dict
            Mapping from labels to severity codes
        """
        # Process features
        X_processed = X.copy()
        
        # Convert to numeric and handle missing values
        for column in X_processed.columns:
            X_processed[column] = pd.to_numeric(X_processed[column], errors='coerce')
        
        # Fill missing values with median
        medians = X_processed.median()
        X_processed = X_processed.fillna(medians)
        
        # Process labels
        if cardiac_pathology_groups is not None:
            # Use provided severity mapping
            severity_mapping = {
                category: details['severity'] 
                for category, details in cardiac_pathology_groups.items()
            }
        else:
            raise ValueError("Cardiac pathology groups must be provided for severity mapping")
        
        # Map labels to severity scores
        y_processed = np.array([severity_mapping.get(label, -1) for label in y])
        
        # Store parameters for future use
        self.preprocessing_params = {
            'feature_names': list(X_processed.columns),
            'feature_medians': medians.to_dict()
        }
        
        # Store label mapping
        self.label_mapping = severity_mapping
        
        return X_processed, y_processed, severity_mapping
    
    @timing_decorator
    def fit(self, X_preprocessed, y_preprocessed, cardiac_pathology_groups=None):
        """
        Fit model to training data with preprocessing
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target labels
        cardiac_pathology_groups : dict, optional
            Dictionary of cardiac pathology groups with severity scores
        """
        # Preprocess data
        
        
        # Fit model
        self.model = self.create_pipeline()
        self.model.fit(X_preprocessed, y_preprocessed)

        return self
    
    def create_shap_explainer(self, X_background=None, max_background_samples=100):
        """Create and return a SHAP TreeExplainer for the trained model
        
        Parameters:
        -----------
        X_background : DataFrame, optional
            Background data for the explainer (should be preprocessed)
        max_background_samples : int
            Maximum number of samples to use as background
            
        Returns:
        --------
        explainer : shap.TreeExplainer
            Trained SHAP explainer
        """
        import shap
        
        if self.model is None:
            raise ValueError("Model must be trained before creating explainer")
        
        # Get the underlying classifier from the pipeline
        classifier = self.model.named_steps['classifier']
        
        # Use provided background data or create empty explainer
        if X_background is not None:
            # Limit sample size if needed
            if len(X_background) > max_background_samples:
                X_background = X_background.iloc[:max_background_samples]
            
            # Create explainer with background data
            self.explainer = shap.TreeExplainer(classifier, X_background)
        else:
            # Create explainer without background data
            self.explainer = shap.TreeExplainer(classifier)
        
        return self.explainer
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        X_processed = self._preprocess_features(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        X_processed = self._preprocess_features(X)
        return self.model.predict_proba(X_processed)
    
    def _preprocess_features(self, X):
        """Preprocess features for model input"""
        X_processed = X.copy()
        
        # Ensure all expected features are present
        for feature in self.preprocessing_params.get('feature_names', []):
            if feature not in X_processed:
                X_processed[feature] = np.nan
        
        # Convert to numeric and fill missing values
        for column in X_processed.columns:
            X_processed[column] = pd.to_numeric(X_processed[column], errors='coerce')
            
        # Fill missing values with stored medians
        if self.preprocessing_params.get('feature_medians'):
            for col, median in self.preprocessing_params['feature_medians'].items():
                if col in X_processed:
                    X_processed[col] = X_processed[col].fillna(median)
        
        return X_processed
    
    @timing_decorator
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation analysis"""
        # Preprocess data if not already done
        if not self.preprocessing_params:
            X_processed, y_processed, _ = self.preprocess(X, y)
        else:
            X_processed = self._preprocess_features(X)
            y_processed = np.array([self.label_mapping.get(label, -1) for label in y])
        
        pipeline = self.create_pipeline()
        
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
                'balanced_accuracy': 'balanced_accuracy',
                'roc_auc_ovr': 'roc_auc_ovr'
            },
            return_train_score=True
        )
        
        return scores
    
    @timing_decorator
    def analyze_feature_importance(self, X, y, n_iterations=10):
        """
        Analyze feature importance stability
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target labels
        n_iterations : int
            Number of bootstrap iterations
            
        Returns:
        --------
        dict : Feature importance analysis results
        """
        # Preprocess data if not already done
        if not self.preprocessing_params:
            X_processed, y_processed, _ = self.preprocess(X, y)
        else:
            X_processed = self._preprocess_features(X)
            y_processed = np.array([self.label_mapping.get(label, -1) for label in y])
        
        importances = []
        
        for _ in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(len(X_processed), len(X_processed), replace=True)
            X_boot = X_processed.iloc[indices]
            y_boot = y_processed[indices]
            
            # Fit model and get importance
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=n_workers
            )
            rf.fit(X_boot, y_boot)
            importances.append(rf.feature_importances_)
        
        importances = np.array(importances)
        mean_imp = np.mean(importances, axis=0)
        std_imp = np.std(importances, axis=0)
        cv = std_imp / (mean_imp + 1e-10)  # Avoid division by zero
        
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
    
    def save(self, path):
        """Save model, preprocessing parameters, and SHAP explainer"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'preprocessing_params': self.preprocessing_params,
            'label_mapping': self.label_mapping,
            'explainer': getattr(self, 'explainer', None),  # Save explainer if exists
            'config': {
                'n_estimators': self.n_estimators,
                'random_state': self.random_state,
                'feature_scaling': self.feature_scaling,
                'class_weight': self.class_weight
            }
        }
        
        joblib.dump(model_data, path)
        
    @classmethod
    def load(cls, path):
        """Load model from file"""
        model_data = joblib.load(path)
        
        # Create new instance
        instance = cls(
            n_estimators=model_data['config']['n_estimators'],
            random_state=model_data['config']['random_state'],
            feature_scaling=model_data['config']['feature_scaling'],
            class_weight=model_data['config']['class_weight']
        )
        
        # Restore model and parameters
        instance.model = model_data['model']
        instance.preprocessing_params = model_data['preprocessing_params']
        instance.label_mapping = model_data['label_mapping']
        
        # Restore explainer if available
        if 'explainer' in model_data and model_data['explainer'] is not None:
            instance.explainer = model_data['explainer']
        
        return instance


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
        Plot cross-validation results for model evaluation
        
        Parameters:
        -----------
        scores : dict
            Cross-validation scores from sklearn
        multiclass : bool
            Whether to use multiclass-specific visualization
        """
        # Select metrics to plot
        if multiclass:
            # Multiclass-specific metrics
            metrics = [
                'test_accuracy', 
                'test_precision_macro', 
                'test_recall_macro', 
                'test_f1_macro', 
                'test_balanced_accuracy',
                'test_roc_auc_ovr'
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


    def plot_multiclass_confusion_matrix(self, y_true, y_pred, class_names, normalise=None):
        """
        Plot multiclass confusion matrix with sensitivity and specificity metrics
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list
            Names of classes
        normalise : int or None
            Whether to normalize confusion matrix (0: by column, 1: by row)
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fmt = 'd'
        
        if normalise == 0:  # normalize columns
            col_sums = cm.sum(axis=0, keepdims=True)
            # Handle zero division
            col_sums = np.where(col_sums == 0, 1, col_sums)
            cm = cm / col_sums
            fmt = '.3f'  # More decimal places
        elif normalise == 1:  # normalize rows  
            row_sums = cm.sum(axis=1, keepdims=True)
            # Handle zero division
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm = cm / row_sums
            fmt = '.3f'  # More decimal places  
        
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
        
        # Calculate sensitivity and specificity for each class
        n_classes = len(class_names)
        sensitivity = np.zeros(n_classes)
        specificity = np.zeros(n_classes)
        
        for i in range(n_classes):
            # Sensitivity = TP / (TP + FN) = Recall
            sensitivity[i] = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
            
            # Specificity = TN / (TN + FP)
            # TN is the sum of all elements except those in row i and column i
            # FP is the sum of column i minus the true positive
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate F1 score (harmonic mean of sensitivity and precision)
        # Precision = TP / (TP + FP)
        precision = np.zeros(n_classes)
        for i in range(n_classes):
            precision[i] = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
        
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        f1 = np.nan_to_num(f1)  # Replace NaN with 0
        
        # Calculate support (number of occurrences of each class in y_true)
        support = np.sum(cm, axis=1)
        
        # Create custom report
        report_data = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1-score': f1,
            'support': support
        }
        
        # Calculate macro average (unweighted mean of each class)
        for metric in ['sensitivity', 'specificity', 'f1-score']:
            report_data[f'macro avg {metric}'] = np.mean(report_data[metric])
        
        # Calculate weighted average (weighted by support)
        for metric in ['sensitivity', 'specificity', 'f1-score']:
            report_data[f'weighted avg {metric}'] = np.sum(report_data[metric] * support) / np.sum(support)
        
        # Create pretty table
        print("\nClassification Report with Sensitivity and Specificity:")
        print("="*60)
        print(f"{'Class':<15} {'Sensitivity':<12} {'Specificity':<12} {'F1-Score':<12} {'Support':<10}")
        print("-"*60)
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<15} {sensitivity[i]:<12.4f} {specificity[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10.0f}")
        
        print("-"*60)
        print(f"{'macro avg':<15} {report_data['macro avg sensitivity']:<12.4f} {report_data['macro avg specificity']:<12.4f} {report_data['macro avg f1-score']:<12.4f} {np.sum(support):<10.0f}")
        print(f"{'weighted avg':<15} {report_data['weighted avg sensitivity']:<12.4f} {report_data['weighted avg specificity']:<12.4f} {report_data['weighted avg f1-score']:<12.4f} {np.sum(support):<10.0f}")
        print("="*60)
    
    def plot_one_vs_all_multiclass_roc(self, y_true, y_score, classes, save_path=None):
        """
        Plot one-vs-all ROC curves for multiclass classification
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_score : array-like
            Predicted probabilities
        classes : list
            Names of classes
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        dict : ROC AUC values for each class
        """
        # Binarize the labels
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
    
    def plot_one_vs_normal_roc(self, y_true, y_pred_proba, classes, normal_idx=0):
        """
        Plot ROC curves for each class vs. the normal class
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        classes : list
            Names of classes
        normal_idx : int
            Index of the normal class
        """
        plt.figure(figsize=(10, 8))
        
        # Get indices of normal class
        normal_indices = (y_true == normal_idx)
        
        for i, class_name in enumerate(classes):
            if i == normal_idx:
                continue  # Skip normal class
                
            # Get indices of current class
            class_indices = (y_true == i)
            
            # Combine normal and current class
            binary_indices = normal_indices | class_indices
            binary_true = y_true[binary_indices]
            binary_proba = y_pred_proba[binary_indices, i]
            
            # Create binary labels (1 for current class, 0 for normal)
            binary_true = (binary_true == i).astype(int)
            
            # Calculate ROC
            fpr, tpr, _ = roc_curve(binary_true, binary_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{classes[i]} vs Normal (AUC={roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC: Each Class vs Normal')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_results):
        """
        Plot feature importance with stability indicators
        
        Parameters:
        -----------
        importance_results : dict
            Results from analyze_feature_importance method
        """
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
    
    @staticmethod
    def plot_decision_space(X, y, model, classes, binary=True):
        """Plot decision boundaries in PCA space with feature vectors and contribution heatmap"""
        # Color mapping (existing code)
        if binary:
            severity_colors = {
                0: '#4daf4a',  # Normal - Green
                1: '#e41a1c',  # All pathologies - Red
                2: '#e41a1c',  
                3: '#e41a1c',  
                4: '#e41a1c',  
                5: '#e41a1c',  
                6: '#e41a1c',  
                7: '#e41a1c'   
            }
        else:
            severity_colors = {
                0: '#4daf4a',  # Normal - Green
                1: '#80b1d3',  # Benign Variants - Light Blue
                2: '#bebada',  # Others - Light Purple
                3: '#fdb462',  # Atrial Abnormalities - Orange
                4: '#fb8072',  # Chamber Abnormalities - Light Red
                5: '#bc80bd',  # Conduction System Disease - Purple
                6: '#ff7f00',  # Primary Electrical Disorders - Dark Orange
                7: '#e41a1c'   # Ischemic Disorders - Red
            }
        
        # Get feature names
        feature_names = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]
        
        # Reduce dimensions with PCA
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
        
        # Create visualization with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})
        
        # PLOT 1: Decision boundary with feature vectors
        # Create mesh grid for decision boundary
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        # Transform grid points back to original space
        grid_reduced = np.c_[xx.ravel(), yy.ravel()]
        grid_original = reducer.inverse_transform(grid_reduced)
        
        # Get predictions
        Z_proba = model.predict_proba(grid_original)
        Z = np.argmax(Z_proba, axis=1)
        
        # Plot decision boundaries
        axes[0].tricontourf(grid_reduced[:, 0], grid_reduced[:, 1], Z, 
                    levels=len(np.unique(Z)), 
                    colors=[severity_colors[cls] for cls in np.unique(Z)], 
                    alpha=0.2)
        
        # Plot data points
        for cls in sorted(np.unique(y)):
            mask = (y == cls)
            axes[0].scatter(
                X_reduced[mask, 0], 
                X_reduced[mask, 1], 
                color=severity_colors[cls],
                edgecolor='black', 
                s=10,
                label=f"{classes[cls]} (Severity: {cls})"
            )
        
        # Add feature vectors as arrows
        # Calculate scaling factor for visibility
        scale_factor = min(x_max - x_min, y_max - y_min) * 0.15
        
        for i, (name, x, y) in enumerate(zip(feature_names, 
                                            reducer.components_[0] * scale_factor, 
                                            reducer.components_[1] * scale_factor)):
            axes[0].arrow(0, 0, x, y, color='red', width=0.3, head_width=1, head_length=2)
            axes[0].text(x*1.1, y*1.1, name, color='red', fontsize=12)
        
        axes[0].legend(title='Cardiac Conditions', loc='best', markerscale=2)
        axes[0].set_title('Cardiac Condition Classification in PCA Space')
        axes[0].set_xlabel('PCA Component 1')
        axes[0].set_ylabel('PCA Component 2')
        
        # PLOT 2: Feature contributions heatmap
        loadings = reducer.components_
        
        # Create heatmap of feature contributions
        sns.heatmap(
            loadings.T,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            xticklabels=['PC1', 'PC2'],
            yticklabels=feature_names,
            ax=axes[1]
        )
        axes[1].set_title('Feature Contributions to Principal Components')
        axes[1].set_xlabel('Principal Components')
        axes[1].set_ylabel('Features')
        
        plt.tight_layout()
        plt.show()
        
        return X_reduced, loadings

# Combined pipeline class that incorporates both data processing and modeling
class ECGClassificationPipeline:
    """
    End-to-end pipeline for ECG classification, combining data processing and modeling
    """
    
    def __init__(self):
        # Import data pipeline components from ecg_data_pipeline
        try:
            from ecg_data_pipeline import ECGFeatureExtractor, CardiacConditionClassifier, ECGPipeline
            self.ECGFeatureExtractor = ECGFeatureExtractor
            self.CardiacConditionClassifier = CardiacConditionClassifier
            self.ECGPipeline = ECGPipeline
        except ImportError:
            print("WARNING: Could not import data pipeline components. Some functionality will be limited.")
        
        # Initialize components
        self.data_pipeline = None
        self.model_trainer = None
        self.visualizer = ModelVisualizer()
        
    def setup(self, feature_extractor=None, condition_classifier=None, model_trainer=None):
        """Setup pipeline components"""
        # Setup data pipeline
        self.data_pipeline = self.ECGPipeline().setup(
            feature_extractor=feature_extractor or self.ECGFeatureExtractor(),
            condition_classifier=condition_classifier or self.CardiacConditionClassifier()
        )
        
        # Setup model trainer
        self.model_trainer = model_trainer or ECGModelTrainer()
        
        return self
    
    @timing_decorator
    def process_and_train(self, ecg_data, patient_data, features=None):
        """
        Process data and train model in one step
        
        Parameters:
        -----------
        ecg_data : dict
            Dictionary of patient ECG data
        patient_data : DataFrame
            DataFrame with patient metadata
        features : list, optional
            List of features to use for training
            
        Returns:
        --------
        DataFrame : Processed data
        """
        # Process data
        processed_data = self.data_pipeline.process_data(ecg_data, patient_data)
        
        # Select features
        if features is None:
            features = ['median_hr', 'mean_hr', 'std_hr', 'min_hr', 'max_hr', 'age']
        
        # Train model
        X = processed_data[features]
        y = processed_data['cardiac_condition']
        
        self.model_trainer.fit(X, y)

        
        return processed_data
    
    def evaluate(self, processed_data, features=None):
        """
        Evaluate model on processed data
        
        Parameters:
        -----------
        processed_data : DataFrame
            Processed data with features and labels
        features : list, optional
            List of features to use for evaluation
        """
        if features is None:
            features = ['median_hr', 'mean_hr', 'std_hr', 'min_hr', 'max_hr', 'age']
            
        X = processed_data[features]
        y = processed_data['cardiac_condition']
        
        # Get predictions
        y_pred = self.model_trainer.predict(X)
        y_pred_proba = self.model_trainer.predict_proba(X)
        
        # Get class names
        inv_mapping = {v: k for k, v in self.model_trainer.label_mapping.items()}
        ordered_classes = [inv_mapping[i] for i in range(len(inv_mapping))]
        
        # Plot confusion matrix
        self.visualizer.plot_multiclass_confusion_matrix(y_pred, y, ordered_classes)
        
        # Plot ROC curves
        self.visualizer.plot_one_vs_all_multiclass_roc(y, y_pred_proba, ordered_classes)
        
        # Plot one-vs-normal ROC curves
        self.visualizer.plot_one_vs_normal_roc(y, y_pred_proba, ordered_classes)
        
        # Plot decision boundaries
        BoundaryVisualizer.plot_decision_space(X, y, self.model_trainer.model, ordered_classes, binary=True)
        
    def save(self, base_path):
        """Save pipeline components to disk"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save data pipeline
        self.data_pipeline.save(os.path.join(base_path, 'data_pipeline'))
        
        # Save model
        self.model_trainer.save(os.path.join(base_path, 'model.joblib'))
        
        # Save configuration
        config = {
            'pipeline_version': '1.0',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(base_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)