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

class RandomForestAnalyzer:
   """Class to handle Random Forest classification analysis."""
   
   def __init__(self, n_estimators=100, random_state=42):
       self.n_estimators = n_estimators
       self.random_state = random_state
       
   def analyze_oob(self, X, y):
       """Analyze model using Out-of-Bag error estimation."""
       # Preprocess data
       X_processed = X.copy()
       for column in X_processed.columns:
           X_processed[column] = pd.to_numeric(X_processed[column], errors='coerce')
       X_processed = X_processed.fillna(X_processed.median())
       y_processed = pd.factorize(y)[0]
       
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
   
   def cross_validate(self, X, y, cv=5):
       """Perform cross-validation analysis."""
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
               'precision': 'precision_macro',
               'recall': 'recall_macro',
               'f1': 'f1_macro'
           },
           return_train_score=True
       )
       
       return scores
   
   def analyze_feature_importance(self, X, y, n_iterations=10):
       """Analyze feature importance stability."""
       # Preprocess data
       X_processed = X.copy()
       for column in X_processed.columns:
           X_processed[column] = pd.to_numeric(X_processed[column], errors='coerce')
       X_processed = X_processed.fillna(X_processed.median())
       y_processed = pd.factorize(y)[0]
       
       importances = []
       
       for _ in range(n_iterations):
           # Bootstrap sample
           indices = np.random.choice(len(X_processed), len(X_processed), replace=True)
           X_boot = X_processed.iloc[indices]
           y_boot = y_processed[indices]
           
           # Fit model and get importance
           rf = RandomForestClassifier(
               n_estimators=self.n_estimators,
               random_state=self.random_state
           )
           rf.fit(X_boot, y_boot)
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
   
   def get_fitted_model(self, X, y):
       """Get a fitted RF model."""

       
       model = RandomForestClassifier(
           n_estimators=self.n_estimators,
           random_state=self.random_state
       )
       model.fit(X, y)
       
       return model

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
   
   def plot_cv_results(self, scores):
       metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
       
       plt.figure(figsize=self.figsize)
       data_to_plot = [scores[metric] for metric in metrics]
       plt.boxplot(data_to_plot, labels=[m.replace('test_','') for m in metrics])
       plt.title('Distribution of Performance Metrics Across CV Folds')
       plt.ylabel('Score')
       plt.grid(True)
       plt.show()
       
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



from matplotlib.colors import ListedColormap

class BoundaryVisualizer:
   """Class to handle decision boundary visualization."""
   
   def __init__(self, figsize=(10, 10)):
       self.figsize = figsize
       self.colors = ['#FF9999', '#66B2FF']
       self.cmap = ListedColormap(self.colors)
       
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