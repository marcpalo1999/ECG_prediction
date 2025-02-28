# ECG Disease Classification using Random Forest

## Overview
This project classifies cardiac conditions from 12-lead ECG data using machine learning, with a focus on clinical severity prioritization.

## Data Source
[PhysioNet ECG Arrhythmia Database](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
- 12-lead ECG recordings with SNOMED-CT coded diagnoses

## Methodology

### Signal Processing
- Bandpass filtering (5-140 Hz) and notch filtering (50 Hz)
- R-peak detection with physiological validation
- Signal detrending across all 12 leads

### Cardiac Condition Classification
Hierarchical severity-based grouping:
1. Normal (Severity 0)
2. Benign Variants (Severity 2)
3. Atrial Abnormalities (Severity 3)
4. Chamber Abnormalities (Severity 4)
5. Conduction System Disease (Severity 5)
6. Primary Electrical Disorders (Severity 6)
7. Ischemic Disorders (Severity 7)

### Feature Engineering
- Heart rate metrics (median, mean, std, min, max), extracted through peak to peak (R-R) detection algorithm.
- Patient demographics (age)
- *Planned: ECG morphology features (PR intervals, QRS width, ST elevation)*

### Data Visualization
- Sample size distribution
- Missing data visualization
- Normalised KDE of Probability density functions of data by pathology group
- Boxplots
- Feature correlation analysis (Spearman)
- Key observation: Data distribution strongly depends on detection algorithm

### Machine Learning Approach
- Random Forest classifier with class balancing
- Feature importance stability analysis
- ROC curve analysis with one-vs-all and one-vs-normal approaches
- PCA-based decision boundary visualization

## Key Components
- `data_processing.py`: Signal preprocessing
- `ecg_data_pipeline.py`: Feature extraction
- `ecg_model_pipeline.py`: Model training and evaluation
- `auxiliary.py`: Cardiac condition mapping

## Model Performance
**On Development**
- Overall:
  - sensitivity: ~41%
  - specificity: ~91%
  - ROC AUC: ~73%
- Binary (Normal va pathological):
  - Sensitivity: ~92%
  - Specificity: ~92%
  - ROC AUC: 98%

**On Validation**
- Overall:
  - sensitivity: ~42%
  - specificity: ~92%
- Binary (Normal va pathological):
  - Sensitivity: ~92%
  - Specificity: ~92%
  - ROC AUC: 98%
- Age dependency: Best performance in 40-65 age group
- Strengths:
  - Excellent normal ECG detection
  - Strong normal vs. abnormal discrimination

## Limitations
- ~10% of high-severity conditions misclassified as normal/benign
- Poor detection of Conduction System Disease and Ischemic Disorders
- Current features insufficient for separating morphologically distinct conditions
- Significant class imbalance affects model performance

## Future Directions
1. Implement ECG morphological features
2. Improve R-R peak detection algorithm
3. Develop two-stage classification approach (normal/abnormal first)
4. Apply advanced class balancing techniques
5. Implement condition-specific classification thresholds
6. Explore age-adaptive modeling for elderly patients
7. Integrate deep learning for feature extraction

## Usage
```bash
# Process raw ECG data
python data_processing.py

# Run training pipeline
jupyter notebook main.ipynb

# Validate model
jupyter notebook validation.ipynb
```