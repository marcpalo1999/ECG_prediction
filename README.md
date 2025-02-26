# ECG Disease Classification using Random Forest

A comprehensive approach to cardiac condition classification from 12-lead ECG data using machine learning techniques.

## Data Source
[PhysioNet ECG Arrhythmia Database](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
- Large-scale 12-lead ECG database for arrhythmia and cardiac condition analysis
- Contains SNOMED-CT coded diagnoses for supervised learning

## Project Structure
```
├── data_processing.py        # ECG signal filtering and preprocessing pipeline
├── auxiliary.py              # Helper functions and cardiac condition mapping
├── ecg_data_pipeline.py      # Feature extraction and condition classification 
├── ecg_model_pipeline.py     # Model training, evaluation and visualization
├── main.ipynb                # Development and training pipeline
└── main_validation.ipynb     # Independent model validation (planned)
```

## Methodology

### Signal Processing
1. **Filtering Pipeline**
   - Bandpass filter (5-140Hz) to remove baseline drift and high-frequency noise
   - Notch filter (50Hz) to eliminate power line interference
   - Signal detrending for each of the 12 leads

2. **Quality Control**
   - R-peak validation with physiological RR interval constraints (0.2-2.0s)
   - Automated quality metrics assessment
   - Artifact detection and handling

### Disease Categorization
The classifier organizes cardiac conditions into severity-based groups:

1. **Normal**: Normal sinus rhythm (severity 0)
2. **Benign Variants**: Sinus bradycardia/tachycardia, normal variants (severity 2)
3. **Atrial Abnormalities**: P-wave changes, atrial premature contractions (severity 3)
4. **Chamber Abnormalities**: LVH, RVH, chamber enlargement (severity 4)
5. **Conduction System Disease**: Bundle branch blocks, AV blocks (severity 5)
6. **Primary Electrical Disorders**: Arrhythmias including AF, VT, VF (severity 6)
7. **Ischemic Disorders**: Myocardial infarction, ST-T changes (severity 7)

### Pipeline Components

1. **Data Preparation**
   - Stratified development/validation split (70/30)
   - Cardiac condition classification by severity
   - Heart rate variability metrics extraction
   - Missing value analysis and handling

2. **Feature Engineering**
   - Heart rate statistics from filtered Lead II (mean, median, std, min, max)
   - Patient demographic features (age)
   - Calculated using neurokit2 library

3. **Model Training**
   - Random Forest classifier with class balancing
   - Cross-validation performance assessment
   - Feature importance stability analysis
   - PCA-based decision boundary visualization

4. **Model Evaluation**
   - Multi-class ROC curves and AUC analysis
   - Confusion matrices with sensitivity/specificity
   - Per-class precision, recall, F1-score
   - Both binary (normal vs abnormal) and multi-class evaluation

## Requirements
- Python 3.8+
- scikit-learn
- neurokit2
- pandas
- numpy
- matplotlib
- seaborn

Full dependencies in `full_requirements.txt`

## Usage
```bash
# Process raw ECG data
python data_processing.py

# Run training pipeline
jupyter notebook main.ipynb

# Validate model (planned)
jupyter notebook main_validation.ipynb
```

## Validation Approach
The validation pipeline (main_validation.ipynb) will use a completely independent dataset to assess the model's generalization ability:

- Uses validation indices not seen during development
- Evaluates against clinical gold standard diagnoses
- Assesses model performance across different demographic groups
- Analyzes critical misclassification patterns

## Future Improvements
Based on analysis of current limitations:
- Enhanced feature extraction with ECG morphology measurements
- Hierarchical classification approach (normal vs abnormal, then specific condition)
- Advanced techniques for handling class imbalance
- Focus on reducing clinically significant misclassifications