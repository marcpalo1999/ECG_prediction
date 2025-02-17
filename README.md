# ECG Disease Classification

Multi-class ECG disease classification using Random Forest and CNN approaches.



## Data Origin:

https://physionet.org/content/ecg-arrhythmia/1.0.0/

## Project Structure

- `data_processing.py`: ECG signal preprocessing
- `RF_pipeline.py`: Random Forest classifier with HRV features
- `CNN_pipeline.py`: Deep learning model using raw ECG signals
- `main.ipynb`: Main training pipeline
- `main_validation.ipynb`: Model validation on held-out data

## Script workflow

Download data from source -> `data_processing.py` -> `main.ipynb` -> `main_validation.ipynb`

## Methodology

### Data Split
- 70/30 development/validation split
- Stratified by condition
- Post-split categorization into four disease classes

### Data Processing
- 12-lead ECG signal filtering (5-140Hz bandpass, 50Hz notch)
- HRV feature extraction from Lead II
- Disease mapping: Normal/Arrhythmia/Conduction/Structural

### Random Forest Approach
- Features: HRV metrics + patient age
- Cross-validated performance
- Feature importance analysis
- PCA-based decision boundary visualization

### CNN Architecture
- Input: 12-lead ECG (12x5000)
- 3 convolutional blocks
- BatchNorm, LeakyReLU, MaxPool
- Dense layers with dropout

### Performance assessment
- ROC curve by condition
- Confusion matrix by condition
- Accuracy, precision, recall & f1-score

## Requirements
- Python 3.8+
- Main libraries: keras, scikit-learn, neurokit2
- Full dependencies in `full_requirements.txt`