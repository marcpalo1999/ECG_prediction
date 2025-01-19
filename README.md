# ECG_prediction
This repository is a project on ECG signal processing and ML disease prediction using python.

The data used is 12 lead electrocardiogram data of 10 seconds / recording. The data can be found in physionet: https://physionet.org/content/ecg-arrhythmia/1.0.0/
It consists of the beforementioned recordings labeled with the SNOMED-CT code of the cardiopathology diagnosed to the recording. 

The project consists of classifying the patients in pathological vs healty in two different ways:

	- ML pipeline:
		- Feature extraction on 12 lead ecg to extract HRV metrics. 
		- PCA of the data
		- RF classification with outstanding results
		- RF colouring of the PCA space to get space borders
		- shapley values to further understand the importance and impact of the features on model output
		- CV accuracy, precision, recall and f1 distribution scores for the RF classifier

	- DL pipeline:
		## Pipeline Structure

		### Training Phase
		1. **Data Preparation**
  			- Raw ECG dictionary + labels → `prepare_data()` → `normalize_signals()`
  			- Train/val/test split
  			- Dataset & DataLoader creation for batching

		2. **Training Cycle**
  			- DataLoader feeds batches to ModelTrainer
  			- Forward pass through ECGNet
  			- Loss calculation, backpropagation
  			- Validation performance check
 			- Save best model
  - Track metrics history

### Evaluation Phase
1. **Model Assessment**
  - Load best model weights
  - Full forward pass on test set
  - Generate predictions/probabilities

2. **Results**
  - Performance metrics calculation
  - Visualization generation
  - Save all results

## DL Architecture

### Input Processing
- 12-lead ECG signals
- 5000 timepoints per lead
- Normalized per lead

### Feature Extraction
- Conv1d (k=50): QRS complex detection
- Conv1d (k=7): Wave morphology
- Conv1d (k=5): Fine details
- Increasing channels (12→32→64→128) for feature hierarchy

### Each Conv Block
- BatchNorm: Training stability
- ReLU: Non-linearity
- MaxPool: Dimension reduction

### Classification
- AdaptivePool: Fixed output size
- FC layers (6400→256→64→2)
- Dropout layers prevent overfitting
- Output: Binary classification probabilities
