# %% [markdown]
# TO DO : 
# 
#     - Separate data into metadata df and signal dict, so that it is easier to analyse data and to access signal just with key (sample_id) DONE
#     -  Check for normal ecgs and train the classifier with AF vs normal (NORMAL ECGS ARE STATED AS SINUS RYTHM)
#     - Only 100 samples are loaded for velocity. All should be loaded at the end DONE
#     - If waiting time is too long use a sql based system

# %% [markdown]
# ## Loading data

# %%
# !python -V > full_requirements.txt && pip list --format=freeze >> full_requirements.txt

# %%
import os
import wfdb  # To read the .hea file
import scipy.io as sio  # To read .mat files
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
import numpy as np
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal 
import pandas as pd


#Directories
dataset_dir = '../a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords'

g_leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
fs = 500

# %%
disease_map_path = '../a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/ConditionNames_SNOMED-CT.csv'
disease_map = pd.read_csv(disease_map_path)
disease_map = disease_map.set_index('Snomed_CT')
def disease_map_func(diagnosis, disease_map = disease_map):


    diagnosis = [disease_map.loc[int(code), "Full Name"] for code in diagnosis.split(',')]
    return diagnosis

# %%
ecg_data = {}
patient_data = {}

# Loop through each record in the dataset directory
for directory1 in sorted([dir for dir in os.listdir(f"{dataset_dir}") if not dir.startswith('.')])[0:3]: #errase the [0:3] to get all the files
    for directory2 in sorted([dir for dir in os.listdir(f"{dataset_dir}/{directory1}") if not dir.startswith('.')]):
        for record_hea in [dir for dir in os.listdir(f"{dataset_dir}/{directory1}/{directory2}") if not dir.startswith('.')]:
            if record_hea.endswith(".hea"):  # Process .hea files
                try:
                    patient_id = record_hea.split(".hea")[0]
                    record_path = f"{dataset_dir}/{directory1}/{directory2}/{patient_id}"

                    # Read the header (.hea) file
                    record = wfdb.rdheader(record_path)

                    # Extract metadata
                    age = None
                    sex = None
                    diagnosis = None
                    for comment in record.comments:
                        if comment.startswith("Age:"):
                            age = comment.split(":")[1].strip()
                        if comment.startswith("Sex:"):
                            sex = comment.split(":")[1].strip()
                        if comment.startswith("Dx:"):
                            diagnosis = comment.split(":")[1].strip()  # This gives you the SNOMED codes
                    
                    # Read the 12-lead ECG signal from the .mat file
                    mat_file_path = f"{record_path}.mat"
                    mat_data = sio.loadmat(mat_file_path)
                    ecg_signals = mat_data['val']  # 'val' typically holds the ECG signal in PhysioNet datasets
                    ecg_signals= pd.DataFrame(ecg_signals.T, columns  = g_leads)
                    # Store the data in the dictionary
                    ecg_data[patient_id] = {
                        "ecg_signals": ecg_signals,  # 12-lead ECG signals
                        }
                    patient_data[patient_id] = {
                        "diagnosis_code": [disease for disease in diagnosis.split(',')],  # Disease label (SNOMED codes)
                        "diagnosis_name": disease_map_func(diagnosis),
                        "age": age,
                        "sex": sex
                    }
                except Exception as e:
                    print(f"{patient_id}: {e}")

patient_data=pd.DataFrame(patient_data).T


# %%
disease_map

# %%
pd.DataFrame(patient_data).T

# %% [markdown]
# ## Funcitons

# %%
from scipy import signal
import scipy
import random
from scipy.signal import find_peaks, welch

def temp_freq_plot(signal, title):

    fig,ax = plt.subplots(12,2, figsize=(20,12*4))
    for i in range(0,12):
        _key = g_leads[i]
        #ax[i].set_title(_key)
        ax[i,0].plot(signal[_key], color='black', linewidth=0.6)
        ax[i,0].set_ylim((-1500,1500))
        ax[i,0].set_xticks(   np.arange(0,5001,500)  )   
        ax[i,0].set_xticklabels(   np.arange(0,5001,500)/fs  )   
        ax[i,0].grid(axis='x')
        ax[i,0].annotate(_key,(-200,0))
        #ax[i].set_xlabel('Time(sec)')
        #ax[i].set_ylabel('mV')
        ax[i,0].hlines(0,0,5000,color='black', linewidth=0.3)

        frequencies, psd_values = welch( signal[_key], fs, nperseg=1024)

        # Plotting the estimated PSD
        ax[i,1].semilogy(frequencies, psd_values)
        ax[i,1].set_title('Power Spectral Density (PSD) Estimate using Welch\'s Method')
        ax[i,1].set_xlabel('Frequency (Hz)')
        ax[i,1].set_ylabel('PSD (V^2/Hz)')

    plt.suptitle(f"{title}")
    plt.tight_layout()
    plt.show()




def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Design a Notch Filter to remove 60 Hz power line interference
def notch_filter(frequency, fs, quality_factor=30):
    b, a = signal.iirnotch(frequency, quality_factor, fs)
    return b, a

def apply_notch_filter(data, frequency, fs, quality_factor=30):
    b, a = notch_filter(frequency, fs, quality_factor)
    y = signal.filtfilt(b, a, data)
    return y


def scipy_notch_filter(data, fs, frequency, quality):
    return scipy.signal.filtfilt(*scipy.signal.iirnotch(frequency / (fs / 2), quality), data)



# %% [markdown]
# ## Signal filtering pipeline

# %% [markdown]
# Frequency response and plot of random patient

# %%
np.random.seed(42)  # Ensure reproducibility

random_patient = random.choice(list(ecg_data.keys()))
patient_signal = ecg_data[random_patient]["ecg_signals"]

temp_freq_plot(signal = patient_signal, title=random_patient)

# %%
#applying filters
for patient in ecg_data.keys():

    filtered_signal = ecg_data[patient]['ecg_signals']#.apply(lambda x: signal.detrend(x))
    filtered_signal= filtered_signal.apply(lambda x:  signal.detrend(x))
    filtered_signal = filtered_signal.apply(lambda x: highpass_filter(data = x, cutoff=5, fs=fs))

    filtered_signal = filtered_signal.apply(lambda x: lowpass_filter(data = x, cutoff=140, fs=fs))
    filtered_signal = filtered_signal.apply(lambda x: scipy_notch_filter(data=x, frequency=50, fs=fs, quality=30))
    ecg_data[patient]['ecg_signals_filtered'] = filtered_signal



# %%
patient_signal_filtered = ecg_data[random_patient]['ecg_signals_filtered']

temp_freq_plot(signal = patient_signal_filtered, title = random_patient)

# %% [markdown]
# ## Illness distribution and data filtering

# %%
import sys
sys.path.append('/Users/marcpalomer/Documents/Personal/ECG_prediction/utils')
import auto_EDA as eda 
%matplotlib inline


# %%
def classify_arrhythmia(diagnosis_codes):
   # Arrhythmia SNOMED codes
    #    snomed_codes = {
    #    'AF': '49436004',
    #    'RBBB': '59118001', 
    #    'LBBB': '28189009',
    #    'IAVB': '270492004',
    #    'PAC': '284470004',
    #    'PVC': '427172004',
    #    'MI': '22298006'
    #}
   arrhythmia_codes = ['49436004', '59118001', '28189009', '270492004', '284470004', '427172004', '22298006']
   
   # Sinus Rhythm Normal code
   healthy_code = '426783006'
   
   # Check if any arrhythmia code is present
   if any(code in diagnosis_codes for code in arrhythmia_codes):
       return 'Arrhythmia'
   # Check if healthy code present
   elif healthy_code in diagnosis_codes:
       return 'Healthy'
   # Otherwise other
   else:
       return 'Other'

# Apply to dataframe
patient_data['arrhythmia'] = patient_data['diagnosis_code'].apply(classify_arrhythmia)
patient_data['arrhythmia'].hist()
plt.show()

# %% [markdown]
# ### HRV calculation in lead II (gold standard) (it is a time saries for patient so like a new lead...)

# %%
import neurokit2 as nk
import numpy as np

def validate_rpeaks(rpeaks, fs):
    # Remove physiologically impossible R-peaks
    rr_intervals = np.diff(rpeaks) / fs
    valid_rr = (rr_intervals >= 0.2) & (rr_intervals <= 2.0)  
    valid_peaks = rpeaks[1:][valid_rr]
    return valid_peaks

def calculate_hr_metrics(rpeaks, fs):
    rr_intervals = np.diff(rpeaks) / fs
    hr = 60 / rr_intervals
    return np.median(hr), np.mean(hr), np.std(hr), np.min(hr), np.max(hr)

def calculate_heartrate(record, fs):
    # Find R-peaks using neurokit2
    rpeaks = list(nk.ecg_findpeaks(record, sampling_rate=fs).values())[0]
    rpeaks = validate_rpeaks(rpeaks, fs)
    return calculate_hr_metrics(rpeaks, fs)

def add_hr_metrics(patient_data, ecg_data):
    metrics = {'median_hr': [], 'mean_hr': [], 'std_hr': [], 'min_hr': [], 'max_hr': []}
    
    for id in patient_data.index:
        if id in ecg_data:
            lead_II = ecg_data[id]['ecg_signals_filtered'].loc[:,'II']
            try:
                median_hr, mean_hr, std_hr, min_hr, max_hr = calculate_heartrate(lead_II, fs=500)
            except:
                median_hr, mean_hr, std_hr, min_hr, max_hr = [np.nan,np.nan,np.nan,np.nan,np.nan]
            metrics['median_hr'].append(median_hr)
            metrics['mean_hr'].append(mean_hr)
            metrics['std_hr'].append(std_hr)
            metrics['min_hr'].append(min_hr)
            metrics['max_hr'].append(max_hr)
        else:
            for key in metrics:
                metrics[key].append(None)
    
    for metric, values in metrics.items():
        patient_data[metric] = values
    
    return patient_data

patient_data = add_hr_metrics(patient_data, ecg_data)

# %%
patient_data['Healthy'] = ['HEALTHY' if 'Health' in col else 'ILL' for col in patient_data['arrhythmia'] ]
patient_data

# %%
# import automatic_reporting as AR 
# reportName = "Patient Data and HRV"
analyse_features= ['median_hr',	'mean_hr','std_hr',	'min_hr', 'max_hr', 'age']
# control_features= ['sex', 'arrhythmia', 'Healthy']




# PATH = '.'

# report_builder = AR.AutoReport(main_path = f"{PATH}",
#                                     data = patient_data,
#                                     analyse_features= analyse_features,
#                                     control_features= control_features)

# report_builder.generate_report(ReportName = reportName)

# %% [markdown]
# # Dumb classifier

# %%
ML_dataset = patient_data[analyse_features+['Healthy']]

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline



def quick_classify(df, target_col, features=None):
    """
    Quick classification using RandomForest with minimal preprocessing.
    
    Args:
        df: pandas DataFrame with your data
        target_col: name of the column to predict
        features: list of feature columns to use (optional, uses all except target if None)
    """
    # Select features
    if features is None:
        features = [col for col in df.columns if col != target_col]
    
    # Prepare data
    X = df[features]
    y = df[target_col]
    
    # Handle non-numeric columns
    # X = pd.get_dummies(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    return {
        'model': model,
        'scaler': scaler,
        'features': X.columns.tolist()
    }

# %%
quick_classify_dict = quick_classify(ML_dataset, target_col='Healthy')
quick_classify_dict

# %%


# %%
from sklearn.model_selection import train_test_split
import RF_pipeline

def analyze_model(df, target_col, features=None, test_size=0.2, random_state=42):
    """Main function to analyze the model."""
    if features is None:
        features = [col for col in df.columns if col != target_col]
    

    X = df[features]
    y = df[target_col]

    # Preprocess data
    X_processed = X.copy()
    for column in X_processed.columns:
        X_processed[column] = pd.to_numeric(X_processed[column], errors='coerce')
    X_processed = X_processed.fillna(X_processed.median())
    y_processed = pd.factorize(y)[0]
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=test_size, random_state=random_state)
    
    analyzer = RF_pipeline.RandomForestAnalyzer()
    visualizer = RF_pipeline.ModelVisualizer()

    return X_train, X_test, y_train, y_test, analyzer, visualizer, features
   
X_train, X_test, y_train, y_test, analyzer, visualizer, features = analyze_model(df=ML_dataset, target_col='Healthy', features=None)

# %%
# OOB Analysis
print("Performing OOB Analysis...")
oob_results = analyzer.analyze_oob(X_train, y_train)
visualizer.plot_oob_analysis(oob_results)

# Feature Importance Analysis
print("\nAnalyzing Feature Importance...")
importance_results = analyzer.analyze_feature_importance(X_train, y_train)
visualizer.plot_feature_importance(importance_results)

# Cross Validation
print("\nPerforming Cross Validation...")
cv_results = analyzer.cross_validate(X_train, y_train)
visualizer.plot_cv_results(cv_results)

# %% [markdown]
# As we have seen, the distribution in cv of the performance metrics is better than the performance over the uncrossvalidated healthy, also pointing to the fact that the model is better at predicting the majority class than the minority. Anyway it is incredibly good at both.

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# This goes inline with the PCA...

# %% [markdown]
# ## Decision boundary display

# %%
# Decision Boundary Analysis
print("\nVisualizing Decision Boundaries...")
model = analyzer.get_fitted_model(X_train, y_train)

print("PCA-based boundaries:")
boundary_viz = RF_pipeline.BoundaryVisualizer()
X_pca = boundary_viz.plot_boundaries_2d(X_train, y_train, model, method='pca')

# %% [markdown]
# ## Shap values: 
# - What favours arrythmia?

# %%
import shap


X_sub = shap.sample(X_train)
explainer = shap.Explainer(model.predict_proba, X_sub)
shap_values = explainer(X_test)

class_index = 1
data_index = 1

shap.plots.waterfall(shap_values[data_index,:,class_index], )

shap.initjs()
shap.plots.force(shap_values[data_index,:,class_index])

shap.plots.beeswarm(shap_values[:,:,class_index])

shap.plots.bar(shap_values[:,:,class_index])

shap.plots.scatter(shap_values[:, 'max_hr',1])

shap.plots.scatter(shap_values[:, 'max_hr',1], color=shap_values[:,:,1])

# %% [markdown]
# # DL CNN prediction: 
# - Will we beat HRV RF classification? (With only 1 lead!)

# %% [markdown]
# 
# ## Pipeline Structure
# 
# ### Training Phase
# 1. **Data Preparation**
#   - Raw ECG dictionary + labels → `prepare_data()` → `normalize_signals()`
#   - Train/val/test split
#   - Dataset & DataLoader creation for batching
# 
# 2. **Training Cycle**
#   - DataLoader feeds batches to ModelTrainer
#   - Forward pass through ECGNet
#   - Loss calculation, backpropagation
#   - Validation performance check
#   - Save best model
#   - Track metrics history
# 
# ### Evaluation Phase
# 1. **Model Assessment**
#   - Load best model weights
#   - Full forward pass on test set
#   - Generate predictions/probabilities
# 
# 2. **Results**
#   - Performance metrics calculation
#   - Visualization generation
#   - Save all results
# 
# ## DL Architecture
# 
# ### Input Processing
# - 12-lead ECG signals
# - 5000 timepoints per lead
# - Normalized per lead
# 
# ### Feature Extraction
# - Conv1d (k=50): QRS complex detection
# - Conv1d (k=7): Wave morphology
# - Conv1d (k=5): Fine details
# - Increasing channels (12→32→64→128) for feature hierarchy
# 
# ### Each Conv Block
# - BatchNorm: Training stability
# - ReLU: Non-linearity
# - MaxPool: Dimension reduction
# 
# ### Classification
# - AdaptivePool: Fixed output size
# - FC layers (6400→256→64→2)
# - Dropout layers prevent overfitting
# - Output: Binary classification probabilities

# %% [markdown]
# 

# %% [markdown]
# Label encoding

# %%
# Prepare your labels
labels_dict = patient_data['Healthy'].reset_index(drop=False)
labels_dict = labels_dict.rename(columns={'Healthy':'label'})

# Save categories before encoding
categories = pd.Categorical(labels_dict['label']).categories

# Encode labels
labels_dict['label'] = pd.Categorical(labels_dict['label']).codes

# Create and print the encoding dictionary
encoding_dict = dict(enumerate(categories))

print("\nLabel encoding dictionary:")
for code, label in encoding_dict.items():
    print(f"{label} -> {code}")

print("\nLabel distribution:")
print(labels_dict['label'].value_counts())

# %% [markdown]
# ### DL model loading (or training)

# %%
ecg_data['JS00067'].keys()

# %%
import importlib
import CNN
import traceback
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Reload the module to get the latest changes
importlib.reload(CNN)



# Verify model path
model_path = "/Users/marcpalomer/Documents/Personal/ECG_prediction/Results/DL_model/results_20241214_202359/best_model.pth"
print(f"Model file exists: {Path(model_path).exists()}")

# Configuration settings
config = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'seed': 42
}

# Run the main function
try:
    print(f"Loading pretrained model from {model_path}")
    results = CNN.main(ecg_data, labels_dict, config, load_model_path=model_path)
    
    # Print results
    print("\nResults keys:", results.keys() if results else "No results")
    model = results['model']
    output_dir = results['output_dir']
    importance_results = results['importance_results']

    print(f"\nResults saved in: {output_dir}")
    if 'test_metrics' in results:
        print("\nTest Metrics:")
        print(f"ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
        print(f"PR AUC: {results['test_metrics']['pr_auc']:.4f}")
        print(f"Test Loss: {results['test_metrics']['test_loss']:.4f}")

        
except Exception as e:
    print(f"Error occurred: {str(e)}")
    traceback.print_exc()

# %% [markdown]
# ### External validation Performance
# 
# - Try performance on other 4 random directories of samples to assess actual performance of model

# %%
sorted([dir for dir in os.listdir(f"{dataset_dir}") if not dir.startswith('.')])[-3::]

# %%
external_validation_ecg_data = {}
external_validation_data = {}

# Loop through each record in the dataset directory
for directory1 in sorted([dir for dir in os.listdir(f"{dataset_dir}") if not dir.startswith('.')])[-10::-6]: #errase the [0:3] to get all the files
    for directory2 in sorted([dir for dir in os.listdir(f"{dataset_dir}/{directory1}") if not dir.startswith('.')]):
        for record_hea in [dir for dir in os.listdir(f"{dataset_dir}/{directory1}/{directory2}") if not dir.startswith('.')]:
            if record_hea.endswith(".hea"):  # Process .hea files
                try:
                    patient_id = record_hea.split(".hea")[0]
                    record_path = f"{dataset_dir}/{directory1}/{directory2}/{patient_id}"

                    # Read the header (.hea) file
                    record = wfdb.rdheader(record_path)

                    # Extract metadata
                    age = None
                    sex = None
                    diagnosis = None
                    for comment in record.comments:
                        if comment.startswith("Age:"):
                            age = comment.split(":")[1].strip()
                        if comment.startswith("Sex:"):
                            sex = comment.split(":")[1].strip()
                        if comment.startswith("Dx:"):
                            diagnosis = comment.split(":")[1].strip()  # This gives you the SNOMED codes
                    
                    # Read the 12-lead ECG signal from the .mat file
                    mat_file_path = f"{record_path}.mat"
                    mat_data = sio.loadmat(mat_file_path)
                    ecg_signals = mat_data['val']  # 'val' typically holds the ECG signal in PhysioNet datasets
                    ecg_signals= pd.DataFrame(ecg_signals.T, columns  = g_leads)
                    # Store the data in the dictionary
                    external_validation_ecg_data[patient_id] = {
                        "ecg_signals": ecg_signals,  # 12-lead ECG signals
                        }
                    external_validation_data[patient_id] = {
                        "diagnosis_code": [disease for disease in diagnosis.split(',')],  # Disease label (SNOMED codes)
                        "diagnosis_name": disease_map_func(diagnosis),
                        "age": age,
                        "sex": sex
                    }
                except Exception as e:
                    print(f"{patient_id}: {e}")

external_validation_data=pd.DataFrame(external_validation_data).T


# %%
#applying filters
for patient in external_validation_ecg_data.keys():

    filtered_signal = external_validation_ecg_data[patient]['ecg_signals']#.apply(lambda x: signal.detrend(x))
    filtered_signal= filtered_signal.apply(lambda x:  signal.detrend(x))
    filtered_signal = filtered_signal.apply(lambda x: highpass_filter(data = x, cutoff=5, fs=fs))

    filtered_signal = filtered_signal.apply(lambda x: lowpass_filter(data = x, cutoff=140, fs=fs))
    filtered_signal = filtered_signal.apply(lambda x: scipy_notch_filter(data=x, frequency=50, fs=fs, quality=30))
    external_validation_ecg_data[patient]['ecg_signals_filtered'] = filtered_signal



# %%
def classify_arrhythmia(diagnosis_codes):
   # Arrhythmia SNOMED codes
    #    snomed_codes = {
    #    'AF': '49436004',
    #    'RBBB': '59118001', 
    #    'LBBB': '28189009',
    #    'IAVB': '270492004',
    #    'PAC': '284470004',
    #    'PVC': '427172004',
    #    'MI': '22298006'
    #}
   arrhythmia_codes = ['49436004', '59118001', '28189009', '270492004', '284470004', '427172004', '22298006']
   
   # Sinus Rhythm Normal code
   healthy_code = '426783006'
   
   # Check if any arrhythmia code is present
   if any(code in diagnosis_codes for code in arrhythmia_codes):
       return 'Arrhythmia'
   # Check if healthy code present
   elif healthy_code in diagnosis_codes:
       return 'Healthy'
   # Otherwise other
   else:
       return 'Other'

# Apply to dataframe
external_validation_data['arrhythmia'] = external_validation_data['diagnosis_code'].apply(classify_arrhythmia)
external_validation_data['arrhythmia'].hist()
plt.show()

# %%
external_validation_data

# %%
external_validation_labels_dict = external_validation_data['arrhythmia'].reset_index(drop=False)

external_validation_labels_dict['label'] = ['HEALTHY' if 'Health' in col else 'ILL' for col in external_validation_labels_dict['arrhythmia'] ]
external_validation_labels_dict

# %%


# Save categories before encoding
categories = pd.Categorical(external_validation_labels_dict['label']).categories

# Encode labels
external_validation_labels_dict['label'] = pd.Categorical(external_validation_labels_dict['label']).codes

# Create and print the encoding dictionary
encoding_dict = dict(enumerate(categories))

print("\nLabel encoding dictionary:")
for code, label in encoding_dict.items():
    print(f"{label} -> {code}")

print("\nLabel distribution:")
print(external_validation_labels_dict['label'].value_counts())

# %%
external_validation_labels_dict

# %%
import importlib
import CNN
import traceback
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Reload the module to get the latest changes
importlib.reload(CNN)



# Verify model path
model_path = "/Users/marcpalomer/Documents/Personal/ECG_prediction/Results/DL_model/results_20241214_202359/best_model.pth"
print(f"Model file exists: {Path(model_path).exists()}")

# Configuration settings
config = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'seed': 42
}

# Run the main function
try:
    print(f"Loading pretrained model from {model_path}")
    results = CNN.main(external_validation_ecg_data, external_validation_labels_dict, config, load_model_path=model_path)
    
    # Print results
    print("\nResults keys:", results.keys() if results else "No results")
    model = results['model']
    output_dir = results['output_dir']
    importance_results = results['importance_results']

    print(f"\nResults saved in: {output_dir}")
    if 'test_metrics' in results:
        print("\nTest Metrics:")
        print(f"ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
        print(f"PR AUC: {results['test_metrics']['pr_auc']:.4f}")
        print(f"Test Loss: {results['test_metrics']['test_loss']:.4f}")

        
except Exception as e:
    print(f"Error occurred: {str(e)}")
    traceback.print_exc()


