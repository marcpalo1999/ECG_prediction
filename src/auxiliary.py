pathology_severity_groups= {
    'Ischemic Disorders': {
        'codes': [
            '164865005',  # MI (includes MIBW, MIFW, MILW, MISW)
            '164917005',  # Abnormal Q wave
            '429622005',  # ST drop down
            '164930006',  # ST extension
            '428750005',  # ST-T Change
            '164931005',  # ST tilt up
            '164934002',  # T wave Change
            '59931005',   # T wave opposite
        ],
        'severity': 7
    },
    'Primary Electrical Disorders': {
        'codes': [
            '164889003',  # Atrial Fibrillation
            '164890007',  # Atrial Flutter
            '164896001', # Ventricular fibrilation
            '425856008', #  (Paroxysmal ventricular tachycardia) - critical arrhythmi
            '81898007', # (Ventricular escape rhythm) - compensatory arrhythmia
            '29320008', # (Ectopic rhythm) - abnormal rhythm origin'
            '426761007',  # Supraventricular Tachycardia
            '713422000',  # Atrial Tachycardia
            '233896004',  # AVNRT
            '233897008',  # AVRT
            '17338001',   # Ventricular premature beat
            '75532003',   # Ventricular escape beat
            '11157007',   # Ventricular bigeminy
            '251180001',  # Ventricular escape trigeminy
            '195060002',  # Ventricular preexcitation
            '74390002',   # WPW
            '111975006',  # QT interval extension
            '428417006',  # Early repolarization of ventricles
            '13640000',   # Ventricular fusion wave
        ],
        'severity': 6
    },
    'Conduction System Disease': {
        'codes': [
            '270492004',  # 1st degree AV block
            '195042002',  # 2nd degree AV block
            '54016002',   # 2nd degree AV block type 1
            '28189009',   # 2nd degree AV block type 2
            '27885002',   # 3rd degree AV block
            '233917008',  # AV block
            '164909002',  # LBBB (includes LBBBB, LFBBB)
            '59118001',   # RBBB
            '698252002',  # IDC/IVB
            '164947007',  # PR interval extension
            '195101003',  # WAVN/SAAWR
        ],
        'severity': 5
    },
    'Chamber Abnormalities': {
        'codes': [
            '164873001',  # LVH
            '89792004',   # RVH
            '446358003',  # RAH
            '251146004',  # LVQRSAL
            '251148003',  # LVQRSCL
            '251147008',  # LVQRSLL
            '39732003',   # ALS
            '47665007',   # ARS
            '251199005',  # CCR
            '251198002',  # CR
            '164942001',  # fQRS Wave
            '55827005', # (Left ventricular hypertrophy)
        ],
        'severity': 4
    },
    'Atrial Abnormalities': {
        'codes': [
            '251173003',  # Atrial bigeminy
            '284470004',  # Atrial premature beats
            '426995002',  # Junctional escape beat
            '251164006',  # Junctional premature beat
            '164912004',  # P wave Change
        ],
        'severity': 3
    },
    'Benign Variants': {
        'codes': [
            '426177001',  # Sinus Bradycardia
            '427084000',  # Sinus Tachycardia
            '427393009',  # Sinus Irregularity
            '164937009',  # U wave
        ],
        'severity': 2
    },
    'Others': {
        'codes': [
            '10370003', # (Artificial pacing) - intervention, not natural condition
            '106068003', # (Atrial rhythm) - too general to classify specifically],
        ],
        'severity':1
    },
    'Normal': {
        'codes': ['426783006'],  # Sinus Rhythm
        'severity': 0
    }
}

def map_cardiac_condition(diagnosis_codes):
    # Find matching categories
    matching_categories = [
        category for category, details in pathology_severity_groups.items()
        if any(code in diagnosis_codes for code in details['codes'])
    ]
    
    # If multiple matches, select highest severity
    if matching_categories:
        return max(
            matching_categories, 
            key=lambda cat: pathology_severity_groups[cat]['severity']
        )
    
    return 'Others'


import seaborn as sns
import matplotlib.pyplot as plt

def histogram_labels(labels, pathology_order):


    # Assuming dev_labels_downsized is your Series with the class labels
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        x=labels,
        order=pathology_order
    )
    plt.title('ECG Classification Distribution by severity level')
    plt.xlabel('Diagnosis Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()


import seaborn as sns

def analyze_feature_distributions(df, features=None, target_col='diagnosis_label'):
    """
    Analyze feature distributions using KDE plots by class.
    """
    if features is None:
        features = ['median_hr', 'mean_hr', 'std_hr', 'min_hr', 'max_hr', 'age']
    
    # Calculate skewness for each feature
    skewness = df[features].skew()
    print("Features skewness:")
    for feature, skew_val in skewness.items():
        print(f"  - {feature}: {skew_val:.2f}")
    
    # Plot KDE distributions by class
    for feature in features:
        plt.figure(figsize=(12, 6))
        sns.kdeplot(
            data=df,
            x=feature, 
            hue=target_col,
            fill=True,
            common_norm=False,  # Each group normalized separately
            palette="viridis",
            alpha=0.5
        )
        plt.title(f'{feature} Distribution by Class (Skewness: {skewness[feature]:.2f})')
        plt.tight_layout()
        plt.show()
    
    return skewness

import pandas as pd
def assess_missing_values(df, features=None, target_col='diagnosis_label', hue = False):
    """
    Assess missing values in the dataset and visualize their relationship with target classes.
    """
    if features is None:
        features = ['median_hr', 'mean_hr', 'std_hr', 'min_hr', 'max_hr', 'age']
    
    # Calculate missing values
    missing = df[features].isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    # Create summary dataframe
    missing_summary = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_pct
    }).sort_values('Missing Values', ascending=False)
    
    print("Missing Values Summary:")
    print(missing_summary)
    
    # Bar plot of missing values by feature
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=missing_summary.index, y='Percentage', data=missing_summary)
    plt.title('Percentage of Missing Values by Feature')
    plt.ylabel('Missing Values (%)')
    plt.xlabel('Features')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    if hue==True:
        if missing.sum() > 0 and target_col in df.columns:
            # For each feature with missing values
            for feature in features:
                if missing[feature] > 0:
                    # Create figure for this feature
                    plt.figure(figsize=(12, 6))
                    
                    # Count plot of missing values by class
                    ax = sns.countplot(x=target_col, 
                                    data=df,
                                    hue=df[feature].isnull(),
                                    hue_order=[True, False],
                                    palette=['#ff7f0e', '#1f77b4'])
                    
                    # Add labels and formatting
                    plt.title(f'Missing Values for {feature} by {target_col}')
                    plt.xlabel(target_col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=90)
                    plt.legend(title='Missing', labels=['Missing', 'Not Missing'])
                    
                    # Add value labels on bars
                    for p in ax.patches:
                        height = p.get_height()
                        if height > 0:  # Only add text if bar has height
                            ax.text(p.get_x() + p.get_width()/2., height + 0.1, 
                                f'{int(height)}', ha="center")
                    
                    plt.tight_layout()
                    plt.show()
    
    return missing_summary


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
            lead_II = ecg_data[id].loc[:,'II']
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

def plot_grouped_boxplots(df, features, group_var, hue_var=None, order=None, figsize=(14, 8)):
    """
    Create boxplots of a feature grouped by a primary variable and colored by a secondary variable.
    
    Args:
        df: DataFrame containing the data
        feature: Feature to plot (y-axis)
        group_var: Primary grouping variable (x-axis)
        hue_var: Secondary grouping variable (color differentiation)
        order: List specifying the order of categories on x-axis
        figsize: Figure size tuple
    """
    for feature in features:
        plt.figure(figsize=figsize)
        
        # Use the pathology_order if order not provided
        if order is None and 'pathology_order' in globals():
            order = pathology_order
        
        # Create boxplot with seaborn
        ax = sns.boxplot(
            data=df,
            x=group_var,
            y=feature,
            hue=hue_var,
            order=order,
            palette="viridis"
        )
        
        # Add descriptive elements
        plt.title(f'Distribution of {feature} by {group_var}' + 
                (f' and {hue_var}' if hue_var else ''))
        plt.xlabel(group_var)
        plt.ylabel(feature)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=90)
        
        # Adjust legend if hue is provided
        if hue_var:
            plt.legend(title=hue_var, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()