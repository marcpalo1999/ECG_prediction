from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import neurokit2 as nk
from functools import wraps
import time


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


class ECGFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from ECG data including HRV metrics"""
    
    def __init__(self, fs=500, features_to_extract=None):
        self.fs = fs
        self.features_to_extract = features_to_extract or ['hr_metrics', 'age']
        self.feature_medians = {}
        
    def fit(self, X, y=None):
        """Fit the feature extractor to the data"""
        # Nothing to fit, but we could compute feature statistics here if needed
        return self
        
    def transform(self, ecg_data, patient_data=None):
        """
        Transform ECG data dictionary and patient metadata into feature matrix
        
        Parameters:
        -----------
        ecg_data : dict
            Dictionary of patient ECG data with patient IDs as keys
        patient_data : DataFrame, optional
            DataFrame with patient metadata
            
        Returns:
        --------
        DataFrame : Extracted features for each patient
        """
        features = pd.DataFrame(index=list(ecg_data.keys()))
        
        if 'hr_metrics' in self.features_to_extract:
            hr_features = self._extract_hr_metrics(ecg_data)
            features = features.join(hr_features)
        
        # Add patient metadata if available
        if patient_data is not None and 'age' in self.features_to_extract:
            if 'age' in patient_data.columns:
                features = features.join(patient_data[['age']])
        
        # Store medians for future transformations
        self.feature_medians = features.median().to_dict()
        
        return features
    
    def _extract_hr_metrics(self, ecg_data):
        """Extract heart rate metrics from lead II"""
        hr_metrics = {
            'median_hr': [], 
            'mean_hr': [], 
            'std_hr': [], 
            'min_hr': [], 
            'max_hr': []
        }
        patient_ids = []
        
        for patient_id, ecg in ecg_data.items():
            patient_ids.append(patient_id)
            try:
                # Extract lead II
                lead_II = ecg.loc[:, 'II'] if isinstance(ecg, pd.DataFrame) else ecg
                
                # Calculate heart rate metrics
                median_hr, mean_hr, std_hr, min_hr, max_hr = self._calculate_heartrate(lead_II, self.fs)
                
                hr_metrics['median_hr'].append(median_hr)
                hr_metrics['mean_hr'].append(mean_hr)
                hr_metrics['std_hr'].append(std_hr)
                hr_metrics['min_hr'].append(min_hr)
                hr_metrics['max_hr'].append(max_hr)
            except Exception as e:
                # Handle errors
                print(f"Error extracting HR metrics for patient {patient_id}: {e}")
                for key in hr_metrics:
                    hr_metrics[key].append(np.nan)
        
        # Create DataFrame
        hr_df = pd.DataFrame(hr_metrics, index=patient_ids)
        return hr_df

    def _calculate_heartrate(self, record, fs):
        """Calculate heart rate from ECG record"""
        # Find R-peaks using neurokit2
        rpeaks = list(nk.ecg_findpeaks(record, sampling_rate=fs).values())[0]
        
        # Remove physiologically impossible R-peaks
        rr_intervals = np.diff(rpeaks) / fs
        valid_rr = (rr_intervals >= 0.2) & (rr_intervals <= 2.0)  
        valid_peaks = rpeaks[1:][valid_rr]
        
        # Calculate HR from RR intervals
        valid_rr_intervals = np.diff(valid_peaks) / fs
        hr = 60 / valid_rr_intervals
        
        return np.median(hr), np.mean(hr), np.std(hr), np.min(hr), np.max(hr)
    
    def save(self, path):
        """Save the feature extractor"""
        extractor_data = {
            'fs': self.fs,
            'features_to_extract': self.features_to_extract,
            'feature_medians': self.feature_medians
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            joblib.dump(extractor_data, f)
    
    @classmethod
    def load(cls, path):
        """Load the feature extractor from a file"""
        with open(path, 'rb') as f:
            extractor_data = joblib.load(f)
        
        instance = cls(
            fs=extractor_data['fs'],
            features_to_extract=extractor_data['features_to_extract']
        )
        instance.feature_medians = extractor_data['feature_medians']
        
        return instance


class CardiacConditionClassifier(BaseEstimator, TransformerMixin):
    """Classify cardiac conditions from diagnosis codes"""
    
    def __init__(self, pathology_hierarchy=None):
        self.pathology_hierarchy = pathology_hierarchy or self._default_hierarchy()
        self.severity_mapping = {
            category: details['severity'] 
            for category, details in self.pathology_hierarchy.items()
        }
        
    def fit(self, X, y=None):
        """Fit method (no fitting needed for this transformer)"""
        return self
        
    def transform(self, X):
        """
        Transform diagnosis codes into cardiac condition categories
        
        Parameters:
        -----------
        X : DataFrame
            DataFrame with 'diagnosis_code' column
            
        Returns:
        --------
        DataFrame : Original DataFrame with added 'cardiac_condition' column
        """
        result = X.copy()
        if 'diagnosis_code' in result.columns:
            result['diagnosis_label'] = result['diagnosis_code'].apply(self._classify)
        return result
    
    def _classify(self, diagnosis_codes):
        """Classify a diagnosis code into a cardiac condition category"""
        # Find matching categories
        matching_categories = [
            category for category, details in self.pathology_hierarchy.items()
            if any(code in diagnosis_codes for code in details['codes'])
        ]
        
        # If multiple matches, select highest severity
        if matching_categories:
            return max(
                matching_categories, 
                key=lambda cat: self.pathology_hierarchy[cat]['severity']
            )
        
        return 'Others'
    
    def _default_hierarchy(self):
        """Return the default cardiac pathology hierarchy"""
        return {
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
                    '164896001',  # Ventricular fibrilation
                    '425856008',  # (Paroxysmal ventricular tachycardia) - critical arrhythmi
                    '81898007',   # (Ventricular escape rhythm) - compensatory arrhythmia
                    '29320008',   # (Ectopic rhythm) - abnormal rhythm origin
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
                    '55827005',   # (Left ventricular hypertrophy)
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
                    '10370003',   # (Artificial pacing) - intervention, not natural condition
                    '106068003',  # (Atrial rhythm) - too general to classify specifically
                ],
                'severity': 1
            },
            'Normal': {
                'codes': ['426783006'],  # Sinus Rhythm
                'severity': 0
            }
        }
    
    def save(self, path):
        """Save the classifier configuration"""
        classifier_data = {
            'pathology_hierarchy': self.pathology_hierarchy,
            'severity_mapping': self.severity_mapping
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            joblib.dump(classifier_data, f)
    
    @classmethod
    def load(cls, path):
        """Load the classifier from a file"""
        with open(path, 'rb') as f:
            classifier_data = joblib.load(f)
        
        instance = cls(pathology_hierarchy=classifier_data['pathology_hierarchy'])
        instance.severity_mapping = classifier_data['severity_mapping']
        
        return instance


class ECGPipeline:
    """Complete ECG analysis pipeline from raw data to processed features"""
    
    def __init__(self):
        self.feature_extractor = None
        self.condition_classifier = None
        
    def setup(self, feature_extractor=None, condition_classifier=None):
        """Setup pipeline components"""
        self.feature_extractor = feature_extractor or ECGFeatureExtractor()
        self.condition_classifier = condition_classifier or CardiacConditionClassifier()
        return self
        
    @timing_decorator
    def process_data(self, ecg_data, patient_data):
        """
        Process raw ECG data and patient metadata
        
        Parameters:
        -----------
        ecg_data : dict
            Dictionary of patient ECG signals with patient IDs as keys
        patient_data : DataFrame
            DataFrame with patient metadata including diagnosis codes
            
        Returns:
        --------
        DataFrame : Processed data with features and cardiac conditions
        """
        # Extract features
        features = self.feature_extractor.transform(ecg_data, patient_data)
        
        # Combine with patient data
        combined_data = patient_data.copy()
        combined_data = combined_data.join(features)
        
        # Classify cardiac conditions if diagnosis codes are present
        if 'diagnosis_code' in combined_data.columns:
            processed_data = self.condition_classifier.transform(combined_data)
        else:
            processed_data = combined_data
            
        return processed_data
    
    def save(self, base_path):
        """Save all pipeline components"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save individual components
        self.feature_extractor.save(os.path.join(base_path, 'feature_extractor.joblib'))
        self.condition_classifier.save(os.path.join(base_path, 'condition_classifier.joblib'))
        
        # Save configuration
        config = {
            'pipeline_version': '1.0',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(base_path, 'pipeline_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def load(cls, base_path):
        """Load pipeline from saved components"""
        instance = cls()
        
        # Load components
        instance.feature_extractor = ECGFeatureExtractor.load(
            os.path.join(base_path, 'feature_extractor.joblib')
        )
        instance.condition_classifier = CardiacConditionClassifier.load(
            os.path.join(base_path, 'condition_classifier.joblib')
        )
        
        return instance