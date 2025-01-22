import os
import wfdb
import scipy.io as sio
import pandas as pd
import numpy as np
from scipy import signal
import scipy
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import welch
import multiprocessing
import dask.dataframe as dd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
from multiprocessing import Pool, cpu_count
from functools import partial

from RF_pipeline import timing_decorator


n_workers = max(1, multiprocessing.cpu_count() - 1)


class ECGProcessor:
    """Class for ECG signal processing and data management."""
    
    def __init__(self, dataset_dir, output_dir, fs=500):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.fs = fs
        self.leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        
        # Create output directories
        self.processed_dir = self.output_dir / 'processed_data'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load disease mapping
        self.disease_map = self._load_disease_map()
    
    def _load_disease_map(self):
        """Load SNOMED-CT disease mapping."""
        map_path = self.dataset_dir.parent / 'ConditionNames_SNOMED-CT.csv'
        return pd.read_csv(map_path).set_index('Snomed_CT')
    
    def _apply_filters(self, signal_data):
        """Apply signal processing filters."""
        # Detrend
        detrended = signal.detrend(signal_data)
        
        # High-pass filter (5 Hz)
        b, a = signal.butter(5, 5/(self.fs/2), btype='high')
        highpassed = signal.filtfilt(b, a, detrended)
        
        # Low-pass filter (140 Hz)
        b, a = signal.butter(5, 140/(self.fs/2), btype='low')
        lowpassed = signal.filtfilt(b, a, highpassed)
        
        # Notch filter (50 Hz)
        notched = scipy.signal.filtfilt(
            *scipy.signal.iirnotch(50 / (self.fs/2), 30),
            lowpassed
        )
        
        return notched
    
    def plot_signal_quality(self, signal, filtered_signal, patient_id):
        """Plot original vs filtered signal in time and frequency domain."""
        fig, axes = plt.subplots(12, 2, figsize=(20, 12*4))
        
        for i, lead in enumerate(self.leads):
            # Time domain
            axes[i,0].plot(signal[lead], 'gray', label='Original', alpha=0.5)
            axes[i,0].plot(filtered_signal[lead], 'black', label='Filtered', linewidth=0.6)
            axes[i,0].set_ylim((-1500,1500))
            axes[i,0].set_xticks(np.arange(0,5001,500))
            axes[i,0].set_xticklabels(np.arange(0,5001,500)/self.fs)
            axes[i,0].grid(True)
            axes[i,0].set_title(f'Lead {lead}')
            if i == 0:
                axes[i,0].legend()
            
            # Frequency domain
            f_orig, psd_orig = welch(signal[lead], self.fs, nperseg=1024)
            f_filt, psd_filt = welch(filtered_signal[lead], self.fs, nperseg=1024)
            
            axes[i,1].semilogy(f_orig, psd_orig, 'gray', label='Original', alpha=0.5)
            axes[i,1].semilogy(f_filt, psd_filt, 'black', label='Filtered')
            axes[i,1].set_title(f'Lead {lead} - PSD')
            axes[i,1].set_xlabel('Frequency (Hz)')
            if i == 0:
                axes[i,1].legend()
        
        plt.suptitle(f'Signal Quality Analysis - Patient {patient_id}')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.processed_dir / f'quality_check_{patient_id}.png'
        plt.savefig(plot_path)
        plt.close()
    
    def process_data(self, max_files=None):
        """Process all ECG files in dataset directory."""
        processed_count = 0
        ecg_data = {}
        patient_data = {}
        
        for dir1 in sorted(d for d in self.dataset_dir.iterdir() if d.is_dir()):
            for dir2 in sorted(d for d in dir1.iterdir() if d.is_dir()):
                for record_path in dir2.glob('*.hea'):
                    if max_files and processed_count >= max_files:
                        break
                        
                    try:
                        patient_id = record_path.stem
                        print(f"Processing {patient_id}...")
                        
                        # Read header
                        record = wfdb.rdheader(str(record_path.with_suffix('')))
                        metadata = self._extract_metadata(record)
                        
                        # Read and process ECG signals
                        mat_data = sio.loadmat(str(record_path.with_suffix('.mat')))
                        signals = pd.DataFrame(mat_data['val'].T, columns=self.leads)
                        filtered_signals = signals.apply(self._apply_filters)
                        
                        # Plot quality check
                        if processed_count % 10 == 0:  # Plot every 10th patient
                            self.plot_signal_quality(signals, filtered_signals, patient_id)
                        
                        # Store processed data
                        ecg_data[patient_id] = {
                            'ecg_signals': signals,
                            'ecg_signals_filtered': filtered_signals
                        }
                        patient_data[patient_id] = metadata
                        
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"Error processing {patient_id}: {str(e)}")
        
        # Save processed data
        self._save_processed_data(ecg_data, pd.DataFrame.from_dict(patient_data, orient='index'))
        return ecg_data, patient_data
    
    def _extract_metadata(self, record):
        """Extract metadata from WFDB header."""
        metadata = {
            'age': None,
            'sex': None,
            'diagnosis_code': None,
            'diagnosis_name': None
        }
        
        for comment in record.comments:
            if comment.startswith("Age:"):
                metadata['age'] = comment.split(":")[1].strip()
            elif comment.startswith("Sex:"):
                metadata['sex'] = comment.split(":")[1].strip()
            elif comment.startswith("Dx:"):
                diagnosis = comment.split(":")[1].strip()
                metadata['diagnosis_code'] = [code for code in diagnosis.split(',')]
                metadata['diagnosis_name'] = self._map_diagnosis(diagnosis)
        
        return metadata
    
    def _map_diagnosis(self, diagnosis):
        """Map SNOMED-CT codes to diagnosis names."""
        try:
            return [self.disease_map.loc[int(code), "Full Name"] 
                   for code in diagnosis.split(',')]
        except:
            return None
    
    def _save_processed_data(self, ecg_data, patient_df):
        """Save processed data to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.processed_dir / f"processed_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save patient metadata
        patient_df.to_csv(save_dir / 'patient_metadata.csv')
        
        # Save ECG signals
        for patient_id, data in ecg_data.items():
            patient_dir = save_dir / patient_id
            patient_dir.mkdir(exist_ok=True)
            
            data['ecg_signals'].to_csv(patient_dir / 'raw_signals.csv')
            data['ecg_signals_filtered'].to_csv(patient_dir / 'filtered_signals.csv')
        
        print(f"Processed data saved to: {save_dir}")

from multiprocessing import Pool, cpu_count
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_single_patient(patient_dir):
    """
    Load data for a single patient directory.
    
    Args:
        patient_dir (Path): Directory containing patient data
        
    Returns:
        tuple: (patient_id, dict with filtered signals) or None if error
    """
    try:
        patient_id = patient_dir.name
        filtered_signals = pd.read_csv(patient_dir / 'filtered_signals.csv', index_col=0)
        return patient_id, {
            'ecg_signals_filtered': filtered_signals
        }
    except Exception as e:
        print(f"Error loading {patient_id}: {str(e)}")
        return None

@timing_decorator
def load_processed_data(base_dir, indices = None, n_jobs=None):
    """
    Load processed ECG data from the most recent processing run.
    
    Args:
        base_dir (str): Base directory containing processed data
        n_jobs (int, optional): Number of CPUs to use. If None, uses all available CPUs.
        
    Returns:
        tuple: (ecg_data dict, patient_metadata DataFrame)
    """
    # Get number of CPUs to use
    n_cpus = n_jobs if n_jobs is not None else cpu_count()-1
    print(f"Using {n_cpus} CPUs for parallel processing")
    
    data_dir = Path(base_dir)
    
    # Find most recent processed directory
    processed_dirs = sorted(list(data_dir.glob("processed*")), reverse=True)
    if not processed_dirs:
        raise FileNotFoundError(f"No processed data directories found in {base_dir}")
    
    latest_dir = processed_dirs[0]
    print(f"Loading data from: {latest_dir}")
    
    # Load patient metadata
    metadata_file = latest_dir / 'patient_metadata.csv'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found in {latest_dir}")
    
    patient_df = pd.read_csv(metadata_file, index_col=0)

    try:
        patient_df = patient_df[patient_df.index.isin(indices)]
    except:
        pass
    
    # Get list of patient directories
    if indices is None:
        patient_dirs = [d for d in latest_dir.glob("*") if d.is_dir()]
    else:
        patient_dirs = [d for d in latest_dir.glob("*") if (d.is_dir()) and (d.name in indices) ]

    total_patients = len(patient_dirs)
    
    # Parallel loading with progress bar
    with Pool(processes=n_cpus) as pool:
        results = list(tqdm(
            pool.imap(load_single_patient, patient_dirs),
            total=total_patients,
            desc="Loading patient data",
            unit="patient"
        ))
    
    # Filter out None results and convert to dictionary
    ecg_data = {}
    for result in results:
        if result is not None:
            patient_id, data = result
            ecg_data[patient_id] = data
    
    print(f"Successfully loaded {len(ecg_data)} out of {total_patients} patients from {latest_dir}")
    return ecg_data, patient_df

def main():
    # Define paths
    dataset_dir = '../a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords'
    output_dir = './Results'
    
    # Initialize processor
    processor = ECGProcessor(dataset_dir, output_dir)
    
    # Process data
    print("Processing ECG data...")
    ecg_data, patient_data = processor.process_data()
    
    print("Processing complete!")

if __name__ == "__main__":
    main()