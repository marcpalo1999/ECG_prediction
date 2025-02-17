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
import wfdb
import h5py
from tqdm import tqdm


import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
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
        """Process ECG files and save directly to HDF5."""
        processed_count = 0
        patient_data = {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.processed_dir / f"processed_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(save_dir / 'ecg_signals.h5', 'w') as f:
            filtered_group = f.create_group('filtered_signals')
            
            for dir1 in sorted(d for d in self.dataset_dir.iterdir() if d.is_dir()):
                for dir2 in sorted(d for d in dir1.iterdir() if d.is_dir()):
                    for record_path in dir2.glob('*.hea'):
                        if max_files and processed_count >= max_files:
                            break
                            
                        try:
                            patient_id = record_path.stem
                            print(f"Processing {patient_id}...")
                            
                            record = wfdb.rdheader(str(record_path.with_suffix('')))
                            metadata = self._extract_metadata(record)
                            
                            mat_data = sio.loadmat(str(record_path.with_suffix('.mat')))
                            signals = pd.DataFrame(mat_data['val'].T, columns=self.leads)
                            filtered_signals = signals.apply(self._apply_filters)
                            
                            filtered_group.create_dataset(
                                patient_id, 
                                data=filtered_signals.values,
                                compression="gzip",
                                compression_opts=9
                            )
                            
                            patient_data[patient_id] = metadata
                            processed_count += 1
                            
                            if processed_count % 10 == 0:
                                self.plot_signal_quality(signals, filtered_signals, patient_id)
                                
                        except Exception as e:
                            print(f"Error processing {patient_id}: {str(e)}")
        
        pd.DataFrame.from_dict(patient_data, orient='index').to_csv(save_dir / 'patient_metadata.csv')
        return save_dir
    
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
    


# from multiprocessing import Pool, cpu_count
# from pathlib import Path
# import pandas as pd
# from tqdm import tqdm

# def load_single_patient(patient_dir):
#     """
#     Load data for a single patient directory.
    
#     Args:
#         patient_dir (Path): Directory containing patient data
        
#     Returns:
#         tuple: (patient_id, dict with filtered signals) or None if error
#     """
#     try:
#         patient_id = patient_dir.name
#         filtered_signals = pd.read_csv(patient_dir / 'filtered_signals.csv', index_col=0)
#         return patient_id, {
#             'ecg_signals_filtered': filtered_signals
#         }
#     except Exception as e:
#         print(f"Error loading {patient_id}: {str(e)}")
#         return None

@timing_decorator
def load_processed_data(base_dir, indices=None, n_jobs=None):
    """Load processed ECG data."""
    data_dir = Path(base_dir)
    processed_dirs = sorted(list(data_dir.glob("processed*")), reverse=True)
    latest_dir = processed_dirs[0]
    
    patient_df = pd.read_csv(latest_dir / 'patient_metadata.csv', index_col=0)
    if indices is not None:
        patient_df = patient_df[patient_df.index.isin(indices)]
    
    ecg_data = {}
    leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    
    with h5py.File(latest_dir / 'ecg_signals.h5', 'r') as f:
        patient_ids = patient_df.index if indices is not None else list(f['filtered_signals'].keys())
        for patient_id in tqdm(patient_ids, desc="Loading patient data"):
            try:
                ecg_data[patient_id] = {
                    'ecg_signals_filtered': pd.DataFrame(
                        f['filtered_signals'][patient_id][()],
                        columns=leads
                    )
                }
            except Exception as e:
                print(f"Error loading {patient_id}: {str(e)}")
    
    return ecg_data, patient_df

def main():
    # Define paths
    dataset_dir = '/Users/marcpalomer/Documents/Personal/ECG_prediction/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords'
    output_dir = './Results'
    
    # Initialize processor
    processor = ECGProcessor(dataset_dir, output_dir)
    
    # Process data
    print("Processing ECG data...")
    save_dir = processor.process_data()
    
    print("Processing complete!")

if __name__ == "__main__":
    main()