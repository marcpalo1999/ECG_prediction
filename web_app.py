import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import h5py
from src.data_processing import load_processed_data
from ecg_data_pipeline import ECGFeatureExtractor, CardiacConditionClassifier
from ecg_model_pipeline import ECGModelTrainer

# Load the model and necessary components
@st.cache_resource
def load_components():
    model_path = './Results/RF_model/ecg_rf_model.joblib'
    model_trainer = ECGModelTrainer.load(model_path)
    condition_classifier = CardiacConditionClassifier()
    feature_extractor = ECGFeatureExtractor(fs=500)
    return model_trainer, condition_classifier, feature_extractor

# Load patient metadata only (not ECG data)
@st.cache_data
def load_patient_metadata():
    data_path = './Results/processed_data'
    processed_dirs = sorted(list(os.path.join(data_path, d) for d in os.listdir(data_path) if d.startswith('processed')), reverse=True)
    latest_dir = processed_dirs[0]
    patient_df = pd.read_csv(os.path.join(latest_dir, 'patient_metadata.csv'), index_col=0)
    return patient_df, latest_dir

# Load single patient ECG data
def load_single_patient_ecg(patient_id, data_dir):
    leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    try:
        # Load filtered ECG data
        with h5py.File(os.path.join(data_dir, 'ecg_signals.h5'), 'r') as f:
            if patient_id in f['filtered_signals']:
                filtered_ecg = pd.DataFrame(
                    f['filtered_signals'][patient_id][()],
                    columns=leads
                )
                
        # Load raw ECG data if available
        try:
            with h5py.File(os.path.join(data_dir, 'raw_ecg_signals.h5'), 'r') as f:
                if patient_id in f:
                    raw_ecg = pd.DataFrame(
                        f[patient_id][()],
                        columns=leads
                    )
                else:
                    raw_ecg = filtered_ecg.copy()  # Use filtered as fallback
        except:
            raw_ecg = filtered_ecg.copy()  # Use filtered as fallback
                
        return {'raw': raw_ecg, 'filtered': filtered_ecg}
    except Exception as e:
        st.error(f"Error loading ECG data: {str(e)}")
        return None

def main():
    st.title("ECG Cardiac Condition Classifier")
    
    # Load components
    model_trainer, condition_classifier, feature_extractor = load_components()
    
    # Load only patient metadata (not ECG data)
    patient_metadata, data_dir = load_patient_metadata()
    
    # Patient selection
    patient_ids = list(patient_metadata.index)
    selected_patient = st.selectbox("Select Patient ID", patient_ids)
    
    if selected_patient:
        # Load ECG data for selected patient only
        with st.spinner("Loading ECG data..."):
            ecg_data = load_single_patient_ecg(selected_patient, data_dir)
            
        if ecg_data is not None:
            # Display patient metadata
            st.subheader("Patient Information")
            patient_info = patient_metadata.loc[selected_patient]
            st.write(f"Age: {patient_info['age']}")
            st.write(f"Sex: {patient_info['sex']}")
            
            # Display only Lead II (raw and filtered)
            st.subheader("Lead II ECG")
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))
            
            # Plot raw Lead II
            axes[0].plot(ecg_data['raw']['II'], color='black')
            axes[0].set_title("Lead II (Raw)")
            axes[0].grid(True)
            
            # Plot filtered Lead II
            axes[1].plot(ecg_data['filtered']['II'], color='blue')
            axes[1].set_title("Lead II (Filtered)")
            axes[1].grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Generate prediction
            if st.button("Generate Prediction"):
                # Extract features for this patient only
                patient_df = patient_metadata[patient_metadata.index == selected_patient]
                features = feature_extractor.transform({selected_patient: ecg_data['filtered']}, patient_df)
                
                # Make prediction
                feature_names = ['median_hr', 'mean_hr', 'std_hr', 'min_hr', 'max_hr', 'age']
                X = features[feature_names]

                X_processed = model_trainer._preprocess_features(X)
                prediction_idx = model_trainer.predict(X)[0]
                probabilities = model_trainer.predict_proba(X)
                
                # Map prediction to class name
                inv_mapping = {v: k for k, v in model_trainer.label_mapping.items()}
                predicted_class = inv_mapping[prediction_idx]
                predicted_class_prob = probabilities[0][prediction_idx]
                
                # Display results
                st.success(f"Predicted Cardiac Condition: {predicted_class} (Probability: {predicted_class_prob:.3f})")
                
                # Get severity levels for all cardiac conditions
                severity_mapping = {condition: details['severity'] 
                                  for condition, details in condition_classifier.pathology_hierarchy.items()}
                
                # Show probabilities for each class, ordered by severity
                st.subheader("Prediction Probabilities")
                probs_df = pd.DataFrame({
                    'Cardiac Condition': [inv_mapping[i] for i in range(len(probabilities[0]))],
                    'Probability': probabilities[0],
                    'Severity': [severity_mapping.get(inv_mapping[i], 0) for i in range(len(probabilities[0]))]
                }).sort_values('Severity', ascending=True)
                
                # Custom bar chart with ordered x-axis
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(probs_df['Cardiac Condition'], probs_df['Probability'], color='blue')
                ax.set_xticklabels(probs_df['Cardiac Condition'], rotation=45, ha='right')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add severity info for the top prediction
                top_severity = severity_mapping[predicted_class]
                severity_color = "green" if top_severity < 3 else "orange" if top_severity < 5 else "red"
                st.markdown(f"<h3 style='color:{severity_color}'>Severity Level: {top_severity}/7</h3>", unsafe_allow_html=True)


                st.subheader(f"Model Explainability for {predicted_class}")

                with st.spinner("Generating SHAP explanations..."):
                    
                    shap_values = model_trainer.explainer.shap_values(X_processed)
                    
                    # Transpose to get format needed for waterfall plot - select class of interest
                    class_shap_values = shap_values[0, :, prediction_idx]
                    
                    # Create waterfall plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots._waterfall.waterfall_legacy(
                        model_trainer.explainer.expected_value[prediction_idx],
                        class_shap_values,
                        feature_names=feature_names,
                        show=False
                    )
                    plt.title(f"SHAP Explanation for {predicted_class}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    

        else:
            st.error(f"ECG data for patient {selected_patient} not found.")

if __name__ == "__main__":
    main()