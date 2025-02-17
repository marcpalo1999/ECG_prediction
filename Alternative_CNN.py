import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from datetime import datetime
import traceback
from pathlib import Path

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

class ECGNet(nn.Module):
    def __init__(self, input_length, num_classes=6):
        super(ECGNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=50, stride=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(50)
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 50, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# def explain_ecg(model, example_data, lead_names=None):
#     if lead_names is None:
#         lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
#     device = next(model.parameters()).device
#     model.eval()
    
#     if len(example_data.shape) == 2:
#         example_data = example_data[None, :, :]
#     example_tensor = torch.FloatTensor(example_data).to(device)
    
#     example_tensor.requires_grad_()
#     output = model(example_tensor)
#     output_class = output.argmax(1).item()
    
#     model.zero_grad()
#     output[0, output_class].backward()
#     gradients = example_tensor.grad.abs().cpu().numpy()[0]
    
#     plt.figure(figsize=(20, 15))
    
#     for i in range(12):
#         plt.subplot(4, 3, i+1)
#         signal = example_data[0, i]
#         importance = gradients[i]
#         time = np.arange(len(signal))
#         plt.plot(time, signal, 'gray', alpha=0.5)
#         plt.scatter(time, signal, c=importance, cmap='RdBu_r', s=1, alpha=0.5)
#         mean_imp = np.mean(importance)
#         plt.title(f'{lead_names[i]} (Imp: {mean_imp:.3f})')
#         if i % 3 == 0:
#             plt.ylabel('Amplitude')
#         if i >= 9:
#             plt.xlabel('Time')
    
#     plt.tight_layout()
#     plt.show()
    
#     lead_importance = pd.DataFrame({
#         'Lead': lead_names,
#         'Importance': [np.mean(gradients[i]) for i in range(12)]
#     }).sort_values('Importance', ascending=False)
    
#     return lead_importance

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, model_save_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_save_path = model_save_path
        self.history = defaultdict(list)
        
    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        predictions = []
        true_labels = []
        
        for signals, labels in train_loader:
            signals, labels = signals.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
        return train_loss / len(train_loader), predictions, true_labels
    
    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())
        
        return val_loss / len(val_loader), predictions, true_labels, probabilities
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss, train_preds, train_true = self.train_epoch(train_loader)
            val_loss, val_preds, val_true, val_probs = self.evaluate(val_loader)
            
            train_acc = np.mean(np.array(train_preds) == np.array(train_true))
            val_acc = np.mean(np.array(val_preds) == np.array(val_true))
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, self.model_save_path)
        
        return self.history

# def prepare_data(data_dict, labels):
#     signals_list = []
#     labels_list = []
#     indices = []
    
#     if isinstance(labels, pd.DataFrame):
#         labels_dict = dict(zip(labels['index'], labels['label']))
#     else:
#         labels_dict = labels
    
#     for idx in data_dict:
#         if idx in labels_dict:
#             filtered_signals = data_dict[idx]['ecg_signals_filtered']
#             patient_signals = filtered_signals.to_numpy().T
            
#             signals_list.append(patient_signals)
#             labels_list.append(labels_dict[idx])
#             indices.append(idx)
    
#     signals = np.array(signals_list)
#     labels = np.array(labels_list)
    
#     return signals, labels, indices

def normalize_signals(signals):
    normalized = np.zeros_like(signals)
    for i in range(signals.shape[1]):
        mean = np.mean(signals[:, i, :])
        std = np.std(signals[:, i, :])
        normalized[:, i, :] = (signals[:, i, :] - mean) / (std + 1e-8)
    return normalized

def plot_metrics(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# def plot_roc_pr_curves(true_labels, pred_probs, save_path=None):
#     fpr, tpr, _ = roc_curve(true_labels, pred_probs)
#     roc_auc = auc(fpr, tpr)
    
#     precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
#     pr_auc = average_precision_score(true_labels, pred_probs)
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
#     ax1.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     ax1.plot([0, 1], [0, 1], 'k--')
#     ax1.set_xlabel('False Positive Rate')
#     ax1.set_ylabel('True Positive Rate')
#     ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
#     ax1.legend()
    
#     ax2.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
#     ax2.set_xlabel('Recall')
#     ax2.set_ylabel('Precision')
#     ax2.set_title('Precision-Recall Curve')
#     ax2.legend()
    
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()
    
#     return roc_auc, pr_auc

# def plot_confusion_matrix(true_labels, predictions, save_path=None):
#     cm = confusion_matrix(true_labels, predictions)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()


from src.RF_pipeline import ModelVisualizer
def main(signals, labels, config, label_mapping, load_model_path=None):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"Results/DL_model/results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        #Normalise signals        
        signals = normalize_signals(signals)

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            signals, labels, test_size=0.2, random_state=config['seed'], stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=config['seed'], stratify=y_temp
        )
        
        # Create datasets and loaders
        train_dataset = ECGDataset(X_train, y_train)
        val_dataset = ECGDataset(X_val, y_val)
        test_dataset = ECGDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ECGNet(signals.shape[2]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Initialize RF_pipeline visualizer
        visualizer = ModelVisualizer()
        
        # Load or train model
        if load_model_path is not None:
            print(f"Loading pre-trained model from {load_model_path}")
            checkpoint = torch.load(load_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            history = None
        else:
            print("Training new model...")
            trainer = ModelTrainer(
                model, criterion, optimizer, device,
                model_save_path=os.path.join(output_dir, 'best_model.pth')
            )
            history = trainer.train(train_loader, val_loader, config['num_epochs'])
            plot_metrics(history, save_path=os.path.join(output_dir, 'training_metrics.png'))
            
            # Load best model
            checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Model evaluation
        model.eval()
        trainer = ModelTrainer(model, criterion, optimizer, device, 
                             model_save_path=os.path.join(output_dir, 'best_model.pth'))
        test_loss, test_preds, test_true, test_probs = trainer.evaluate(test_loader)
        
        # Calculate and save metrics using RF_pipeline functions
        print("\nTest Set Performance:")
        
        # Plot confusion matrix using RF_pipeline
        visualizer.plot_multiclass_confusion_matrix(
            test_true, 
            test_preds, 
            class_names=list(label_mapping.keys()),
            normalise=1
        )
        
        # Plot ROC curve using RF_pipeline
        roc_auc = visualizer.plot_multiclass_roc(
            test_true,
            np.column_stack((1-np.array(test_probs), test_probs)),
            classes=list(label_mapping.keys())
        )

        # Save test set predictions
        test_results = pd.DataFrame({
            'true_label': test_true,
            'predicted_label': test_preds,
            'abnormal_probability': test_probs
        })
        test_results.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

        return {
            'model': model,
            'history': history,
            'test_metrics': {
                'roc_auc': roc_auc,
                'test_loss': test_loss
            },
            'output_dir': output_dir,
            'X_test': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()
        return None
