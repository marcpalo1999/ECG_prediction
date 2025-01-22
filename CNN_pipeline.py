import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from RF_pipeline import ModelVisualizer
import os
from sklearn.metrics import precision_score, recall_score, f1_score

class ECGClassifier:
    """ECG classification model using CNN architecture.

    Args:
        input_shape (tuple): Shape of input ECG signals (samples, leads, timepoints)
        encode_dict (dict): Dictionary mapping class indices to class names
    """
    def __init__(self, input_shape, encode_dict):
        if len(input_shape) != 2:
            raise ValueError("Input shape must be 3D (samples, leads, timepoints)")
            
        self.input_shape = input_shape
        self.n_classes = len(encode_dict)
        self.encode_dict = encode_dict
        self.model = self.build_model()

    def normalize_signals(self, signals):
        """Normalize ECG signals per lead."""
        try:
            normalized = np.zeros_like(signals)
            for i in range(signals.shape[1]):
                mean = np.mean(signals[:, i, :])
                std = np.std(signals[:, i, :])
                normalized[:, i, :] = (signals[:, i, :] - mean) / (std + 1e-8)
            return normalized
        except Exception as e:
            raise RuntimeError(f"Error normalizing signals: {str(e)}")

    def build_model(self):
        """Build CNN model architecture."""
        inputs = layers.Input(shape=(12, 5000))

        # Reshape to make Conv1D work
        x = layers.Reshape((5000, 12))(inputs)

        x = layers.Conv1D(filters=32, kernel_size=25, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(x)

        # Second conv block
        x = layers.Conv1D(64, kernel_size=7)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(x)

        # Third conv block
        x = layers.Conv1D(128, kernel_size=5)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(x)

        # Dense layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
   
    def train(self, X_train, y_train, X_val, y_val, batch_size=128, epochs=50, patience=10):
        """Train the model with early stopping and checkpointing."""
        try:
            X_train = self.normalize_signals(X_train)
            X_val = self.normalize_signals(X_val)
            
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.n_classes)
            y_val = tf.keras.utils.to_categorical(y_val, num_classes=self.n_classes)
            
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    mode='min'
                ),
                callbacks.ModelCheckpoint(
                    filepath='./Results/final_CNN_model/best_model.keras',
                    monitor='val_loss',
                    save_best_only=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Data generator for memory efficiency
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
                .shuffle(len(X_train))\
                .batch(batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
                .batch(batch_size)
            
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=1
            )
            return history
            
        except Exception as e:
            raise RuntimeError(f"Error during training: {str(e)}")

    def predict(self, X):
        """Generate predictions for input data."""
        try:
            X = self.normalize_signals(X)
            return self.model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")

    def save_model(self, filepath):
        """Save model weights to file."""
        try:
            self.model.save_weights(filepath)
        except Exception as e:
            raise RuntimeError(f"Error saving model: {str(e)}")
        
    def load_model(self, filepath):
        """Load model weights from file."""
        try:
            self.model.load_weights(filepath)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance and generate visualizations."""
        try:
            X_test = self.normalize_signals(X_test)
            
            y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=self.n_classes)
            y_pred = self.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test_onehot, axis=1)
            
            metrics = {
                'accuracy': np.mean(y_pred_classes == y_test_classes),
                'precision': precision_score(y_test_classes, y_pred_classes, average='macro'),
                'recall': recall_score(y_test_classes, y_pred_classes, average='macro'),
                'f1': f1_score(y_test_classes, y_pred_classes, average='macro')
            }
            
            visualizer = ModelVisualizer()
            
            # Generate evaluation plots
            visualizer.plot_multiclass_confusion_matrix(
                y_test_classes, 
                y_pred_classes,
                class_names=list(self.encode_dict.values())
            )
            
            roc_auc = visualizer.plot_multiclass_roc(
                y_test_classes,
                y_pred, 
                classes=list(self.encode_dict.values())
            )
            
            # visualizer.plot_cv_results({
            #     'test_accuracy': [metrics['accuracy']],
            #     'test_precision': [metrics['precision']], 
            #     'test_recall': [metrics['recall']],
            #     'test_f1': [metrics['f1']]
            # })
            
            metrics['roc_auc'] = roc_auc
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Error during evaluation: {str(e)}")