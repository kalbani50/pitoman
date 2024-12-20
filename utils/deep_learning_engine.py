import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timezone

class DeepLearningEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.performance_metrics = {}
        
    async def build_advanced_model(self, model_type: str) -> Model:
        """Builds advanced neural network models"""
        try:
            if model_type == 'price_prediction':
                return self._build_price_prediction_model()
            elif model_type == 'pattern_recognition':
                return self._build_pattern_recognition_model()
            elif model_type == 'sentiment_analysis':
                return self._build_sentiment_analysis_model()
            elif model_type == 'risk_assessment':
                return self._build_risk_assessment_model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logging.error(f"Error building model: {e}")
            raise
            
    def _build_price_prediction_model(self) -> Model:
        """Builds price prediction model with attention mechanism"""
        try:
            input_layer = Input(shape=(self.config['sequence_length'], self.config['n_features']))
            
            # Convolutional layers for feature extraction
            conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
            pool1 = MaxPooling1D(pool_size=2)(conv1)
            
            conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(pool1)
            pool2 = MaxPooling1D(pool_size=2)(conv2)
            
            # LSTM layers with attention
            lstm1 = LSTM(128, return_sequences=True)(pool2)
            attention = tf.keras.layers.Attention()([lstm1, lstm1])
            
            lstm2 = LSTM(64)(attention)
            
            # Dense layers
            dense1 = Dense(64, activation='relu')(lstm2)
            dropout1 = Dropout(0.2)(dense1)
            
            dense2 = Dense(32, activation='relu')(dropout1)
            dropout2 = Dropout(0.1)(dense2)
            
            output_layer = Dense(1)(dropout2)
            
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mse',
                         metrics=['mae', 'mape'])
                         
            return model
            
        except Exception as e:
            logging.error(f"Error building price prediction model: {e}")
            raise
            
    def _build_pattern_recognition_model(self) -> Model:
        """Builds pattern recognition model"""
        try:
            model = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=(self.config['sequence_length'], self.config['n_features'])),
                MaxPooling1D(2),
                Conv1D(128, 3, activation='relu'),
                MaxPooling1D(2),
                Conv1D(256, 3, activation='relu'),
                MaxPooling1D(2),
                LSTM(128, return_sequences=True),
                LSTM(64),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(self.config['n_patterns'], activation='softmax')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
                         
            return model
            
        except Exception as e:
            logging.error(f"Error building pattern recognition model: {e}")
            raise
            
    async def train_models(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Trains all models with provided data"""
        try:
            training_results = {}
            
            for model_type, model_data in data.items():
                # Prepare data
                X_train, y_train = self._prepare_training_data(model_data, model_type)
                
                # Build or load model
                if model_type not in self.models:
                    self.models[model_type] = await self.build_advanced_model(model_type)
                    
                # Train model
                history = self.models[model_type].fit(
                    X_train, y_train,
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    validation_split=0.2,
                    callbacks=self._get_callbacks(model_type)
                )
                
                # Store training history
                self.history[model_type] = history.history
                
                # Evaluate model
                evaluation = self._evaluate_model(model_type, X_train, y_train)
                training_results[model_type] = evaluation
                
            return training_results
            
        except Exception as e:
            logging.error(f"Error training models: {e}")
            raise
            
    async def predict(self, data: pd.DataFrame, model_type: str) -> np.ndarray:
        """Makes predictions using specified model"""
        try:
            # Prepare data
            X = self._prepare_prediction_data(data, model_type)
            
            # Make prediction
            predictions = self.models[model_type].predict(X)
            
            # Post-process predictions
            processed_predictions = self._post_process_predictions(
                predictions,
                model_type
            )
            
            return processed_predictions
            
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise
            
    async def evaluate_performance(self) -> Dict:
        """Evaluates performance of all models"""
        try:
            performance = {}
            
            for model_type, model in self.models.items():
                # Calculate metrics
                metrics = self._calculate_model_metrics(model_type)
                
                # Analyze predictions
                prediction_analysis = self._analyze_predictions(model_type)
                
                # Generate insights
                insights = self._generate_model_insights(
                    metrics,
                    prediction_analysis
                )
                
                performance[model_type] = {
                    'metrics': metrics,
                    'analysis': prediction_analysis,
                    'insights': insights
                }
                
            return performance
            
        except Exception as e:
            logging.error(f"Error evaluating performance: {e}")
            raise
            
    def _get_callbacks(self, model_type: str) -> List:
        """Gets training callbacks"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'models/{model_type}_best.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
    def _prepare_training_data(self,
                             data: pd.DataFrame,
                             model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares data for training"""
        try:
            # Scale data
            if model_type not in self.scalers:
                self.scalers[model_type] = MinMaxScaler()
                
            scaled_data = self.scalers[model_type].fit_transform(data)
            
            # Create sequences
            X, y = self._create_sequences(
                scaled_data,
                self.config['sequence_length']
            )
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error preparing training data: {e}")
            raise
            
    def _create_sequences(self,
                         data: np.ndarray,
                         seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sequences for training"""
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
            
        return np.array(X), np.array(y)
        
    async def save_models(self) -> None:
        """Saves all models"""
        try:
            for model_type, model in self.models.items():
                model.save(f'models/{model_type}_model.h5')
                
        except Exception as e:
            logging.error(f"Error saving models: {e}")
            raise
            
    async def load_models(self) -> None:
        """Loads all models"""
        try:
            for model_type in self.config['model_types']:
                model_path = f'models/{model_type}_model.h5'
                self.models[model_type] = load_model(model_path)
                
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
