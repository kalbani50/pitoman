import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime, timezone

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out.transpose(0, 1)
        
        # Take the last output
        last_output = combined[:, -1, :]
        
        # Fully connected layers
        out = self.fc(last_output)
        return out

class DeepLearningPredictor:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
    def prepare_data(self,
                    df: pd.DataFrame,
                    sequence_length: int,
                    prediction_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares data for training/prediction"""
        try:
            # Create features
            features = self._create_features(df)
            
            # Scale features
            scaled_features = self.feature_scaler.fit_transform(features)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_features) - sequence_length - prediction_length + 1):
                X.append(scaled_features[i:(i + sequence_length)])
                y.append(features.iloc[i + sequence_length + prediction_length - 1]['close'])
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}", exc_info=True)
            return np.array([]), np.array([])
            
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates advanced features for the model"""
        try:
            features = pd.DataFrame()
            
            # Price features
            features['close'] = df['close']
            features['high'] = df['high']
            features['low'] = df['low']
            features['volume'] = df['volume']
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(df['close'])
            features['macd'] = self._calculate_macd(df['close'])
            features['bollinger'] = self._calculate_bollinger(df['close'])
            
            # Price changes
            features['price_change'] = df['close'].pct_change()
            features['volume_change'] = df['volume'].pct_change()
            
            # Volatility
            features['volatility'] = df['close'].rolling(window=20).std()
            
            # Remove NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logging.error(f"Error creating features: {e}", exc_info=True)
            return pd.DataFrame()
            
    async def train_model(self,
                         train_data: Tuple[np.ndarray, np.ndarray],
                         val_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Trains the deep learning model"""
        try:
            # Create datasets
            train_dataset = TimeSeriesDataset(train_data[0], train_data[1])
            val_dataset = TimeSeriesDataset(val_data[0], val_data[1])
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size']
            )
            
            # Initialize model
            self.model = LSTMPredictor(
                input_dim=train_data[0].shape[2],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            # Initialize optimizer and loss function
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate']
            )
            criterion = nn.MSELoss()
            
            # Training loop
            best_val_loss = float('inf')
            early_stopping_counter = 0
            training_history = []
            
            for epoch in range(self.config['epochs']):
                # Training
                self.model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                # Validation
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        y_pred = self.model(X_batch)
                        val_loss += criterion(
                            y_pred,
                            y_batch.unsqueeze(1)
                        ).item()
                        
                # Calculate average losses
                train_loss = train_loss / len(train_loader)
                val_loss = val_loss / len(val_loader)
                
                # Store training history
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    early_stopping_counter += 1
                    
                if early_stopping_counter >= self.config['patience']:
                    break
                    
            # Load best model
            self.model.load_state_dict(torch.load('best_model.pth'))
            
            return {
                'training_history': training_history,
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1
            }
            
        except Exception as e:
            logging.error(f"Error training model: {e}", exc_info=True)
            return {}
            
    async def predict(self, X: np.ndarray) -> Dict:
        """Makes predictions using the trained model"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")
                
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                predictions = self.model(X_tensor)
                
            # Calculate prediction intervals
            predictions_np = predictions.cpu().numpy()
            
            return {
                'predictions': predictions_np,
                'confidence_intervals': self._calculate_confidence_intervals(predictions_np),
                'prediction_quality': self._assess_prediction_quality(predictions_np)
            }
            
        except Exception as e:
            logging.error(f"Error making predictions: {e}", exc_info=True)
            return {}
            
    def _calculate_confidence_intervals(self,
                                     predictions: np.ndarray,
                                     confidence: float = 0.95) -> Dict:
        """Calculates confidence intervals for predictions"""
        try:
            z_score = 1.96  # 95% confidence interval
            
            std_dev = np.std(predictions)
            mean_pred = np.mean(predictions)
            
            lower_bound = mean_pred - z_score * std_dev
            upper_bound = mean_pred + z_score * std_dev
            
            return {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence_level': confidence
            }
            
        except Exception as e:
            logging.error(f"Error calculating confidence intervals: {e}", exc_info=True)
            return {}
            
    def _assess_prediction_quality(self, predictions: np.ndarray) -> Dict:
        """Assesses the quality of predictions"""
        try:
            # Calculate prediction statistics
            volatility = np.std(predictions)
            trend_strength = np.abs(np.mean(np.diff(predictions)))
            
            # Calculate prediction confidence
            confidence = 1 / (1 + volatility)
            
            # Determine prediction reliability
            if confidence > 0.8:
                reliability = 'high'
            elif confidence > 0.6:
                reliability = 'medium'
            else:
                reliability = 'low'
                
            return {
                'confidence': float(confidence),
                'reliability': reliability,
                'volatility': float(volatility),
                'trend_strength': float(trend_strength)
            }
            
        except Exception as e:
            logging.error(f"Error assessing prediction quality: {e}", exc_info=True)
            return {}
