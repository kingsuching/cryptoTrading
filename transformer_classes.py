import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from CONSTANTS import FILL
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from functions import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (mask should be boolean: True=valid, False=masked)
        if mask is not None:
            # Expand mask to match attention scores dimensions
            if len(mask.shape) == 2:  # (batch_size, seq_length)
                mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_length)
                mask = mask.expand(-1, -1, attn_scores.size(-2), -1)

            # Apply mask: False positions get -1e9
            attn_scores = attn_scores.masked_fill(~mask, self.mask_value)  # Note the ~ (NOT operator)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        output = self.W_o(attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BaseTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_length=5000, dropout=0.1, output_dim=1,
                 learning_rate=1e-4, batch_size=32, mask_value=0.0):
        super(BaseTransformer, self).__init__()

        self.d_model = d_model
        self.mask_value = mask_value  # Value to mask (0.0 for padding)
        self.input_projection = None
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.transformer_encoder = TransformerEncoder(
            num_layers, d_model, num_heads, d_ff, dropout
        )
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.history = {'loss': []}

    def create_padding_mask(self, x):
        """Create mask for padded positions (True = valid, False = padded)"""
        # x shape: (batch_size, seq_length) or (batch_size, seq_length, features)
        if len(x.shape) == 3:
            # For 3D input, check if entire feature vector is zeros
            mask = (x != self.mask_value).any(dim=-1)  # (batch_size, seq_length)
        else:
            # For 2D input
            mask = (x != self.mask_value)  # (batch_size, seq_length)

        return mask

    def _create_input_projection(self, input_dim):
        """Create input projection layer based on data dimensions"""
        if self.input_projection is None:
            self.input_projection = nn.Linear(input_dim, self.d_model).to(self.device)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_length, input_dim) or (batch_size, seq_length)

        # Handle 2D input (batch_size, seq_length) by adding feature dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # (batch_size, seq_length, 1)

        # Create input projection if not exists
        if self.input_projection is None:
            self._create_input_projection(x.shape[-1])

        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x, mask)

        # Global average pooling
        x = x.mean(dim=1)
        output = self.output_projection(x)

        return output

    def _prepare_data(self, X, y):
        """Convert DataFrame with lists and Series to proper tensor format"""
        # Handle pandas DataFrame with lists in a single column
        if isinstance(X, pd.DataFrame):
            # Get the column name (assuming single column with lists)
            column_name = X.columns[0]

            # Extract the lists from the DataFrame column
            list_data = X[column_name].tolist()

            # Convert lists to numpy arrays and stack them
            # Each list becomes a sequence (row in the final array)
            sequences = []
            for seq in list_data:
                if isinstance(seq, list):
                    sequences.append(np.array(seq, dtype=np.float32))
                else:
                    # Handle case where it's already an array
                    sequences.append(np.array(seq, dtype=np.float32))

            # Stack all sequences into a 2D array: (num_samples, seq_length)
            X_array = np.stack(sequences)

            if len(X_array.shape) == 2:
                # X_array is (batch_size, seq_length) - add feature dimension
                X_array = X_array.reshape(X_array.shape[0], X_array.shape[1], 1)

        elif isinstance(X, pd.Series):
            # Handle Series with lists
            list_data = X.tolist()
            sequences = [np.array(seq, dtype=np.float32) for seq in list_data]
            X_array = np.stack(sequences)

            if len(X_array.shape) == 2:
                X_array = X_array.reshape(X_array.shape[0], X_array.shape[1], 1)
        else:
            # Handle other formats (numpy arrays, etc.)
            X_array = np.array(X)
            if len(X_array.shape) == 2:
                X_array = X_array.reshape(X_array.shape[0], X_array.shape[1], 1)

        # Handle pandas Series for y
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)

        # Ensure y is 2D: (batch_size, output_dim)
        if len(y_array.shape) == 1:
            y_array = y_array.reshape(-1, 1)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_array).to(self.device)
        y_tensor = torch.FloatTensor(y_array).to(self.device)

        return X_tensor, y_tensor

    def fit(self, X_train, y_train, epochs=100, validation_data=None, verbose=True):
        """
        Train the transformer model

        Args:
            X_train: pandas DataFrame with a single column containing lists
            y_train: pandas Series with target values
            epochs: Number of training epochs
            validation_data: Tuple of (X_val, y_val) for validation
            verbose: Whether to print training progress
        """
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X_train, y_train)
        self.sequence_length = X_tensor.shape[1]

        if verbose:
            print(f"Data shapes - X: {X_tensor.shape}, y: {y_tensor.shape}")
            if isinstance(X_train, pd.DataFrame):
                print(f"DataFrame column: {X_train.columns[0]}")
                print(f"Sample sequence length: {len(X_train.iloc[0, 0])}")

        # Create dataset and dataloader
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Validation data preparation
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training loop
        self.train()
        trainLosses = []
        validationLosses = []
        for epoch in tqdm(range(epochs), desc="Training Progress", disable=not verbose):
            total_loss = 0
            num_batches = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            self.history['loss'].append(avg_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                if 'val_loss' not in self.history:
                    self.history['val_loss'] = []
                self.history['val_loss'].append(val_loss)

            # early stopping if validation loss
            trainLosses.append(avg_loss)
            validationLosses.append(val_loss)

        if verbose:
            print("Training completed!")

        # make a plot of the training and validation losses

        plt.figure(figsize=(10, 5))
        plt.plot(trainLosses, label='Training Loss')
        if val_loader is not None:
            plt.plot(validationLosses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        return self

    def _validate(self, val_loader, criterion):
        """Compute validation loss"""
        self.eval()
        total_val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                num_batches += 1

        self.train()
        return total_val_loss / num_batches

    def predict(self, X, scaler=None):
        """Make predictions on new data"""
        self.eval()

        # Create dummy y for _prepare_data consistency
        if isinstance(X, pd.DataFrame):
            dummy_y = pd.Series([0] * len(X))
        else:
            dummy_y = np.zeros(len(X))

        X_tensor, _ = self._prepare_data(X, dummy_y)

        with torch.no_grad():
            predictions = self(X_tensor)

        value = predictions.cpu().numpy()
        if scaler is not None:
            value = scaler.inverse_transform(value)
        return value

    def predict_value(self, starter, scaler=None):
        new = starter
        new, next = sequence(new, len(new) - 1)
        new = pd.DataFrame({'sequences': [new]})
        if scaler:
            new = pd.DataFrame({
                'sequences': normalize_sequences(new.iloc[:, 0], scaler)
            })
        else:
            new = pd.DataFrame({'sequences': [new]})
        return self.predict(new, scaler)[0, 0]


    def score(self, X, y):
        """Calculate R² score"""
        predictions = self.predict(X)

        # Handle pandas Series
        if isinstance(y, pd.Series):
            y_true = y.values.reshape(-1, 1)
        else:
            y_true = np.array(y).reshape(-1, 1)

        ss_res = np.sum((y_true - predictions) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def rmse(self, X, y):
        """Calculate RMSE"""
        predictions = self.predict(X)

        # Handle pandas Series
        if isinstance(y, pd.Series):
            y_true = y.values.reshape(-1, 1)
        else:
            y_true = np.array(y).reshape(-1, 1)

        return np.sqrt(np.mean((y_true - predictions) ** 2))