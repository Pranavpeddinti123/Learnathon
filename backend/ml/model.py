import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for Human Activity Recognition
    
    Architecture:
    - 2 LSTM layers with dropout
    - Fully connected output layer
    - Softmax for classification
    """
    
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, 
                 num_classes=6, dropout=0.3):
        """
        Args:
            input_size: Number of input features (9 sensor signals)
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of activity classes (6 activities)
            dropout: Dropout probability
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        # out: (batch_size, seq_length, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use output from last timestep
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out
    
    def predict(self, x):
        """
        Make predictions with softmax probabilities
        
        Args:
            x: Input tensor
        
        Returns:
            Predicted class indices and probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities


def get_model(input_size=9, hidden_size=128, num_layers=2, 
              num_classes=6, dropout=0.3):
    """
    Factory function to create and return model instance
    
    Args:
        input_size: Number of input features
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        num_classes: Number of output classes
        dropout: Dropout probability
    
    Returns:
        LSTMClassifier model instance
    """
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )
    
    return model


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    model = get_model()
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 32
    seq_length = 128
    input_size = 9
    
    dummy_input = torch.randn(batch_size, seq_length, input_size)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    predictions, probabilities = model.predict(dummy_input)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
