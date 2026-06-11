import torch
import torch.nn as nn
import numpy as np
import random
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionModel(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [256, 128],
                 num_classes: int = 10, dropout: float = 0.2):
        super(ProductionModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        self.config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'num_classes': num_classes,
            'dropout': dropout
        }
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.softmax(self(x), dim=1)
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
            'network': self.network
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

class Trainer:
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 lr: float = 0.001, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return val_loss / len(val_loader), 100. * correct / total
    
    def fit(self, train_loader, val_loader, epochs: int = 100, patience: int = 10):
        logger.info(f"Training for {epochs} epochs with patience {patience}")
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                self.model.save('models/best_model.pt')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.2f}%")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return self.history


def main():
    logger.info("Starting training pipeline")
    
    # Set all seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Example: simple dataset (replace with real data)
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create synthetic dataset
    n_samples = 1000
    X = torch.randn(n_samples, 784)
    y = torch.randint(0, 10, (n_samples,))
    
    train_dataset = TensorDataset(X, y)
    val_dataset = TensorDataset(X, y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = ProductionModel(input_dim=784, hidden_dims=[512, 256, 128], num_classes=10)
    
    # Train
    trainer = Trainer(model, lr=0.001)
    history = trainer.fit(train_loader, val_loader, epochs=50, patience=10)
    
    logger.info("Training completed successfully")
    
    return history

if __name__ == '__main__':
    main()
