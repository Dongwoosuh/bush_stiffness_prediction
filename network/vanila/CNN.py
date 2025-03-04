import os
from copy import deepcopy
import numpy as np
import logging
import csv
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from sklearn.preprocessing import  QuantileTransformer, MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from network.model.basic_model import CNN_small_dropout, CNN_linear_stiff
from source import *

__all__ = ["BaseCNN, LSCNN"]

logger = logging.getLogger(__name__)

class BaseCNN():
    def __init__(
        self,
        device: str,
        num_DV: int,
    ):
        
        self.device = device
        
        self.hparams = {
            "num_DV": num_DV,
        }
        
        self.model = CNN_small_dropout(num_DV).to(device)
        
        self.input_scaler = None
        self.output_scaler = None
        
    def train(self, dataset, n_epochs:int, batch_size:int, lr:float, test_key:str, save_path:str ):
        
    
        train_dataset, val_dataset, input_scaler, output_scaler= dataset.get_datasets()

        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
            
        current_out_path = os.path.join(save_path, f"{test_key}")
        os.makedirs(current_out_path, exist_ok=True)
        logger.debug(f"Output path: {current_out_path}")
        
        csv_logger_path = os.path.join(current_out_path, "loss.csv")
        csv_logger = csv.writer(open(csv_logger_path, "w", newline=""))
        csv_logger.writerow(["epoch", "train_loss", "val_loss"])
        logger.debug(f"CSV logger path: {csv_logger_path}")        
        
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=94, T_mult=1, eta_min=0, verbose=False)
        criterion = nn.MSELoss()
        
        epoch_progress = trange(n_epochs, desc="Epoch", leave=True)
        best_val_loss = float('inf')
        best_model_weights = None
        
        for epoch in epoch_progress:
            self.model.train()
            total_train_loss = 0.0
            
            for inputs, targets in tqdm(train_loader, desc="Batch", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                predictions= self.forward(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_train_loss += loss.item()
                
            total_train_loss = total_train_loss / len(train_loader)
            
            self.model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    predictions = self.forward(inputs)
                    val_loss = criterion(predictions, targets)
                    
                    total_val_loss += val_loss.item()
                
                total_val_loss = total_val_loss / len(val_loader)
                epoch_progress.set_postfix({"average loss": total_val_loss})
                
                if best_val_loss > total_val_loss:
                    best_val_loss = total_val_loss
                    
                    self.save(current_out_path)
                    logger.debug(f"Best model updated: {best_val_loss}, updated model saved in {current_out_path}")

                scheduler.step()

            csv_logger.writerow([epoch, total_train_loss, total_val_loss])
            

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
    
    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            input_unscaled = self.input_scaler.inverse_transform(inputs)
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs)
            outputs = outputs.detach().cpu().numpy()
            
            outputs_flat = outputs.reshape(-1, 6*16*16)
            # output_scaler = self.output_scaler[int(input_unscaled[:, -3])-1]
            outputs_flat = self.output_scaler.inverse_transform(outputs_flat)
            
            outputs = outputs_flat.reshape(outputs.shape)
        
        return outputs
            
    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        torch.save(self.input_scaler, os.path.join(path, "input_scaler.pth"))
        torch.save(self.output_scaler, os.path.join(path, "output_scalers.pth"))
        json.dump(self.hparams, open(os.path.join(path, "hparams.json"), "w"))
        
    @classmethod
    def load(cls, path, device):
        hparams = json.load(open(os.path.join(path, "hparams.json"), "r"))
        model = cls(**hparams, device=device)
        model.model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
        model.input_scaler = torch.load(os.path.join(path, "input_scaler.pth"))
        model.output_scaler = torch.load(os.path.join(path, "output_scalers.pth"))
        
        return model
    
class LSCNN(BaseCNN):
    def __init__(
        self,
        device: str,
        num_DV: int,
    ):
        self.device = device
        
        self.hparams = {
            "num_DV": num_DV,
        }
        
        self.model = CNN_linear_stiff(num_DV).to(device)
        
        self.input_scaler = None
        self.output_scaler = None
        
    