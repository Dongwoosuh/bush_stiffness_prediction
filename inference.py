import os
import argparse
import logging
import json
import torch
import datetime
import pathlib
import pandas as pd
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.model_selection import train_test_split

import numpy as np
import tqdm

from source import *
from network.vanila import *

logger = logging.getLogger(__name__)

def build_model(model_type:str, **hparams):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        logger.warning("CUDA is not available. Running on CPU")
        
    if model_type == "CNN":
        model = BaseCNN(device, **hparams)
        
    elif model_type == "SHCNN":
        model = SHCNN(device, **hparams)
        
    elif model_type == "DWCNN":
        model = DWCNN(device, **hparams)
            
    elif model_type == "Transformer":
        model = BaseTransformer(device, **hparams)
        
    elif model_type == "MLP":
        model = MLP(device, **hparams)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    return model

def train_model(
    model_type:str,
    dataset,
    n_epochs:int,
    batch_size:int,
    lr:float,
    save_path: str
    ):
    
    if model_type == "CNN":
        hparams = {
            "num_DV" : 11
        }
        
    elif model_type == "SHCNN":
        hparams = {
            "num_DV" : 17,
            "BN_momentum" : 0.1,
            "dropout_rate" : 0.3,
            "start_ch" : 2048,
            "embedding_dim1" : 1024,
            "embedding_dim2" : 1024,
            'activation' : 'ELU'
        }

    elif model_type == "DWCNN":
        hparams = {
            "num_DV" : 17,
            "BN_momentum" : 0.2,
            "dropout_rate" : 0.2,
            "start_ch" : 6*16*16,
            "padding_param" : 0,
            "kernel_size" : 3,
            "stride" : 2,
            "embdding_dim" : 1024}
     
    elif model_type == "Transformer":
        hparams = {
            "num_DV" : 17,
            "embed_dim" : 128,
            "num_heads" : 4,
            "num_layers" : 2,
            "output_dim" : 256,
            "dropout" : 0.1
        }
        
    elif model_type == "MLP":
        hparams = {
            "num_DV" : 12,
            "hidden_features" : 128,
            "num_layers" : 5,
            "drop_out" : 0.3,
            "hidden_activation" : "SiLU"
        }
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    
    ml_model = build_model(model_type=model_type, **hparams)

    best_val_loss = ml_model.train(dataset, n_epochs, batch_size, lr, save_path=save_path)
    
def model_test(
    model_type:str,
    dataset,
    # test_key:int,
    model_path: str,
    result_path: str = None,
    ):
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        logger.warning("CUDA is not available. Running on CPU")
        
    if model_type == "CNN":
        model = BaseCNN.load(model_path, device)
        
    elif model_type == "SHCNN":
        model = SHCNN.load(model_path, device)
        
    elif model_type == "DWCNN":
        model = DWCNN.load(model_path, device)
        
    elif model_type == "Transformer":
        model = BaseTransformer.load(model_path, device)
        
    elif model_type == "MLP":
        model = MLP.load(model_path, device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    test_key = [dataset.inference_name]
    
    input_scaler_shape = model.input_scaler_shape
    input_scaler_linear = model.input_scaler_linear
    output_scaler = model.output_scaler
    
    test_inputs = dataset.input_data
    test_inputs_shape_unscaled = test_inputs[:,:8]
    
    test_inputs_shape = input_scaler_shape.transform(test_inputs[:,:8])
    test_inputs_linear = input_scaler_linear.transform(test_inputs[:,8:14].flatten().reshape(-1,1))
    test_inputs_linear = test_inputs_linear.reshape(test_inputs[:,8:14].shape)
    
    test_inputs = np.hstack((test_inputs_shape, test_inputs_linear))
    
    test_dataset = BushDataset(test_inputs)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    bush_save_path = os.path.join(result_path, test_key[0])
    if not os.path.exists(bush_save_path):
        os.makedirs(bush_save_path)
        
    for idx, inputs in enumerate(test_loader):
        prediction = model.predict(inputs) # input은 스케일이 이미 된 상태로 들어옴
        prediction = np.expm1(prediction.reshape(-1,16,16))
        
        for idx_ in range(6):
            
            save_path = os.path.join(bush_save_path, f'stiffness_{idx_+1}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            inference_results_extraction(test_inputs_shape_unscaled, prediction[idx_], bush_name= test_key[0], save_path=save_path)

        
        
        
if __name__ == "__main__" :
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--model_type", type=str, default="SHCNN")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    
    train_percents = [7]
    for train_percent in train_percents:
        data_path = f"./resource/250504/combined_{train_percent}.npy" # 데이터 경로
            
        exclude_keys4 = ['Run92', 'Run89', 'Run88', 'Run154', 'Run83', 'Run128', 'Run81', 'Run37',
                            'Run96', 'Run49', 'Run7',  'Run123',   ## 15 %
                            'Run101', 'Run72', 'Run75', 'Run164', 'Run191', 'Run149',
                            'Run166', 'Run28', 'Run138', 'Run62']   ## 20%

        
        exclude_keys = list(set(exclude_keys4))
        
        total_data = np.load(data_path, allow_pickle=True).item()
        
        data_keys_list = sorted(set(total_data.keys()) - set(exclude_keys))
        
        all_indices = np.arange(len(data_keys_list))
        
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=0.1,
            shuffle=True,
            random_state=args.seed,
        )
        
        test_keys  = [data_keys_list[i] for i in test_idx]
        

        
        # 테스트 진행
        model_path = rf'./results/Final_Model/SHCNN_70_seed_2025_20250508_141201'
        result_path = rf'./results/Inference'
            
        dataset = InferenceVEPDataset(csv_path="./resource/inference_example/example.csv")
        result_dict = model_test(args.model_type, dataset=dataset, model_path=rf'{model_path}', result_path=result_path)
            