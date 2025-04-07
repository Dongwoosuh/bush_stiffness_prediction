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
    test_key:int,
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
            "dropout_rate" : 0.1,
            "start_ch" : 1024,
            "embedding_dim" : 128
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
            "embdding_dim" : 64}
     
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

    logger.info(f"LOOCV Iteration: {test_key}_Bush started")

    best_val_loss = ml_model.train(dataset, n_epochs, batch_size, lr, test_key=test_key, save_path=save_path)
    
def model_test(
    model_type:str,
    dataset,
    test_key:int,
    model_path: str
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
    
    input_scaler = model.input_scaler
    output_scaler = model.output_scaler
    
    test_inputs = dataset.np_test_input
    test_inputs = input_scaler.transform(test_inputs)
    test_outputs = dataset.np_test_output
    
    test_dataset = BushDataset(test_inputs, test_outputs)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # pred_percentages = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pred_percentages = [1.0]
    
    result_df = pd.DataFrame(columns=["stiffness_num", "100%"])
    for idx, (inputs, outputs) in enumerate(test_loader):
        prediction = model.predict(inputs) # input은 스케일이 이미 된 상태로 들어옴
        prediction = np.expm1(prediction.reshape(-1,16,16))
        
        gt_output = outputs.numpy().reshape(-1,16,16) # output은 굳이 스케일링해서 넣을 필요 없음
        # gt_output = np.expm1(gt_output)
        
        input_unscaled = model.input_scaler.inverse_transform(inputs.numpy())
        
        for idx_ in range(len(gt_output)):
            
            save_path = os.path.join(model_path, f'stiffness_{idx_+1}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            wmape_per_percent_list, wmape_full_range_list = results_extraction(input_unscaled, prediction[idx_], gt_output[idx_], pred_percentages=pred_percentages, save_path=save_path)
            
            new_row = pd.DataFrame({"stiffness_num": idx_+1, "100%": wmape_full_range_list[0]}, index=[0])
            
            result_df = pd.concat([result_df, new_row], ignore_index=True) 
            
        mean_row = pd.DataFrame({"stiffness_num": "Mean", "100%": result_df["100%"].mean()}, index=[0])   
        result_df = pd.concat([result_df, mean_row], ignore_index=True)
        
        result_df.to_csv(os.path.join(model_path, "result.csv"), index=False)    
        
     
        
        
        
if __name__ == "__main__" :
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_type", type=str, default="SHCNN")
    args = parser.parse_args()
    
    train_percents = [7]
    for train_percent in train_percents:
        data_path = f"./resource/250407_126/combined_{train_percent}.npy" # 데이터 경로
        
        result_path = pathlib.Path("results") / f"{args.model_type}_{train_percent*10}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_keys = ['06_04_NX4', '06_05_NX4', 'G_05_07_IK', 'G_06_04_IK', 'G_07_05_IK', 'G_08_06_IK', 'G_09_05_IK', 'G_10_03_IK',
                    'G_11_06_IK', 'G_12_05_IK', 'G_13_04_IK', 'G_15_01_IK',  '06_06_LX2', '06_07_KA4', '06_08_US4',
                    '06_11_MQ4', 'B_02', 'B_05'] # 현대차 부싱 이름들
        
        test_keys = ['06_05_NX4'] # 단일 부싱 테스트

        # 학습진행
        # for test_key in test_keys:
        #     dataset = VEPDataset(output_path=data_path, test_key=test_key)
        #     train_model(args.model_type, dataset, args.n_epochs, args.batch_size, args.lr, test_key=test_key, save_path=result_path)
        
    # 테스트 진행
    
    for test_key in test_keys:
        dataset = VEPDataset(output_path=data_path, test_key=test_key)
        model_test(args.model_type, dataset=dataset, test_key=test_key, model_path=rf'./results/SHCNN_70_20250407_145059/{test_key}')