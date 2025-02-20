import os
import argparse
import logging
import json
import torch
import datetime
import pathlib
import pandas as pd
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
        model = CNN(device, **hparams)
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
    test_idx:int,
    save_path: str
    ):
    
    if model_type == "CNN":
        hparams = {
            "num_DV" : 11
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

    # logger.info(f"LOOCV Iteration: Bush {test_idx} started")

    ml_model.train(dataset, n_epochs, batch_size, lr, test_idx=0, save_path=save_path)
    
def model_test(
    model_type:str,
    dataset,
    test_idx:int,
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
        model = CNN.load(model_path, device)
    elif model_type == "MLP":
        model = MLP.load(model_path, device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    _, _, test_loader, _, _, _, _ = dataset.get_loader()
    
    pred_percentages = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    result_df = pd.DataFrame(columns=["stiffness_num", "40%", "50%", "60%", "70%", "80%","90%", "100%"])
    for idx, (inputs, outputs) in enumerate(test_loader):
        prediction = model.predict(inputs) # input은 스케일이 이미 된 상태로 들어옴
        prediction = prediction 
        gt_output = outputs.numpy() # output은 굳이 스케일링해서 넣을 필요 없음
        
        input_unscaled = model.input_scaler.inverse_transform(inputs.numpy())
        
        save_path = os.path.join(model_path, f'stiffness_{idx+1}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        wmape_per_percent_list, wmape_full_range_list = results_extraction(input_unscaled, prediction, gt_output, pred_percentages=pred_percentages, save_path=save_path)
        
        new_row = pd.DataFrame({"stiffness_num": idx+1, "40%": wmape_per_percent_list[0], "50%": wmape_per_percent_list[1], "60%": wmape_per_percent_list[2],
                                "70%": wmape_per_percent_list[3], "80%": wmape_per_percent_list[4], "90%": wmape_per_percent_list[5], "100%": wmape_full_range_list[0]}, index=[0])
        
        result_df = pd.concat([result_df, new_row], ignore_index=True) 
        
    mean_row = pd.DataFrame({"stiffness_num": "Mean", "40%": result_df["40%"].mean(), "50%": result_df["50%"].mean(), "60%": result_df["60%"].mean(),
                             "70%": result_df["70%"].mean(), "80%": result_df["80%"].mean(), "90%": result_df["90%"].mean(), "100%": result_df["100%"].mean()}, index=[0])   
    result_df = pd.concat([result_df, mean_row], ignore_index=True)
    
    result_df.to_csv(os.path.join(model_path, "result.csv"), index=False)    
        
     
        
        
        
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_type", type=str, default="CNN")
    args = parser.parse_args()
    
    data_path = r"E:\Dongwoo\TeamWork\Hyundai_bush_2\github\bush_stiffness_prediction\resource\combined_data_16_106_70per_energy_coeff.npy"
    gt_data_path = rf'./resource/combined_data_10.npy'
    test_set = [0,5,13,18,29,31,70,71] # 몇번 인덱스로 테스트 하실래여?
    # test_set = [0] # 몇번 인덱스로 테스트 하실래여?
    result_path = pathlib.Path("results") / f"{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_keys = ['06_04_NX4']

    dataset = VEPDataset(output_path=data_path, test_keys=test_keys)
    train_model(args.model_type, dataset, args.n_epochs, args.batch_size, args.lr, test_idx=0, save_path=result_path)
        
    
    # for test_idx in test_set:
    #     dataset = BushDataset(batch=args.batch_size, output_path=data_path, gt_path=gt_data_path, field_range=256, num_stiffness=6)
    #     dataset.update_test_idx(test_idx)
    #     model_test(args.model_type, dataset=dataset, test_idx=test_idx, model_path=rf'E:\Dongwoo\TeamWork\Hyundai_bush_2\github\bush_stiffness_prediction\results\MLP_20250218_211141\bush_idx[{test_idx}]')