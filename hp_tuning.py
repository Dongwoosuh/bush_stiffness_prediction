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
import optuna

import numpy as np
import tqdm

from source import *
from network.optuna import *

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
            
    # elif model_type == "Transformer":
    #     model = BaseTransformer(device, **hparams)
        
    # elif model_type == "MLP":
    #     model = MLP(device, **hparams)
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
    save_path: str,
    **hparams
    ):

    ml_model = build_model(model_type=model_type,num_DV=17, **hparams)

    logger.info(f"LOOCV Iteration: {test_key}_Bush started")

    best_val_loss = ml_model.train(dataset, n_epochs, batch_size, lr, test_key=test_key, save_path=save_path)
    
    return best_val_loss

def objective(trial):
    
    lr = trial.suggest_float("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
    BN_momentum = trial.suggest_float("BN_momentum", 0.1, 0.9)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    start_ch = trial.suggest_categorical("start_ch", [128, 256, 512, 1024, 2048])
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256, 512, 1024])
    
    data_path = "./resource/250307_122개_linear/combined_7.npy" # 데이터 경로
    
    # test_keys = ['06_04_NX4', '06_05_NX4', 'G_05_07_IK', 'G_06_04_IK', 'G_07_05_IK', 'G_08_06_IK', 'G_09_05_IK', 'G_10_03_IK',
    #             'G_11_06_IK', 'G_12_05_IK', 'G_13_04_IK', 'G_15_01_IK',  '06_06_LX2', '06_07_KA4', '06_08_US4',
    #             '06_11_MQ4', 'B_02', 'B_05'] # 현대차 부싱 이름들
    
    test_key = '06_04_NX4' # 단일 부싱 테스트
    # test_key = ''

    # 학습진행
    dataset = VEPDataset(output_path=data_path, test_key=test_key)
    
    val_loss = train_model("SHCNN", dataset, n_epochs=1000, batch_size=batch_size, lr=lr, test_key=test_key, save_path=result_path, BN_momentum=BN_momentum, dropout_rate=dropout_rate, start_ch=start_ch, embedding_dim=embedding_dim)
    
    return val_loss

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=500)
    args = parser.parse_args()
    
    result_path = pathlib.Path("results") / 'tuned' / f"Tuned_SHCNN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("Best trial:")
    print(study.best_trial)
    
    print("Best parameters:")
    print(study.best_params)
    
    print("Best value:")
    print(study.best_value)
    
    print("All trials:")
    print(study.trials)
    
    print("All trials:")
    print(study.trials_dataframe())
    
    study.trials_dataframe().to_csv("tuning_result.csv", index=False)
    
    save_path = os.path.join(result_path, "tuning_result.json")
    with open(save_path, "w") as f:
        json.dump(study.best_params, f)