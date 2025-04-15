import os
import argparse
import logging
import json
import torch
import datetime
import pathlib
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import optuna 

import numpy as np

from source import *
from network.optuna import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

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
            
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    return model

     
def run_kfold(
    model_type:str,
    dataset,
    n_epochs:int,
    batch_size:int,
    lr:float,
    save_path: str,
    **hparams
    ):
    

    X = dataset.np_train_input
    Y = dataset.np_train_output
    
    kf = KFold(n_splits=10, shuffle=True, random_state=2025)
    
    best_val_loss = -float("inf")
    best_fold = None
    val_loss_list = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr , X_val = X[train_idx], X[val_idx]
        Y_tr , Y_val = Y[train_idx], Y[val_idx]
        
        input_scaler = StandardScaler()
        output_scaler = StandardScaler()
        
        field_range = 1 
        
        Y_tr, Y_val = Y_tr.reshape(-1, field_range), Y_val.reshape(-1, field_range)
        
        X_tr = input_scaler.fit_transform(X_tr)
        X_val = input_scaler.transform(X_val)
        Y_tr = output_scaler.fit_transform(Y_tr)
        Y_val = output_scaler.transform(Y_val)
        
        Y_tr = Y_tr.reshape(-1, 6, 16, 16)
        Y_val = Y_val.reshape(-1, 6, 16, 16)
        # Create datasets and data loaders
        train_dataset = BushDataset(X_tr, Y_tr)
        val_dataset = BushDataset(X_val, Y_val)
    

        ml_model = build_model(model_type=model_type, num_DV = 17 , **hparams)
        
        val_loss = ml_model.train(
            train_dataset= train_dataset,
            val_dataset= val_dataset,
            input_scaler= input_scaler,
            output_scaler= output_scaler,
            n_epochs= n_epochs,
            batch_size= batch_size,
            lr= lr,
            fold_idx= fold_idx,
            save_path= save_path
        )
        val_loss_list.append(val_loss)
        
        logger.info(f"Fold {fold_idx}: val loss: {val_loss}")
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            best_fold = fold_idx
    
    logger.info(f"Best fold: {best_fold} with val loss: {best_val_loss}")
    val_loss_mean = np.mean(val_loss_list)
    
    return val_loss_mean, best_val_loss, best_fold
            
            
def objective(trial):
    
    lr = trial.suggest_float("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    BN_momentum = trial.suggest_float("BN_momentum", 0.1, 0.9)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    start_ch = trial.suggest_categorical("start_ch", [128, 256, 512, 1024, 2048])
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256, 512, 1024])
    activation = trial.suggest_categorical("activation", ['SiLU', 'ReLU', 'LeakyReLU', 'ELU'])
    data_path = "./resource/250413_150개/combined_7.npy" # 데이터 경로

    exclude_keys1 = ['Run82', 'Run83', 'Run85', 'Run86', 'Run88', 'Run89', 'Run100',
                    'Run101','Run102','Run103','Run104','Run105']

    # # exclude under 40%  acc: 82%
    exclude_keys2 = ['Run128', 'Run37', 'Run92', 'Run89', 'Run88', 'Run83', 'Run81', 'Run123','Run7', 'Run96', 'Run73', 'Run72', 'Run149',
                    'Run21', 'Run28', 'Run49', 'Run101', 'Run62', 'Run164', 'Run75', 'Run71', 'Run30', 'Run33', 'Run125', 'Run138']
    
    exclude_keys = list(set(exclude_keys1 + exclude_keys2))
    
    dataset = VEPDataset(output_path=data_path, test_key=None, exclude_keys=exclude_keys)
    val_loss_mean, best_val_loss, best_fold = run_kfold(
                                    "SHCNN",
                                    dataset,
                                    n_epochs=1000, 
                                    batch_size=batch_size,
                                    lr=lr,
                                    save_path=result_path,
                                    BN_momentum=BN_momentum,
                                    dropout_rate=dropout_rate,
                                    start_ch=start_ch, 
                                    embedding_dim=embedding_dim,
                                    activation=activation
                                    )
    
    return val_loss_mean

def save_callback(study, trial):
    df = study.trials_dataframe()
    df.to_csv("tuning_result.csv", index=False)
    
if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=500)
    args = parser.parse_args()
    
    result_path = pathlib.Path("results") / 'tuned' / f"Tuned_SHCNN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials,  callbacks=[save_callback])
    
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
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    save_path = os.path.join(result_path, "tuning_result.json")
    with open(save_path, "w") as f:
        json.dump(study.best_params, f)
    study.trials_dataframe().to_csv(save_path + "tuning_result.csv", index=False)


        