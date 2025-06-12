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
    
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    batch_size = 256
    BN_momentum = trial.suggest_float("BN_momentum", 0.01, 0.3)
    dropout_rate_fc = trial.suggest_float("dropout_rate_fc", 0.1, 0.5)
    dropout_rate_cnn = trial.suggest_float("dropout_rate_cnn", 0.1, 0.5)
    start_ch = trial.suggest_categorical("start_ch", [512, 1024, 2048, 4096])
    embedding_dim1 = trial.suggest_categorical("embedding_dim1", [512, 1024, 2048])
    embedding_dim2 = embedding_dim1
    # activation = trial.suggest_categorical("activation", ['ELU'])
    activation = 'ELU'  # 현재 ELU만 사용
    data_path = "./resource/중실_final/combined_7.npy" # 데이터 경로
    
    test_keys = ['06_04_NX4', '06_05_NX4', 'G_05_07_IK', 'G_06_04_IK', 'G_07_05_IK', 'G_08_06_IK', 'G_09_05_IK', 'G_10_03_IK',
                'G_11_06_IK', 'G_12_05_IK', 'G_13_04_IK', 'G_15_01_IK',  '06_06_LX2', '06_07_KA4', '06_08_US4',
                '06_11_MQ4', 'B_02', 'B_05'] # 현대차 부싱 이름들
    
    # # # exclude under 40% 
    exclude_indices4 = ['Run92', 'Run89', 'Run88', 'Run154', 'Run83', 'Run128', 'Run81', 'Run37',
                        'Run96', 'Run49', 'Run7',  'Run123',]  ## 15 %
                        # 'Run101', 'Run119', 'Run72', 'Run75', 'Run164', 'Run191', 'Run149',
                        # 'Run166', 'Run28', 'Run138', 'Run62']   ## 20%

    exclude_indices6 = ['Run146', 'Run176', 'Run177', 'Run195', 'Run208']
    exclude_keys = list(set(exclude_indices4+exclude_indices6))
        

    # test_key = '06_04_NX4' # 단일 부싱 테스트
    # test_key = ''

    # 학습진행
    val_loss_list = []
    for test_key in test_keys:
        dataset = VEPDataset(output_path=data_path, test_key=test_key, exclude_keys=exclude_keys)
        val_loss = train_model(
                            "SHCNN",
                            dataset,
                            n_epochs=3000, 
                            batch_size=batch_size,
                            lr=lr,
                            test_key=test_key,
                            save_path=result_path,
                            BN_momentum=BN_momentum,
                            dropout_rate_fc=dropout_rate_fc,
                            dropout_rate_cnn=dropout_rate_cnn,
                            start_ch=start_ch, 
                            embedding_dim1=embedding_dim1,
                            embedding_dim2=embedding_dim2,
                            activation=activation
                               )
        val_loss_list.append(val_loss)
        
    val_loss = np.mean(val_loss_list)
    
    return val_loss

def get_save_callback(save_dir):
    def save_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # 전체 trial 기록 저장
        df_all = study.trials_dataframe()
        df_all.to_csv(save_dir / "tuning_result.csv", index=False)

        # best trial이 갱신되었을 경우
        if study.best_trial == trial:
            best_data = trial.params.copy()
            best_data["value"] = trial.value
            best_data["datetime"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            best_df = pd.DataFrame([best_data])

            # best_trial.csv는 덮어쓰기
            best_df.to_csv(save_dir / "best_trial.csv", index=False)

            # best_trial_history.csv는 누적 저장
            history_path = save_dir / "best_trial_history.csv"
            if history_path.exists():
                best_df.to_csv(history_path, mode='a', header=False, index=False)
            else:
                best_df.to_csv(history_path, index=False)

    return save_callback

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=500)
    args = parser.parse_args()
    
    result_path = pathlib.Path("results") / 'tuned' / f"Tuned_SHCNN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_path.mkdir(parents=True, exist_ok=True)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, callbacks=[get_save_callback(result_path)])
    
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