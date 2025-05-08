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
    
    input_scaler_shape = model.input_scaler_shape
    input_scaler_linear = model.input_scaler_linear
    output_scaler = model.output_scaler
    
    test_inputs = dataset.np_test_input
    test_inputs_shape_unscaled = test_inputs[:,:8]
    
    test_inputs_shape = input_scaler_shape.transform(test_inputs[:,:8])
    test_inputs_linear = input_scaler_linear.transform(test_inputs[:,8:14].flatten().reshape(-1,1))
    test_inputs_linear = test_inputs_linear.reshape(test_inputs[:,8:14].shape)
    
    test_inputs = np.hstack((test_inputs_shape, test_inputs_linear))
    test_outputs = dataset.np_test_output
    
    test_dataset = BushDataset(test_inputs, test_outputs)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # pred_percentages = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pred_percentages = [1.0]
    
    bush_save_path = os.path.join(model_path, test_key[0])
    if not os.path.exists(bush_save_path):
        os.makedirs(bush_save_path)
        
    
    result_df = pd.DataFrame(columns=["stiffness_num", "100%"])
    for idx, (inputs, outputs) in enumerate(test_loader):
        prediction = model.predict(inputs) # input은 스케일이 이미 된 상태로 들어옴
        prediction = np.expm1(prediction.reshape(-1,16,16))
        
        gt_output = outputs.numpy().reshape(-1,16,16) # output은 굳이 스케일링해서 넣을 필요 없음
        # gt_output = np.expm1(gt_output)
        
        for idx_ in range(len(gt_output)):
            
            save_path = os.path.join(bush_save_path, f'stiffness_{idx_+1}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            wmape_per_percent_list, wmape_full_range_list = results_extraction(test_inputs_shape_unscaled, prediction[idx_], gt_output[idx_], pred_percentages=pred_percentages, save_path=save_path)
            
            new_row = pd.DataFrame({"stiffness_num": idx_+1, "100%": wmape_full_range_list[0]}, index=[0])
            
            result_df = pd.concat([result_df, new_row], ignore_index=True) 
            
        mean_row = pd.DataFrame({"stiffness_num": "Mean", "100%": result_df["100%"].mean()}, index=[0])   
        result_df = pd.concat([result_df, mean_row], ignore_index=True)
        
        result_df.to_csv(os.path.join(bush_save_path, "result.csv"), index=False)    
    
    result_dict = {'Test Key': test_key[0],
                   'Mean WMAPE': result_df["100%"].mean(),
                   'Std WMAPE': result_df["100%"].std()}
    return result_dict
     
        
        
        
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
        
        result_path = pathlib.Path("results") /"Final_Model"/ f"{args.model_type}_{train_percent*10}_seed_{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        
        # 학습진행
        # dataset = VEPDataset(output_path=data_path, test_key=test_keys, exclude_keys=exclude_keys)
        # train_model(args.model_type, dataset, args.n_epochs, args.batch_size, args.lr, save_path=result_path)
        
        # 테스트 진행
        result_path = rf'./results/Final_Model/SHCNN_70_seed_2025_20250508_141201'
        
        result_dict_list = []
        for test_key in test_keys:
            test_key = [test_key]
            dataset = VEPDataset(output_path=data_path, test_key=test_key, exclude_keys=exclude_keys)
            result_dict = model_test(args.model_type, dataset=dataset, test_key=test_key, model_path=rf'{result_path}')
            result_dict_list.append(result_dict)
                
            pd.DataFrame(result_dict_list).to_csv(os.path.join(result_path, "test_result.csv"), index=False)