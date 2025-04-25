import numpy as np
import torch
import ast
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  QuantileTransformer, MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
import warnings
warnings.filterwarnings(action='ignore')
__all__ = ["BushDataset", "VEPDataset", "InferenceVEPDataset"]
# In[3. Data setting] #############################################################################################

def load_bush_data(total_output_data: dict):
    """
    부시별 input/output 딕셔너리를 
    (bush_names, inputs, outputs) 형태로 반환
    """
    bush_names = list(total_output_data.keys())  # 예: ["06_04_NX4", "06_05_NX4", ...]
    
    # 부시별로 input/output을 하나의 리스트에 쌓기
    all_inputs = []
    all_outputs = []
    for name in bush_names:
        sub_dict = total_output_data[name]
        # 'input'과 'output'이 존재한다고 가정
        bush_input = sub_dict["input"]   # shape 예: (feature_dim,)
        bush_output = sub_dict["output"] # shape 예: (6,16,16) or (16,16) 등
        all_inputs.append(bush_input)
        all_outputs.append(bush_output)

    # 리스트를 numpy 배열로 변환
    # (주의) output이 shape이 제각각이면 object dtype으로 변환될 수 있음
    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_outputs = np.array(all_outputs, dtype=np.float32)

    return bush_names, all_inputs, all_outputs

class BushDataset(Dataset):
    """
    부시별 input, output을 Dataset으로 감싸는 예시
    """
    def __init__(self, inputs, outputs):
        super().__init__()
        self.inputs = inputs  # shape: (N, feature_dim) 또는 object
        self.outputs = outputs
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]

        return x, y

class VEPDataset():
    def __init__(self, output_path:str, test_key:str, exclude_keys:list=None):

        self.test_key = test_key
        self.total_data = np.load(output_path, allow_pickle=True).item()
        
        train_data, test_data = self.get_test_keys(test_key, exclude_keys)
        
        # (2) load_bush_data로 bush_names, inputs, outputs 추출
        _, train_inputs, train_outputs = load_bush_data(train_data)
        _, test_inputs, test_outputs = load_bush_data(test_data)
        
        # 기하적 최대범위 추가가
        x_disp, z_disp, theta_x = get_extrapolation_range(train_inputs[:,:8])
        train_inputs = np.hstack((train_inputs, x_disp.reshape(-1, 1), z_disp.reshape(-1, 1), theta_x.reshape(-1,1))) 
        
        train_inputs = np.array([np.asarray(i, dtype=np.float32) for i in train_inputs])
        train_inputs[:, 8:14] = np.log1p(train_inputs[:, 8:14])
        
        
        train_outputs = np.array([np.asarray(o, dtype=np.float32) for o in train_outputs])
        train_outputs = np.log1p(train_outputs)
        
        if len(test_inputs) == 0:
            pass
        else:
            x_disp_, z_disp_, theta_x_ = get_extrapolation_range(test_inputs[:,:8])
            test_inputs = np.hstack((test_inputs, x_disp_.reshape(-1, 1), z_disp_.reshape(-1, 1), theta_x_.reshape(-1,1)))
            test_inputs = np.array([np.asarray(i, dtype=np.float32) for i in test_inputs])
            test_inputs[:, 8:14] = np.log1p(test_inputs[:, 8:14])
            test_outputs = np.array([np.asarray(o, dtype=np.float32) for o in test_outputs])
            # test_outputs = np.log1p(test_outputs)
        
        
        # Placeholders for dynamic updates
        self.np_train_input = train_inputs
        self.np_train_output = train_outputs
        self.np_test_input = test_inputs
        self.np_test_output = test_outputs
        self.input_scaler_shape = None
        self.input_scaler_linear = None
        self.output_scaler = None
        
    def get_datasets(self):
        
        # Split train data into train and validation sets
        train_input, val_input, train_output, val_output = train_test_split(
            self.np_train_input, self.np_train_output, test_size=0.1, random_state=2025
        )
        
        train_input_shape = train_input[:, :8]
        train_input_linear = train_input[:,8:14].flatten().reshape(-1,1)
        
        val_input_shape = val_input[:, :8]
        val_input_linear = val_input[:,8:14].flatten().reshape(-1,1)
        
        self.input_scaler_shape = StandardScaler()
        self.input_scaler_linear = StandardScaler()
        self.output_scaler = StandardScaler()
        
        field_range = 1
    
        train_output = train_output.reshape(-1, field_range)
        val_output = val_output.reshape(-1, field_range)
        
        self.input_scaler_shape.fit(train_input_shape)
        train_input_shape = self.input_scaler_shape.transform(train_input_shape)
        val_input_shape = self.input_scaler_shape.transform(val_input_shape)
        
        self.input_scaler_linear.fit(train_input_linear)
        train_input_linear = self.input_scaler_linear.transform(train_input_linear).reshape(train_input[:,8:14].shape)
        val_input_linear = self.input_scaler_linear.transform(val_input_linear).reshape(val_input[:,8:14].shape)
        
        train_input = np.hstack((train_input_shape, train_input_linear))
        val_input = np.hstack((val_input_shape, val_input_linear))
        
        train_output = self.output_scaler.fit_transform(train_output)
        val_output = self.output_scaler.transform(val_output)
        
        # train_input = self.input_scaler.fit_transform(train_input)
        # val_input = self.input_scaler.transform(val_input)
        
        train_output = train_output.reshape(-1, 6, 16, 16)
        val_output = val_output.reshape(-1, 6, 16, 16)
        # Create datasets and data loaders
        train_dataset = BushDataset(train_input, train_output)
        val_dataset = BushDataset(val_input, val_output)
        # test_dataset = BushDataset(self.np_test_input, self.np_test_output)

        # train_loader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        # val_loader = DataLoader(val_dataset, batch_size=self.batch, shuffle=True, drop_last=False)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Single sample for LOOCV

        return train_dataset, val_dataset, self.input_scaler_shape, self.input_scaler_linear, self.output_scaler
    
    def get_test_keys(self, test_keys, exclude_keys):
        test_keys = [test_keys]
        exclude_keys = exclude_keys
        # test_data = {key: self.total_data[key] for key in test_keys if key in self.total_data}
        # train_data = {key: self.total_data[key] for key in self.total_data if key not in test_keys}
        test_data = {key: self.total_data[key] for key in test_keys if key in self.total_data}
        train_data = {key: self.total_data[key] for key in self.total_data if key not in test_keys and key not in exclude_keys}
        return train_data, test_data
    
class InferenceVEPDataset():
    def __init__(self, output_path:str, test_key:str):
        self.test_key = test_key
        self.total_data = np.load(output_path, allow_pickle=True).item()
        
        train_data, test_data = self.get_test_keys(test_key)
        
        # (2) load_bush_data로 bush_names, inputs, outputs 추출
        _, train_inputs, train_outputs = load_bush_data(train_data)
        _, test_inputs, test_outputs = load_bush_data(test_data)
        
        # 기하적 최대범위 추가가
        x_disp, z_disp, theta_x = get_extrapolation_range(train_inputs[:,:8])
        train_inputs = np.hstack((train_inputs, x_disp.reshape(-1, 1), z_disp.reshape(-1, 1), theta_x.reshape(-1,1))) 
        
        train_inputs = np.array([np.asarray(i, dtype=np.float32) for i in train_inputs])
        train_inputs[:, 8:14] = np.log1p(train_inputs[:, 8:14])
        
        
        train_outputs = np.array([np.asarray(o, dtype=np.float32) for o in train_outputs])
        train_outputs = np.log1p(train_outputs)
        
        test_outputs = np.array([np.asarray(o, dtype=np.float32) for o in test_outputs])
        test_outputs = np.log1p(test_outputs)
        
        
        # Placeholders for dynamic updates
        self.np_train_input = train_inputs
        self.np_train_output = train_outputs
        self.np_test_input = test_inputs
        self.np_test_output = test_outputs
        self.input_scaler = None
        self.output_scaler = None


        

def get_extrapolation_range(df):

    df = np.array(df, dtype=np.float64)

    scale_factor = 1.0487
    # Calculate rubber parameters
    D_O_RUBBER = 2 * (df[:, 0] + df[:, 1])
    D_I_RUBBER = 2 * df[:, 0]
    L_O_RUBBER = 2 * df[:, 2]
    L_I_RUBBER = 2 * (df[:, 2] + df[:, 3])

    # Calculate displacements and angles
    x_disp = (D_O_RUBBER - D_I_RUBBER) / 2 - df[:, 6]
    z_disp = (L_I_RUBBER * scale_factor - L_O_RUBBER) / 2
    theta_x = (np.arctan(D_O_RUBBER / L_O_RUBBER) - np.arcsin(D_I_RUBBER / np.sqrt(D_O_RUBBER**2 + L_O_RUBBER**2)))

    return x_disp, z_disp, theta_x


def get_extrapolation_range(df):

    df = np.array(df, dtype=np.float64)

    scale_factor = 1.0487
    # Calculate rubber parameters
    D_O_RUBBER = 2 * (df[:, 0] + df[:, 1])
    D_I_RUBBER = 2 * df[:, 0]
    L_O_RUBBER = 2 * df[:, 2]
    L_I_RUBBER = 2 * (df[:, 2] + df[:, 3])

    x_disp = np.where(df[:,6]<= 0,
                  (D_O_RUBBER - D_I_RUBBER) / 2,
                  (D_O_RUBBER - D_I_RUBBER) / 2 - df[:,6])
    
    z_disp = (L_I_RUBBER * scale_factor - L_O_RUBBER) / 2
    theta_x = (np.arctan(D_O_RUBBER / L_O_RUBBER) - np.arcsin(D_I_RUBBER / np.sqrt(D_O_RUBBER**2 + L_O_RUBBER**2)))

    return x_disp, z_disp, theta_x
if __name__ == "__main__":
    # Test code
    output_path = "./resource/250305_122/combined_7.npy"
    batch = 32
    test_keys = ['06_04_NX4', '06_05_NX4']
    for test_key in test_keys:
        dataset = VEPDataset(output_path=output_path, test_key=test_key)
        
        train_dataset, val_dataset, input_scaler, output_scaler= dataset.get_datasets()
    
        print(train_dataset[0])
    