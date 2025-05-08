import os
import argparse
import logging
import json
import torch
import datetime
import pathlib
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split




###### Dataloader ######
class BushDataset(Dataset):
    """
    부시별 input, output을 Dataset으로 감싸는 예시
    """
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs  # shape: (N, feature_dim) 또는 object
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]

        return x
    
class InferenceVEPDataset():
    def __init__(self, csv_path: str):
        
        # Load CSV data
        df = pd.read_csv(csv_path)

        # Extract columns for input_data_shape and input_data_linear
        input_data_shape = df[['DV1', 'DV2', 'DV3', 'DV4', 'DV5', 'DV6', 'DV7', 'DV8']].values.astype(np.float32)
        input_data_linear = df[['Stiffness_1', 'Stiffness_2', 'Stiffness_3', 'Stiffness_4', 'Stiffness_5', 'Stiffness_6']].values.astype(np.float32)

        # Combine into a single numpy array
        input_data = np.hstack((input_data_shape, input_data_linear))
        inference_name = csv_path.split('/')[-1].split('.')[0]  # Extract name before .csv
        
        self.input_data = input_data
        self.inference_name = inference_name
###############################################


##### POLY REGRESSION #####
def predict_on_grid(model, poly, grid_points):
    """
    Predict using the polynomial regression model on the grid points.
    Args:
        model: Trained LinearRegression model.
        poly: PolynomialFeatures instance.
        grid_points: Grid points (2D numpy array).
    Returns:
        Predicted values on the grid.
    """
    grid_points_poly = poly.transform(grid_points)  # Expand grid points to polynomial terms
    return model.predict(grid_points_poly)

def polynomial_regression(X, Z, degree, prev_model=None, prev_poly=None):

    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)  # Expand input features to polynomial terms

    # Linear regression with positive constraint
    model = LinearRegression(positive=True, fit_intercept=False)
    # model = LinearRegression(fit_intercept=False)

    try:
        model.fit(X_poly, Z)  # Fit linear regression on expanded features
    except RuntimeError as e:
        # print(f"Fitting failed for degree {degree}: {e}")
        if prev_model is not None and prev_poly is not None:
            # print("Using previous model and polynomial features as fallback.")
            return prev_model, prev_poly
        else:
            raise RuntimeError("No previous model available to fallback.")
    
    return model, poly

def loocv_optimization(X, Z, max_degree=6):

    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    loo = LeaveOneOut()
    errors = []
    valid_degrees = []

    for degree in range(2, max_degree + 1):
        try:
            mse_list = []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Z_train, Z_test = Z[train_index], Z[test_index]

                # Train the model
                model, poly = polynomial_regression(X_train, Z_train, degree)
                # Predict on the test set
                Z_pred = predict_on_grid(model, poly, X_test)
                # Compute the mean squared error
                mse_list.append(mean_squared_error(Z_test, Z_pred))

            # Average MSE for this degree
            errors.append(np.mean(mse_list))
            valid_degrees.append(degree)

        except Exception as e:
            print(f"Error occurred for degree {degree}: {str(e)}")
            continue

    if not valid_degrees:
        raise ValueError("All degrees failed during LOOCV.")

    # Find the degree with the lowest error
    optimal_degree = valid_degrees[np.argmin(errors)]
    return optimal_degree
###################################################################


######## Result Extraction #######
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


def inference_results_extraction(input_data_unscaled, prediction, bush_name, save_path:str):
    
    x_disp, z_disp, theta_x = get_extrapolation_range(input_data_unscaled[:,:8])
    input_data_unscaled = np.hstack((input_data_unscaled, x_disp.reshape(-1, 1), z_disp.reshape(-1, 1), theta_x.reshape(-1,1))) 
    
    sub_axes_inverse = input_data_unscaled[:,-2]
    main_axes_inverse = input_data_unscaled[:,-1]

    grid_x, grid_y = np.meshgrid(np.linspace(0, sub_axes_inverse*0.7, 16),
                                np.linspace(0, main_axes_inverse*0.7, 16))
    
    grid_x1, grid_y1 = np.meshgrid(np.linspace(0, sub_axes_inverse, 25),
                                np.linspace(0, main_axes_inverse, 25))

    train_X = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    train_X1 = np.column_stack([grid_x1.ravel(), grid_y1.ravel()])
    
    optimal_degree = loocv_optimization(train_X, prediction[:,:].flatten())
    poly_model, poly = polynomial_regression(train_X, prediction[:,:].flatten(), optimal_degree)

    Z_pred = predict_on_grid(poly_model, poly, train_X1)
    Z_pred = Z_pred.reshape(25, 25)
    
    # Save Z_pred, grid_x1, and grid_y1 as CSV
    output_csv_path = os.path.join(save_path, f'Stiffness_Surface.csv')
    with open(output_csv_path, 'w') as f:
        f.write('F,x,y\n')
        for i in range(25):
            for j in range(25):
                f.write(f"{Z_pred[i, j]},{grid_x1[i, j]},{grid_y1[i, j]}\n")
    print(f"Saved CSV: {output_csv_path}")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_x1, grid_y1, Z_pred, color='red', alpha=0.7, label=f'prediction')
    ax.set_xlabel('SubAxes')
    ax.set_ylabel('MainAxes')
    ax.set_zlabel('Value')
    # plt.legend()
    img_path = os.path.join(save_path, f'Stiffness_Surface.png')
    plt.savefig(img_path, dpi=300)
    print(f"Saved: {img_path}")    
#############################################

########## Model Definition ##########
class BaseMLP(nn.Module):
    def __init__(self):
        super(BaseMLP, self).__init__()

    def get_activation(self, name):
        activations = {
            "SiLU": nn.SiLU(),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "ELU": nn.ELU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Mish": nn.Mish(),
            "SeLU": nn.SELU(),
            "ReLU": nn.ReLU(),
            "ReLU6": nn.ReLU6(),
            "None": nn.Identity(),
        }
        if name in activations:
            return activations[name]
        raise ValueError(f"Invalid activation: {name}")
    
class SHCNN_(BaseMLP):
    def __init__(self, 
                num_DV=17,
                dropout_rate=0.1, 
                BN_momentum = 0.1,
                start_ch = 2048,
                embedding_dim1=1024,
                embedding_dim2=1024,
                # embedding_dim3=128,
                activation='SiLU'
                ):
        super(SHCNN_, self).__init__()

        self.padding_param = 0
        self.kernel_size = 3
        self.stride = 1
        self.embedding_dim1 = embedding_dim1
        self.embedding_dim2 = embedding_dim2
        self.start_ch = start_ch 
        self.dropout_rate = dropout_rate
        self.BN_momentum = BN_momentum
        
        seg1_dim = 8
        seg2_dim = 6
        seg3_dim = num_DV - 14

        self.embed1 = nn.Sequential(
            nn.Linear(seg1_dim, self.embedding_dim1),
            nn.BatchNorm1d(self.embedding_dim1, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.Linear(self.embedding_dim1, self.embedding_dim1),
            # nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.embed2 = nn.Sequential(
            nn.Linear(seg2_dim, self.embedding_dim2),
            nn.BatchNorm1d(self.embedding_dim2, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.Linear(self.embedding_dim2, self.embedding_dim2),
            nn.Dropout(dropout_rate)
        )
        self.embed3 = nn.Sequential(
            nn.Linear(seg3_dim, self.embedding_dim1),
            nn.BatchNorm1d(self.embedding_dim1, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.Dropout(dropout_rate)
            )

        embed_total_dim =  self.embedding_dim1 + self.embedding_dim2

        self.fc = nn.Sequential(
            nn.Linear(in_features=embed_total_dim, out_features=self.start_ch * 2 * 2),
            nn.BatchNorm1d(self.start_ch * 2 * 2, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.Linear(self.start_ch * 2 * 2, self.start_ch * 2 * 2),
            nn.Dropout(self.dropout_rate)
            
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(self.start_ch, self.start_ch // 2, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 2, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 2, self.start_ch // 2, kernel_size=self.kernel_size, 
                                 stride= self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 2, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=0, count_include_pad=False),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 2, self.start_ch // 4, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 4, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 4, self.start_ch // 4, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 4, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=0, count_include_pad=False),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 4, self.start_ch // 8, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 8, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 8, self.start_ch // 8, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 8, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=0, count_include_pad=False),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 8, self.start_ch // 16, kernel_size=3, 
                                 stride=self.stride, padding=0),
            nn.BatchNorm2d(self.start_ch // 16, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 16, self.start_ch // 16, kernel_size=3, 
                                 stride=self.stride, padding=0),
            nn.BatchNorm2d(self.start_ch // 16, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=0, count_include_pad=False),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 16, self.start_ch // 32, kernel_size=3, 
                                 stride=self.stride, padding=0),
            nn.BatchNorm2d(self.start_ch // 32, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 32, self.start_ch // 32, kernel_size=3, 
                                 stride=self.stride, padding=0),
            nn.BatchNorm2d(self.start_ch // 32, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=1,count_include_pad=False),
            nn.Dropout(dropout_rate),
        )

        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(self.start_ch // 32, 6, kernel_size=3, stride=self.stride, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=6 * 16 * 16, out_features=6 * 16 * 16),
        )

    def forward(self, input):

        seg1 = input[:, :8]      
        seg2 = input[:, 8:14]      
        # seg3 = input[:, 14:]   

        emb1 = self.embed1(seg1)   
        emb2 = self.embed2(seg2)   
        # emb3 = self.embed3(seg3)

        x_embed = torch.cat([emb1, emb2], dim=1)  

        # x_embed = emb1 + emb2 + emb3
        x = self.fc(x_embed)  
        x = x.view(-1, self.start_ch, 2, 2)
        x = self.conv5(x)
        x = self.conv_last(x).view(-1, 6, 16, 16)
        return x
    
    
    
class SHCNN():
    def __init__(
        self,
        device: str,
        num_DV: int,
        BN_momentum: float,
        dropout_rate: float,
        start_ch: int,
        embedding_dim1: int,
        embedding_dim2: int,
        activation: str = "SiLU",
    ):
        self.device = device
        
        self.hparams = {
            "num_DV": num_DV,
            "BN_momentum": BN_momentum,
            "dropout_rate": dropout_rate,
            "start_ch": start_ch,
            "embedding_dim1": embedding_dim1,
            "embedding_dim2": embedding_dim2,
            "activation": activation,
        }
        
        self.model = SHCNN_(**self.hparams).to(device)
        
        self.input_scaler_shape = None
        self.input_scaler_linear = None
        self.output_scaler = None
        
        
    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
    
    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs)
            outputs = outputs.detach().cpu().numpy()
            
            outputs_flat = outputs.reshape(-1, 6*16*16)
            # output_scaler = self.output_scaler[int(input_unscaled[:, -3])-1]
            outputs_flat = self.output_scaler.inverse_transform(outputs_flat)
            
            outputs = outputs_flat.reshape(outputs.shape)
        
        return outputs
            
    @classmethod
    def load(cls, path, device):
        hparams = json.load(open(os.path.join(path, "hparams.json"), "r"))
        model = cls(**hparams, device=device)
        model.model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
        model.input_scaler_shape = torch.load(os.path.join(path, "input_scaler_shape.pth"))
        model.input_scaler_linear = torch.load(os.path.join(path, "input_scaler_linear.pth"))
        model.output_scaler = torch.load(os.path.join(path, "output_scalers.pth"))
        
        return model
##########################################

logger = logging.getLogger(__name__)
  
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
        
    if model_type == "SHCNN":
        model = SHCNN.load(model_path, device)

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
            