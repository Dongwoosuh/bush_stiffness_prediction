import os
import sys
import argparse
import logging
import json
import torch
import pandas as pd
import numpy as np
import tkinter as tk

import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)



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


def inference_results_extraction(input_data_unscaled, prediction, stiffness_num, save_path:str):
    
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
    
    # # Save Z_pred, grid_x1, and grid_y1 as CSV
    # output_csv_path = os.path.join(save_path, f'Stiffness_Surface.csv')
    # with open(output_csv_path, 'w') as f:
    #     f.write('F,x,y\n')
    #     for i in range(25):
    #         for j in range(25):
    #             f.write(f"{Z_pred[i, j]},{grid_x1[i, j]},{grid_y1[i, j]}\n")
    # print(f"Saved CSV: {output_csv_path}")
    
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
    
    ## 대칭이동 시키기
    # 주축기준 대칭이동
    # grid_x1, grid_y1, Z_pred를 1차원 배열로 펼침 (총 625개 점)
    grid_x1_flat = grid_x1.ravel()
    grid_y1_flat = grid_y1.ravel()
    Z_pred_flat = Z_pred.ravel()
    
    grid_x1_flat_sym = -grid_x1_flat
    grid_y1_flat_sym = grid_y1_flat
    Z_pred_flat_sym = Z_pred_flat
    # Concatenate original and symmetric points
    grid_x2 = np.concatenate((grid_x1_flat, grid_x1_flat_sym))
    grid_y2 = np.concatenate((grid_y1_flat, grid_y1_flat_sym))
    Z_pred2 = np.concatenate((Z_pred_flat, Z_pred_flat_sym))

    # Remove duplicate (x, y) points, keeping the first occurrence
    coords = np.column_stack((grid_x2, grid_y2))
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    unique_indices_sorted = np.sort(unique_indices)  # Keep order of first appearance

    grid_x2 = grid_x2[unique_indices_sorted]
    grid_y2 = grid_y2[unique_indices_sorted]
    Z_pred2 = Z_pred2[unique_indices_sorted]
    
    grid_x2_sym = grid_x2
    grid_y2_sym = -grid_y2
    Z_pred2_sym = -Z_pred2
    
    grid_x3 = np.concatenate((grid_x2, grid_x2_sym))
    grid_y3 = np.concatenate((grid_y2, grid_y2_sym))
    Z_pred3 = np.concatenate((Z_pred2, Z_pred2_sym))
    
    if stiffness_num == 1 or stiffness_num == 2 or stiffness_num == 3:
        # Save Z_pred3, grid_x3, and grid_y3 as CSV
        output_csv_path_sym = os.path.join(save_path, f'Stiffness_Surface.csv')
        with open(output_csv_path_sym, 'w') as f:
            f.write('F,x,y\n')
            for i in range(len(grid_x3)):
                f.write(f"{Z_pred3[i]},{grid_y3[i]},{grid_x3[i]}\n")
        print(f"Saved CSV: {output_csv_path_sym}")

        # Save as TXT
        output_txt_path_sym = os.path.join(save_path, f'Stiffness_Surface.txt')
        with open(output_txt_path_sym, 'w') as f:
            f.write('F,x,y\n')
            for i in range(len(grid_x3)):
                f.write(f"{Z_pred3[i]},{grid_y3[i]},{grid_x3[i]}\n")
        print(f"Saved TXT: {output_txt_path_sym}")

    elif stiffness_num == 4 or stiffness_num == 5 or stiffness_num == 6:
        # Save Z_pred3, grid_x3, and grid_y3 as CSV
        output_csv_path_sym = os.path.join(save_path, f'Stiffness_Surface.csv')
        with open(output_csv_path_sym, 'w') as f:
            f.write('F,x,y\n')
            for i in range(len(grid_x3)):
                f.write(f"{Z_pred3[i]},{grid_x3[i]},{grid_y3[i]}\n")
        print(f"Saved CSV: {output_csv_path_sym}")

        # Save as TXT
        output_txt_path_sym = os.path.join(save_path, f'Stiffness_Surface.txt')
        with open(output_txt_path_sym, 'w') as f:
            f.write('F,x,y\n')
            for i in range(len(grid_x3)):
                f.write(f"{Z_pred3[i]},{grid_x3[i]},{grid_y3[i]}\n")
        print(f"Saved TXT: {output_txt_path_sym}")
        
    # Plot the symmetric surface
    fig_sym = plt.figure()
    ax_sym = fig_sym.add_subplot(111, projection='3d')
    # Create a scatter plot for the symmetric data
    ax_sym.scatter(grid_x3, grid_y3, Z_pred3, c=Z_pred3, cmap='viridis', alpha=0.7)
    ax_sym.set_xlabel('SubAxes')
    ax_sym.set_ylabel('MainAxes')
    ax_sym.set_zlabel('Value')
    img_path_sym = os.path.join(save_path, f'Stiffness_Surface_Final.png')
    plt.savefig(img_path_sym, dpi=300)
    plt.close(fig_sym)
    print(f"Saved: {img_path_sym}")
    
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
        model.model.load_state_dict(torch.load(os.path.join(path, "model.pth"), map_location=device))
        model.input_scaler_shape = torch.load(os.path.join(path, "input_scaler_shape.pth"), weights_only=False)
        model.input_scaler_linear = torch.load(os.path.join(path, "input_scaler_linear.pth"), weights_only=False)
        model.output_scaler = torch.load(os.path.join(path, "output_scalers.pth"), weights_only=False)
        
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
            stiffness_num = idx_ + 1
            save_path = os.path.join(bush_save_path, f'stiffness_{idx_+1}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            inference_results_extraction(test_inputs_shape_unscaled, prediction[idx_], stiffness_num=stiffness_num, save_path=save_path)
            
# ---------------------- Main Function ----------------------
def main():
    # 1) 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description="Run SHCNN inference on a given CSV file.")
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to the input CSV file for inference."
    )

    args = parser.parse_args()

    # 2) 파일 존재 여부 확인
    if not os.path.isfile(args.csv_path):
        print(f"❌ 오류: 지정된 CSV 파일을 찾을 수 없습니다: {args.csv_path}")
        sys.exit(1)

    # 3) 테스트 설정
    model_type  = "SHCNN"
    model_path  = "./model/SHCNN_70_seed_2025_20250508_141201"
    result_path = "./results/Inference"

    # 만약 이 역시 인자로 받고 싶다면 위에서 parser에 정의한 args.model_path, args.result_path를 사용하세요.

    # 4) 데이터셋 및 테스트 실행
    dataset = InferenceVEPDataset(csv_path=args.csv_path)
    model_test(
        model_type=model_type,
        dataset=dataset,
        model_path=model_path,
        result_path=result_path
    )

if __name__ == "__main__":
    main()