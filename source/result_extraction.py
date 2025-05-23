import os
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from source.polynomial_reg import *
from source import calculate_wmape
__all__ = ['results_extraction'] 

def results_extraction(input_data_unscaled, prediction, gt_output, pred_percentages:list, save_path:str):
    
    x_disp, z_disp, theta_x = get_extrapolation_range(input_data_unscaled[:,:14])
    input_data_unscaled = np.hstack((input_data_unscaled, x_disp.reshape(-1, 1), z_disp.reshape(-1, 1), theta_x.reshape(-1,1))) 
    
    sub_axes_inverse = input_data_unscaled[:,-2]
    main_axes_inverse = input_data_unscaled[:,-1]

    grid_x, grid_y = np.meshgrid(np.linspace(-sub_axes_inverse, sub_axes_inverse, 31),
                                np.linspace(-main_axes_inverse, main_axes_inverse, 31))

    train_X = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    optimal_degree = loocv_optimization(train_X, prediction[:,:].flatten())
    poly_model, poly = polynomial_regression(train_X, prediction[:,:].flatten(), optimal_degree)

    Z_pred_original = prediction
    Z_pred_original = Z_pred_original.reshape(31, 31)
    
    Z_pred = predict_on_grid(poly_model, poly, train_X)
    Z_pred = Z_pred.reshape(31, 31)
    gt_output = gt_output.reshape(31, 31)   
    
    wmpae_per_percent_list = []
    wmape_full_range_list = []
    for pred_percentage in pred_percentages:
        
        grid_x1, grid_y1 = np.meshgrid(np.linspace(-sub_axes_inverse*pred_percentage, sub_axes_inverse*pred_percentage, int(31*pred_percentage)),
                                    np.linspace(-main_axes_inverse*pred_percentage, main_axes_inverse*pred_percentage, int(31*pred_percentage)))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid_x1, grid_y1, gt_output[:int(31*pred_percentage),:int(31*pred_percentage)], color='blue', alpha=0.5, label=f'Ground_Truth[{pred_percentage*100}%]')
        # ax.plot_surface(grid_x1, grid_y1, Z_pred[:int(31*pred_percentage),:int(31*pred_percentage)], color='red', alpha=0.5, label=f'prediction[{pred_percentage*100}%]')
        ax.set_xlabel('SubAxes')
        ax.set_ylabel('MainAxes')
        ax.set_zlabel('Value')
        # plt.legend()
        img_path = os.path.join(save_path, f'GroundTruth.png')
        plt.savefig(img_path, dpi=300)
        print(f"Saved: {img_path}")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(grid_x, grid_y, gt_output, color='blue', alpha=0.5, label='Ground_Truth[100%]')
        ax.plot_surface(grid_x, grid_y, Z_pred_original, color='red', alpha=0.5, label='Predicton[100%]')
        ax.set_xlabel('SubAxes')
        ax.set_ylabel('MainAxes')
        ax.set_zlabel('Value')
        # plt.legend()
        img_path = os.path.join(save_path, f'Prediction_original.png')
        plt.savefig(img_path, dpi=300)
        print(f"Saved: {img_path}")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(grid_x, grid_y, gt_output, color='blue', alpha=0.5, label='Ground_Truth[100%]')
        ax.plot_surface(grid_x, grid_y, Z_pred, color='red', alpha=0.5, label='Predicton[100%]')
        ax.set_xlabel('SubAxes')
        ax.set_ylabel('MainAxes')
        ax.set_zlabel('Value')
        # plt.legend()
        img_path = os.path.join(save_path, f'Prediction.png')
        plt.savefig(img_path, dpi=300)
        print(f"Saved: {img_path}")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid_x, grid_y, gt_output, color='blue', alpha=0.5, label='Ground_Truth[100%]')
        ax.plot_surface(grid_x, grid_y, Z_pred, color='red', alpha=0.5, label='Predicton[100%]')
        ax.set_xlabel('SubAxes')
        ax.set_ylabel('MainAxes')
        ax.set_zlabel('Value')
        # plt.legend()
        img_path = os.path.join(save_path, f'Both.png')
        plt.savefig(img_path, dpi=300)
        print(f"Saved: {img_path}")

        
        wmape_per_percent = calculate_wmape(gt_output[:int(31*pred_percentage),:int(31*pred_percentage)], Z_pred[:int(31*pred_percentage),:int(31*pred_percentage)])
        wmape_full_range = calculate_wmape(gt_output, Z_pred)
        
        wmpae_per_percent_list.append(float(wmape_per_percent))
        wmape_full_range_list.append(float(wmape_full_range))
    
    return wmpae_per_percent_list, wmape_full_range_list
    
    
def get_extrapolation_range(df):

    df = np.array(df, dtype=np.float64)

    scale_factor = 1.0487
    # Calculate rubber parameters
    D_O_RUBBER = 2 * (df[:, 0] + df[:, 1])
    D_I_RUBBER = 2 * df[:, 0]
    L_O_RUBBER = 2 * df[:, 2]
    L_I_RUBBER = 2 * (df[:, 2] + df[:, 3])

    Middle_thickness = df[:, 13]
    
    # Calculate displacements and angles
    x_disp = np.where(df[:,4] <= 0,
                    (D_O_RUBBER - D_I_RUBBER - Middle_thickness*2) / 2,
                    (D_O_RUBBER - D_I_RUBBER - Middle_thickness*2) / 2 - df[:,4])
    z_disp = (L_I_RUBBER*scale_factor - L_O_RUBBER) / 2
    theta_x = np.degrees(np.arctan(D_O_RUBBER / L_O_RUBBER) - np.arcsin(D_I_RUBBER / np.sqrt(D_O_RUBBER**2 + L_O_RUBBER**2)))
    
    return x_disp, z_disp, theta_x