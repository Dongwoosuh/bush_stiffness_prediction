import os
import matplotlib.pyplot as plt
from source.polynomial_reg import *
from source import calculate_wmape
__all__ = ['results_extraction'] 

def results_extraction(input_data_unscaled, prediction, gt_output, pred_percentages:list, save_path:str):
    sub_axes_inverse = input_data_unscaled[:,-2]
    main_axes_inverse = input_data_unscaled[:,-1]

    grid_x, grid_y = np.meshgrid(np.linspace(0, sub_axes_inverse, 16),
                                np.linspace(0, main_axes_inverse, 16))

    train_X = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    optimal_degree = loocv_optimization(train_X, prediction[:,:].flatten())
    poly_model, poly = polynomial_regression(train_X, prediction[:,:].flatten(), optimal_degree)

    Z_pred = predict_on_grid(poly_model, poly, train_X)
    Z_pred = Z_pred.reshape(16, 16)
    gt_output = gt_output.reshape(16, 16)   
    
    wmpae_per_percent_list = []
    wmape_full_range_list = []
    for pred_percentage in pred_percentages:
        
        grid_x1, grid_y1 = np.meshgrid(np.linspace(0, sub_axes_inverse*pred_percentage, int(16*pred_percentage)),
                                    np.linspace(0, main_axes_inverse*pred_percentage, int(16*pred_percentage)))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid_x1, grid_y1, gt_output[:int(16*pred_percentage),:int(16*pred_percentage)], color='blue', alpha=0.5, label=f'Ground_Truth[{pred_percentage*100}%]')
        ax.plot_surface(grid_x1, grid_y1, Z_pred[:int(16*pred_percentage),:int(16*pred_percentage)], color='red', alpha=0.5, label=f'prediction[{pred_percentage*100}%]')
        ax.set_xlabel('SubAxes')
        ax.set_ylabel('MainAxes')
        ax.set_zlabel('Value')
        # plt.legend()
        img_path = os.path.join(save_path, f'{pred_percentage*100}%.png')
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
        img_path = os.path.join(save_path, f'100%.png')
        plt.savefig(img_path, dpi=300)
        print(f"Saved: {img_path}")
        
        wmape_per_percent = calculate_wmape(gt_output[:int(16*pred_percentage),:int(16*pred_percentage)], Z_pred[:int(16*pred_percentage),:int(16*pred_percentage)])
        wmape_full_range = calculate_wmape(gt_output, Z_pred)
        
        wmpae_per_percent_list.append(float(wmape_per_percent))
        wmape_full_range_list.append(float(wmape_full_range))
    
    return wmpae_per_percent_list, wmape_full_range_list
    