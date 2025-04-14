import matlab.engine
import numpy as np
import os
import pandas as pd

def run_matlab_code():
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    try:
        # 0-ADD path of IRIS_Tbx and activate it (using new recommended method)
        eng.addpath(r'IRIS_Tbx/', nargout=0)
        # Use the new recommended startup method
        eng.eval("iris.startup()", nargout=0)
        
        # 1-READ & PROCESS DATA
        data_dir = r'D:\\Users\\a.mostafa\\GIT\\macro-project\\'
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Convert data to double precision before passing to MATLAB
        def ensure_double(data):
            if isinstance(data, np.ndarray):
                return matlab.double(data.astype(float).tolist())
            return data
        
        # Call initial_dprocess
        Target_adj, Exp_Var_adj, CPI_Cs_adj, PPI_Cs_adj = eng.initial_dprocess(
            data_dir,
            'data',
            'Target_seasonal',
            0,
            'Exp_seasonal (11)',
            nargout=4)
        
        # Convert to numpy arrays and ensure float type
        Target_adj = np.array(Target_adj, dtype=float)
        Exp_Var_adj = np.array(Exp_Var_adj, dtype=float)
        
        Data = np.column_stack((Target_adj, Exp_Var_adj))
        Data1 = Data[:, [1, 2, 3, 4, 5, 0]]
        T, nn = Data1.shape
        
        # 1.1 - Deseasonalizing
        Target_adj, Exp_Var_adj, CPI_Cs_adj, PPI_Cs_adj = eng.initial_dprocess(
            data_dir,
            'data',
            'Target_seasonal',
            1,
            'Exp_seasonal (11)',
            nargout=4)
        
        Target_adj = np.array(Target_adj, dtype=float)
        Exp_Var_adj = np.array(Exp_Var_adj, dtype=float)
        
        Data = np.column_stack((Target_adj, Exp_Var_adj))
        Data2 = Data[:, [1, 2, 3, 4, 5, 0]]
        
        # 2-COEFFICIENT CONSTRAINT MATRIX
        var_cons1 = np.full((nn, nn), np.nan, dtype=float)
        var_cons1[0, 1] = 0
        var_cons1[0, 2] = 0
        var_cons1[0, 3] = 0
        var_cons1[0, 4] = 0
        var_cons1[0, 5] = 0
        
        # Sanctions
        num_steps = 12
        cf_ex1 = np.full((num_steps, nn), np.nan, dtype=float)
        cf_ex1[:, 0] = [0.16]*12
        
        cf_ex2 = np.full((num_steps, nn), np.nan, dtype=float)
        cf_ex2[:, 0] = [0.16, 0.16] + [0.31]*10
        
        cf_ex3 = np.full((num_steps, nn), np.nan, dtype=float)
        cf_ex3[:, 0] = [0.16, 0.16] + [0.09]*10
        
        # Convert to MATLAB format with explicit double type
        Data2_mat = ensure_double(Data2)
        var_cons1_mat = ensure_double(var_cons1)
        cf_ex1_mat = ensure_double(cf_ex1)
        cf_ex2_mat = ensure_double(cf_ex2)
        cf_ex3_mat = ensure_double(cf_ex3)
        
        # Call MATLAB functions - ensure the lag parameter is a double
        lag = matlab.double([1])  # Convert lag parameter to MATLAB double
        um4, sm4 = eng.cf_var_doan(Data2_mat, lag, cf_ex1_mat, var_cons1_mat, nargout=2)
        um5, sm5 = eng.cf_var_doan(Data2_mat, lag, cf_ex2_mat, var_cons1_mat, nargout=2)
        um6, sm6 = eng.cf_var_doan(Data2_mat, lag, cf_ex3_mat, var_cons1_mat, nargout=2)
        
        # Convert results to numpy
        sm4 = np.array(sm4)
        sm5 = np.array(sm5)
        sm6 = np.array(sm6)
        
        # Seasonal adjustment calculation
        seasonal_adjustment = Data1 - Data2
        num_seasons = 12
        
        seasonal_component = np.zeros((num_seasons, seasonal_adjustment.shape[1]))
        for i in range(num_seasons):
            seasonal_component[i, :] = np.mean(seasonal_adjustment[i::num_seasons, :], axis=0)
        
        num_steps = sm4.shape[0]
        rep_times = int(np.ceil(num_steps / num_seasons))
        tiled_seasonal = np.tile(seasonal_component, (rep_times, 1))[:num_steps, :]
        
        original_prediction_sm4 = sm4 + tiled_seasonal
        original_prediction_sm5 = sm5 + tiled_seasonal
        original_prediction_sm6 = sm6 + tiled_seasonal
        
        return original_prediction_sm4, original_prediction_sm5, original_prediction_sm6
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        # Close MATLAB engine
        eng.quit()

if __name__ == "__main__":
    results = run_matlab_code()
    print("Script completed successfully")
    print("Results shapes:", [r.shape for r in results])

    from openpyxl import load_workbook
    # Load your Excel file
    file_path = "results deseasonalized and P2P from 1380 with BD over nominal GDP - 20250414 - lag4.xlsx"
    book = load_workbook(file_path)

    # Mapping sheets to `um` variables
    sheet_um_map = {
        "ثبات تحریم": results[0],
        "افزایش تحریم": results[1],
        "کاهش تحریم": results[2]
    }

    # For each sheet, update the values
    for sheet_name, um in sheet_um_map.items():
        # Convert um to numpy array if needed and drop first column
        um_data = np.array(um)[:, 1:]  # Drop the first column

        # Read the sheet into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Replace columns C:G (i.e., 2 to 6) in the last 12 rows
        df.iloc[-12:, 2:7] = um_data

        # Write back to the sheet
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("✅ Sheets updated successfully.")