import numpy as np

def transformation2(Dataset, Tr):
    """
    Applies specified transformations to each column of the dataset based on transformation codes.

    Parameters:
    Dataset (numpy.ndarray): Input data matrix (T x N).
    Tr (list or numpy.ndarray): Transformation codes for each column (0 to 5).

    Returns:
    numpy.ndarray: Transformed data matrix.
    """
    T, N = Dataset.shape
    data_trans = np.full((T, N), np.nan)
    s_data = np.zeros(N, dtype=int)

    for i in range(N):
        data_col = Dataset[:, i]
        tr = Tr[i]

        if tr == 0:
            data_trans[:, i] = data_col
            s_data[i] = 0
        elif tr == 1:
            data_trans[:, i] = np.log(data_col)
            s_data[i] = 0
        elif tr == 2:
            diff = np.diff(data_col, 1)
            data_trans[1:, i] = diff / data_col[:-1]
            s_data[i] = 1
        elif tr == 3:
            diff2 = np.diff(data_col, 2)
            data_trans[2:, i] = diff2
            s_data[i] = 2
        elif tr == 4:
            log_diff = np.diff(np.log(data_col))
            data_trans[1:, i] = log_diff
            s_data[i] = 1
        elif tr == 5:
            log_diff2 = np.diff(np.log(data_col), 2)
            data_trans[2:, i] = log_diff2
            s_data[i] = 2
        else:
            data_trans[:, i] = data_col
            s_data[i] = 0

    max_s = np.max(s_data)
    data_trans = data_trans[max_s:, :]

    return data_trans


import os
import pandas as pd
import numpy as np
from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.tsa.seasonal import seasonal_decompose

def initial_dprocess(file_address, xls_name, Target_sheet, seasonal_adjust,
                     Explanatory_sheet=None, CPI_comp=None, PPI_comp=None):
    """
    Processes data from Excel sheets, applies seasonal adjustment and transformations.

    Parameters:
    file_address (str): Directory path of the Excel file.
    xls_name (str): Excel file name.
    Target_sheet (str): Name of the sheet containing the target variable.
    seasonal_adjust (bool): Whether to apply seasonal adjustment.
    Explanatory_sheet (str, optional): Name of the sheet with explanatory variables.
    CPI_comp (str, optional): Name of the sheet with CPI components.
    PPI_comp (str, optional): Name of the sheet with PPI components.

    Returns:
    tuple: Adjusted target, explanatory variables, CPI components, and PPI components.
    """
    # Validate inputs
    if not all([file_address, xls_name, Target_sheet]):
        raise ValueError("Error: Insufficient input arguments.")

    file_path = os.path.join(file_address, xls_name)

    # Function to read sheet data
    def read_sheet(sheet_name):
        if sheet_name is None:
            return None, None, 0
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        Tr = df.iloc[1, 1:].values.astype(int)  # Transformation codes
        data = df.iloc[2:, 1:].values.astype(float)  # Actual data starting from row 2
        return Tr, data, data.shape[1]

    # Read Target sheet
    Tr_target, data_target, N1 = read_sheet(Target_sheet)
    if data_target is None:
        raise ValueError("Error: Target Variable is empty.")
    data_target = data_target.reshape(-1, N1)  # Ensure 2D array
    T = data_target.shape[0]

    # Initialize data parts and transformation codes
    Tr = Tr_target.tolist() if N1 == 1 else []
    data_parts = [data_target]

    # Read other sheets
    sheets = [
        (Explanatory_sheet, 'Explanatory'),
        (CPI_comp, 'CPI_comp'),
        (PPI_comp, 'PPI_comp')
    ]
    sizes = []
    for sheet, name in sheets:
        Tr_sheet, data_sheet, cols = read_sheet(sheet)
        if data_sheet is not None:
            if data_sheet.shape[0] != T:
                raise ValueError("Error: Mismatch in time dimensions.")
            Tr.extend(Tr_sheet)
            data_parts.append(data_sheet)
            sizes.append(cols)
        else:
            sizes.append(0)

    N2, N3, N4 = sizes

    # Combine all data
    data = np.hstack(data_parts)

    # Seasonal adjustment
    if seasonal_adjust:
        data_adj = np.full_like(data, np.nan)
        for i in range(data.shape[1]):
            try:
                ts = pd.Series(data[:, i])
                # res = x13_arima_analysis(ts)
                # data_adj[:, i] = res.seasadj
                res = seasonal_decompose(ts, model='additive', period=12)  # period=12 for monthly data
                data_adj[:, i] = res.trend + res.resid
            except Exception as e:
                raise RuntimeError(f"Seasonal adjustment failed for column {i}: {e}")
        data = data_adj

    # Apply transformations
    Tr_array = np.array(Tr)
    data_trans = transformation2(data, Tr_array)

    # Split into components
    Target_adj = data_trans[:, 0]
    Exp_Var_adj = data_trans[:, 1:1+N2] if N2 > 0 else np.array([])
    CPI_Cs_adj = data_trans[:, 1+N2:1+N2+N3] if N3 > 0 else np.array([])
    PPI_Cs_adj = data_trans[:, 1+N2+N3:1+N2+N3+N4] if N4 > 0 else np.array([])

    return Target_adj, Exp_Var_adj, CPI_Cs_adj, PPI_Cs_adj




import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR, SVAR
from statsmodels.tsa.base.datetools import dates_from_str
from scipy.linalg import cholesky

def cf_var_doan(data, num_maxlag, cf, var_cons=None):
    # data: T x nn numpy array
    # cf: num_step x nn numpy array with NaNs for unrestricted periods
    # var_cons: nn x nn matrix with constraints (NaN for unconstrained)
    T, nn = data.shape
    num_step, nn3 = cf.shape

    if var_cons is None:
        var_cons = np.full((nn, nn), np.nan)
    else:
        nn2, nn1 = var_cons.shape
        if nn != nn1 or nn != nn2 or nn != nn3:
            raise ValueError("Dimensions of data, cf, and var_cons must match")
    
    # Create DataFrame with quarterly dates starting from 1969Q1
    dates = pd.period_range(start='1969Q1', periods=T, freq='Q')
    df = pd.DataFrame(data, index=dates, columns=[f'V_{i+1}' for i in range(nn)])
    
    # Estimate unrestricted VAR
    model_unrestricted = VAR(df)
    results_unrestricted = model_unrestricted.fit(num_maxlag, trend='c')
    
    # Estimate restricted VAR
    # This part is simplified and assumes constraints are to exclude variables
    # For more complex constraints, a custom approach is needed
    restricted_data = df.copy()
    # Placeholder for constrained estimation; replace with actual constrained estimation logic
    model_restricted = VAR(restricted_data)
    results_restricted = model_restricted.fit(num_maxlag, trend='c')  # This doesn't apply constraints
    
    # Structural VAR with Cholesky decomposition
    # Unrestricted SVAR
    sigma_u = results_unrestricted.sigma_u
    B0 = cholesky(sigma_u, lower=True)
    
    # Restricted SVAR (placeholder, assuming same covariance)
    sigma_r = results_restricted.sigma_u
    B1 = cholesky(sigma_r, lower=True)
    
    # Forecast preparation
    start_forecast = df.index[-1] + 1  # Next period
    forecast_steps = num_step
    
    # Unconditional forecast
    u0_forecast = results_unrestricted.forecast(df.values[-num_maxlag:], forecast_steps)
    u1_forecast = results_restricted.forecast(df.values[-num_maxlag:], forecast_steps)
    
    # Conditional forecast (simplified placeholder)
    # This part requires solving for shocks that meet the conditions, which is non-trivial
    # Here, we'll return the unconditional forecasts as placeholders
    su_cforecast = u0_forecast
    sc_cforecast = u1_forecast
    
    return su_cforecast, sc_cforecast

# Example usage:
# data = np.random.randn(100, 3)  # 100 periods, 3 variables
# num_maxlag = 2
# cf = np.full((4, 3), np.nan)  # 4-step forecast, 3 variables
# cf[0, 0] = 1.0  # Example condition
# var_cons = np.full((3, 3), np.nan)
# su, sc = cf_var_doan(data, num_maxlag, cf, var_cons)



import numpy as np
import matplotlib.pyplot as plt

def cf_doan_rmse(Data, num_maxlag, scf, num_steps, var_cons=None):
    # Check for sufficient input arguments
    if len(Data.shape) != 2:
        raise ValueError("Data should be a 2D array")
    
    # Constants
    ins = 40  # Increment step size
    T, nn = Data.shape  # Get dimensions of Data
    
    # Initialize output matrices for RMSE calculations
    su_cf_rmse = np.zeros((num_steps, nn))
    sc_cf_rmse = np.zeros((num_steps, nn))
    cf_r = np.full((num_steps, nn), np.nan)  # Temporary storage for forecasted values
    
    # Preallocate arrays to hold actual and forecasted values
    actual_values_all = np.zeros((ins, nn))  # Store the actual values
    sc_cforecast_all = np.zeros((ins, nn))   # Store the forecasted values

    # Loop through time series for forecasting
    for n in range(num_steps, ins + num_steps):
        # Subset the data for the current iteration
        var_temp = Data[:T-n, :]
        
        # Generate the forecasted values based on the scf input
        for i in range(nn):
            if scf[0, i] == 1:
                cf_r[:, i] = Data[T-n:T-n+num_steps, i]
        
        # Perform forecasting based on provided inputs
        if var_cons is not None:
            su_cforecast, sc_cforecast = cf_var_doan(var_temp, num_maxlag, cf_r, var_cons)
        else:
            su_cforecast, sc_cforecast = cf_var_doan(var_temp, num_maxlag, cf_r)
        
        # Update the RMSE calculations
        su_cf_rmse += (su_cforecast - Data[T-n:T-n+num_steps, :]) ** 2
        sc_cf_rmse += (sc_cforecast - Data[T-n:T-n+num_steps, :]) ** 2

        # Store actual values and corresponding forecasted values for the current step
        actual_values_all[n-num_steps, :] = Data[T-n:T-n+num_steps, :].flatten()
        sc_cforecast_all[n-num_steps, :] = sc_cforecast.flatten()
    
    # Calculate final RMSE results
    su_cf_rmse = np.sqrt(su_cf_rmse / ins)
    sc_cf_rmse = np.sqrt(sc_cf_rmse / ins)
    
    variable_names = ['sii', 'azad growth', 'liquidity growth', 'real gdp growth', 'BD', 'inflation']
    
    # # Plot actual_values_all versus sc_cforecast_all for each variable
    # plt.figure(figsize=(10, 15))
    # for i in range(nn):
    #     plt.subplot(nn, 1, i+1)  # Create a subplot for each variable
    #     plt.plot(np.flip(actual_values_all[:, i]), 'b-', label='Actual Values')
    #     plt.plot(np.flip(sc_cforecast_all[:, i]), 'r--', label='Forecasted Values')
    #     plt.title(variable_names[i])
    #     plt.xlabel('Time Steps')
    #     plt.ylabel('Values')
    #     plt.xticks(np.linspace(0, 40, 41),
    #               ['1393_1', '1393_2', '1393_3', '1393_4', '1394_1', '1394_2', '1394_3', '1394_4',
    #                '1395_1', '1395_2', '1395_3', '1395_4', '1396_1', '1396_2', '1396_3', '1396_4',
    #                '1397_1', '1397_2', '1397_3', '1397_4', '1398_1', '1398_2', '1398_3', '1398_4',
    #                '1399_1', '1399_2', '1399_3', '1399_4', '1400_1', '1400_2', '1400_3', '1400_4',
    #                '1401_1', '1401_2', '1401_3', '1401_4', '1402_1', '1402_2', '1402_3', '1402_4',
    #                '1403_1'], rotation=45)
    #     plt.legend()
    #     plt.grid(True)
    
    # plt.tight_layout()
    # plt.show()
    
    return su_cf_rmse, sc_cf_rmse