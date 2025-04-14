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
from statsmodels.tsa.api import VAR
# from statsmodels.tsa.vector_ar.svar_model import SVAR  # if structural VAR estimation is needed

def cf_var_doan(data, num_maxlag, cf, var_cons=None):
    """
    Conditional forecast calculation based on Doan, Litterman and Sims (1984)
    
    Parameters:
      data      : NumPy array of shape (T, n) with historical data.
      num_maxlag: Integer; the number of lags in the VAR model.
      cf        : NumPy array of shape (num_steps, n). Its cells are np.nan except those 
                  positions where a conditional scenario is provided.
      var_cons  : NumPy array (n x n) containing restrictions for VAR coefficients.
                  Positions with np.nan are unconstrained. If a cell equals 0 this signifies 
                  that all lags of the corresponding variable do not affect the equation.
                  
    Returns:
      su_cforecast: Conditional forecast (unrestricted VAR version) as a DataFrame.
      sc_cforecast: Conditional forecast (restricted/SVAR version) as a DataFrame.
      
    Notes:
      - The original MATLAB code requires the IRIS toolbox functions such as `dboverlay` to overlay
        historical levels on the forecasts. Here we use a placeholder function.
      - The estimation of an SVAR with coefficient constraints is not implemented because no 
        direct analogue exists in standard Python packages.
      - The conditional forecasting routine (passing a scenario) is not natively supported in statsmodels.
      
    """
    # Determine dimensions
    T, n = data.shape
    num_steps = cf.shape[0]
    
    # If no restrictions provided, set var_cons to an n x n matrix of NaNs.
    if var_cons is None:
        var_cons = np.full((n, n), np.nan)
    
    # Check consistency of dimensions
    if cf.shape[1] != n or var_cons.shape != (n, n):
        raise ValueError("The number of columns in cf, and the shape of var_cons, must match the number of columns in data.")
    
    # Create variable names V_1, V_2, ... V_n
    var_names = [f'V_{i+1}' for i in range(n)]
    
    # Define a time index similar to MATLAB's quarterly periods
    # For example, starting from 1969-Q1 until (T + num_steps) quarters later.
    start_period = pd.Period('1969Q1', freq='Q')
    periods = T + num_steps
    time_index = pd.period_range(start=start_period, periods=periods, freq='Q')
    
    # Historical data
    hist_index = time_index[:T]
    forecast_index = time_index[T:]
    
    # Create a DataFrame for the historical data with proper variable names
    df_hist = pd.DataFrame(data, index=hist_index, columns=var_names)
    
    #%% Step 1. Estimate a simple (unrestricted) VAR model with constant
    model = VAR(df_hist)
    results = model.fit(num_maxlag, trend='c')
    
    #%% (Optional) SVAR estimation with restrictions would require additional routines.
    # In MATLAB, the code creates a constraint array for each lag (constrA)
    # and then uses estimate(..., 'A=', constrA) together with SVAR(...,'method=','chol').
    # Python's statsmodels provides SVAR but does not support placing coefficient restrictions 
    # in the same manner. One would need to implement a custom estimator here.
    #
    # Placeholder: We assume that a restricted estimation would yield an alternative set
    # of model results. For now, we use the same 'results' object for both forecasts.
    results_restricted = results  # This is only a placeholder!
    
    #%% Step 2. Unconditional forecasts (for both the unrestricted and the restricted models)
    forecast_unrestricted = results.forecast(df_hist.values[-num_maxlag:], steps=num_steps)
    # The forecasts are returned as a NumPy array; convert to DataFrame.
    fc_unrestricted = pd.DataFrame(forecast_unrestricted, index=forecast_index, columns=var_names)
    
    forecast_restricted = results_restricted.forecast(df_hist.values[-num_maxlag:], steps=num_steps)
    fc_restricted = pd.DataFrame(forecast_restricted, index=forecast_index, columns=var_names)
    
    #%% Step 3. Conditional Forecasting
    # In the MATLAB code, a scenario structure (j1) is built where each series with a non-NaN
    # column in 'cf' is replaced by the provided scenario values.
    #
    # Python does not include a built-in routine for conditional forecasting of VAR/SVAR models.
    # A custom implementation may involve iteratively solving a system of equations to impose
    # the conditional values. Here, we outline a simple adjustment based on replacing the 
    # corresponding forecast series values.
    
    # Create DataFrame for the scenario
    df_cf = pd.DataFrame(cf, index=forecast_index, columns=var_names)
    
    # The following overlay routine is a placeholder for IRIS's dboverlay.
    # For simplicity, we assume that if a conditional value is provided (not NaN),
    # we replace the model’s forecast with that value.
    def dboverlay(forecast_df, conditional_df):
        """Overlay forecast_df with non-NaN values from conditional_df."""
        overlaid = forecast_df.copy()
        for col in forecast_df.columns:
            cond_vals = conditional_df[col]
            # Replace values only where the conditional forecast is provided.
            overlaid[col] = np.where(~cond_vals.isna(), cond_vals, forecast_df[col])
        return overlaid

    # Create conditional forecasts by overlaying the scenarios on unconditional forecasts.
    fc_unrestricted_cond = dboverlay(fc_unrestricted, df_cf)
    fc_restricted_cond   = dboverlay(fc_restricted, df_cf)
    
    # The MATLAB code then extracts the last num_steps observations 
    # (here they already correspond to the forecast period)
    su_cforecast = fc_unrestricted_cond.copy()
    sc_cforecast = fc_restricted_cond.copy()

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