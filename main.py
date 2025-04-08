import numpy as np
import matplotlib.pyplot as plt

# 1-READ & PROCESS DATA
# First call without seasonal adjustment
Target_adj, Exp_Var_adj, CPI_Cs_adj, PPI_Cs_adj = initial_dprocess(
    r'D:\Users\a.mostafa\Downloads\unconditional_f_V2\unconditional_f_V2\\',
    'inflation.xlsx',  # Add appropriate file extension
    'Target_seasonal',
    seasonal_adjust=False,
    Explanatory_sheet='Exp_seasonal (11)'
)

# Combine data and reorder columns (MATLAB uses 1-based indexing)
Data = np.column_stack((Target_adj, Exp_Var_adj))
# Python uses 0-based indexing: [2 3 4 5 6 1] -> [1,2,3,4,5,0]
Data1 = Data[:, [1, 2, 3, 4, 5, 0]]

T, nn = Data1.shape

# Second call with seasonal adjustment
Target_adj_seasonal, Exp_Var_adj_seasonal, _, _ = initial_dprocess(
    r'D:\Users\a.mostafa\Downloads\unconditional_f_V2\unconditional_f_V2\\',
    'inflation.xlsx',
    'Target_seasonal',
    seasonal_adjust=True,
    Explanatory_sheet='Exp_seasonal (11)'
)

# Combine and reorder seasonal adjusted data
Data_seasonal = np.column_stack((Target_adj_seasonal, Exp_Var_adj_seasonal))
Data2 = Data_seasonal[:, [1, 2, 3, 4, 5, 0]]

# 1.9 Plot
original_data = Data1
deseasonalized_data = Data2
seasonal_component = original_data - deseasonalized_data

feature_names = ['SII', 'Azad', 'Liquidity', 'Real GDP', 'BD/Nominal GDP', 'Inflation']
num_features = original_data.shape[1]

# Create figure with subplots
fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(12, 18))

# Plot each feature in separate subplot
for i in range(num_features):
    ax = axes[i]
    ax.plot(original_data[:, i], 'b', label='Original', linewidth=1.5)
    # ax.plot(deseasonalized_data[:, i], 'r', label='Deseasonalized', linewidth=1.5)
    # ax.plot(seasonal_component[:, i], 'g', label='Seasonal', linewidth=1.5)
    
    ax.set_title(feature_names[i])
    ax.set_xlabel('Time (Index)')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

# Adjust layout and add main title
plt.tight_layout()
plt.subplots_adjust(top=0.95)
# fig.suptitle('Comparison of Original, Deseasonalized, and Seasonal Data', fontsize=14)
plt.show()


import numpy as np

# 2-COEFFICIENT CONSTRAINT MATRIX
# First
nn = 6  # Assuming nn is defined somewhere (number of variables)
var_cons1 = np.full((nn, nn), np.nan)
var_cons1[0, 1] = 0  # Python uses 0-based indexing
var_cons1[0, 2] = 0
var_cons1[0, 3] = 0
var_cons1[0, 4] = 0
var_cons1[0, 5] = 0

# 3 Evaluating the models based on RMSE
scf1 = np.array([0, 0, 0, 0, 0, 0, 0])  # Added 7th element as in MATLAB code

# Initialize UU and SS arrays
# Assuming Data2 is defined somewhere with shape (T, nn)
max_lags = 5
UU = np.zeros((1, nn, max_lags))  # Adjust dimensions based on cf_doan_rmse output
SS = np.zeros((1, nn, max_lags))

for zz in range(1, 6):  # zz from 1 to 5 (inclusive)
    UU[:, :, zz-1], SS[:, :, zz-1] = cf_doan_rmse(Data1, zz, scf1.reshape(1, -1), 1, var_cons1)
    
    # Alternative options commented out as in MATLAB code:
    # [UU[:, :, zz-1], SS[:, :, zz-1]] = cf_doan_rmse(Data2, zz, scf1.reshape(1, -1), 5, var_cons1)
    # Kalman version
    # [UU[:, :, zz-1], SS[:, :, zz-1]] = cf_kalman_rmse(Data5, zz, scf1.reshape(1, -1), 5)

# Initialize T_rmse array (assuming you want to store both SS and UU eventually)
T_rmse = np.zeros((1, nn, 8))  # Adjust dimensions as needed
T_rmse[:, :, :5] = SS
# T_rmse[:, :, 4:8] = UU  


import numpy as np

num_steps = 12
nn = 6  # Assuming nn is defined somewhere (number of variables)

# Liquidity
cf_mb1 = np.full((num_steps, nn), np.nan)
cf_mb1[:, 2] = [0.1] * 4 + [np.nan] * (num_steps - 4)

cf_mb2 = np.full((num_steps, nn), np.nan)
cf_mb2[:, 2] = [0.15] * 4 + [np.nan] * (num_steps - 4)

cf_mb3 = np.full((num_steps, nn), np.nan)
cf_mb3[:, 2] = [0.05] * 4 + [np.nan] * (num_steps - 4)

# Sanctions
cf_ex1 = np.full((num_steps, nn), np.nan)
cf_ex1[:, 0] = [0.16] * num_steps

cf_ex2 = np.full((num_steps, nn), np.nan)
cf_ex2[:, 0] = [0.16] * 2 + [0.31] * 10

cf_ex3 = np.full((num_steps, nn), np.nan)
cf_ex3[:, 0] = [0.16] * 2 + [0.09] * 10

# um1, sm1 = cf_var_doan(Data1, 4, cf_mb1, var_cons1)
# um2, sm2 = cf_var_doan(Data1, 4, cf_mb2, var_cons1)
# um3, sm3 = cf_var_doan(Data1, 4, cf_mb3, var_cons1)

um4, sm4 = cf_var_doan(Data1, 1, cf_ex1, var_cons1)
um5, sm5 = cf_var_doan(Data1, 1, cf_ex2, var_cons1)
um6, sm6 = cf_var_doan(Data1, 1, cf_ex3, var_cons1)