%% 0-ADD path of IRIS_Tbx and active that

addpath D:\Users\a.mostafa\Downloads\Conditional_V1\Conditional_V1\IRIS_Tbx
irisstartup

%% 1-READ & PROCESS DATA
[Target_adj, Exp_Var_adj, CPI_Cs_adj, PPI_Cs_adj]=initial_dprocess...
    ('D:\Users\a.mostafa\Downloads\unconditional_f_V2\unconditional_f_V2\','inflation','Target_seasonal',0,'Exp_seasonal (11)');
  Data=[Target_adj Exp_Var_adj];
%reorder variable base on variables endogenity in the model
Data1=Data(:,[2 3 4 5 6 1]);
[T,nn]=size(Data1);
%% 1.1 - Deseasonalizing

[Target_adj, Exp_Var_adj, CPI_Cs_adj, PPI_Cs_adj]=initial_dprocess...
    ('D:\Users\a.mostafa\Downloads\unconditional_f_V2\unconditional_f_V2\','inflation','Target_seasonal',1,'Exp_seasonal (11)');
  Data=[Target_adj Exp_Var_adj];
%reorder variable base on variables endogenity in the model
Data2=Data(:,[2 3 4 5 6 1]);

%% 1.2 Plot
% Assuming Data1 contains original data, Data2 contains deseasonalized data
% with the respective columns: sii, azad, liquidity, real_gdp, BD/NominalGDP, inflation

% Extract columns for easier manipulation
original_data = Data1;  % Original data
%deseasonalized_data = Data2;  % Deseasonalized data

% Placeholder for seasonal component (replace with actual seasonal component data)
%seasonal_component = original_data - deseasonalized_data;  % Example of deriving seasonal component

% Define feature names for labeling
feature_names = {'SII', 'Azad', 'Liquidity', 'Real GDP', 'BD/Nominal GDP', 'Inflation'};

% Number of features
num_features = size(original_data, 2);

% Create a figure for subplots
figure;
for i = 1:num_features
    subplot(num_features, 1, i);
    % Plot original data
    plot(original_data(:, i), 'b', 'DisplayName', 'Original', 'LineWidth', 1.5);
  %  hold on;
  %  % Plot deseasonalized data
  %  plot(deseasonalized_data(:, i), 'r', 'DisplayName', 'Deseasonalized', 'LineWidth', 1.5);
  %  % Plot seasonal component
  %  plot(seasonal_component(:, i), 'g', 'DisplayName', 'Seasonal', 'LineWidth', 1.5);
  %  hold off;
    
    % Adding labels and title
    title(feature_names{i});
    xlabel('Time (Index)');
    ylabel('Value');
    legend('show');
    grid on;
end

% Ensure everything fits well in the figure
sgtitle('Comparison of Original, Deseasonalized, and Seasonal Data');

%% 2-COEFFICINT CONSTRAINT MATRIX
%%% First
var_cons1=NaN(nn,nn);
var_cons1(1,2)=0;
var_cons1(1,3)=0;
var_cons1(1,4)=0;
var_cons1(1,5)=0;
var_cons1(1,6)=0;

%% 3 Evaluating the models based on RMSE
scf1=[0 0 0 0 0 0 0];

 for zz=1:5   
[UU(:,:,zz),SS(:,:,zz)]=cf_doan_rmse(Data1,zz,scf1,1, var_cons1);

%change Data (variables definitions)
%[UU(:,:,zz),SS(:,:,zz)]=cf_doan_rmse(Data2,i,scf1,5,var_cons1);
%Kalman
%[UU(:,:,zz),SS(:,:,zz)]=cf_kalman_rmse(Data5,i,scf1,5)
 end
 
 T_rmse(:,:,1:5) = SS;
 %T_rmse(:,:,5:8) = UU;
 

%% 4 SCENARIO
num_steps=12
%for Liquidity
%cf_mb1=NaN(num_steps,nn);
%cf_mb1(:,3)=[0.1;0.1;0.1;0.1]
%cf_mb2=NaN(num_steps,nn);
%cf_mb2(:,3)=[0.15;0.15;0.15;0.15]
%cf_mb3=NaN(num_steps,nn);
%cf_mb3(:,3)=[0.05;0.05;0.05;0.05]

% Sanctions
cf_ex1=NaN(num_steps,nn);
cf_ex1(:,1)=[0.16;0.16;0.16;0.16;0.16;0.16;0.16;0.16;0.16;0.16;0.16;0.16];
cf_ex2=NaN(num_steps,nn);
cf_ex2(:,1)=[0.16;0.16;0.31;0.31;0.31;0.31;0.31;0.31;0.31;0.31;0.31;0.31];
cf_ex3=NaN(num_steps,nn);
cf_ex3(:,1)=[0.16;0.16;0.09;0.09;0.09;0.09;0.09;0.09;0.09;0.09;0.09;0.09];


%% 5
  %[um1,sm1]=cf_var_doan(Data1,4,cf_mb1,var_cons1);
  %[um2,sm2]=cf_var_doan(Data1,4,cf_mb2,var_cons1);
  %[um3,sm3]=cf_var_doan(Data1,4,cf_mb3,var_cons1);
  
  [um4,sm4]=cf_var_doan(Data1,1,cf_ex1,var_cons1);
  [um5,sm5]=cf_var_doan(Data1,1,cf_ex2,var_cons1);
  [um6,sm6]=cf_var_doan(Data1,1,cf_ex3,var_cons1);
  
  
%% 
% Assuming:
% - Data1 is your original data.
% - Data2 is your deseasonalized data (used for predictions).
% - sm4, sm5, sm6 are the predicted (deseasonalized) outputs.

% Step 1: Calculate the seasonal component
seasonal_adjustment = Data1 - Data2;  % Difference to get seasonal adjustment
num_seasons = 12; % Assuming monthly data, change accordingly for your context

% Calculate the mean seasonal component for each of the 12 periods
seasonal_component = zeros(num_seasons, size(seasonal_adjustment, 2));
for i = 1:num_seasons
    seasonal_component(i, :) = mean(seasonal_adjustment(i:num_seasons:end, :), 1);
end

% Step 2: Reconstruct original predictions by adding seasonal terms
num_steps = size(sm4, 1);  % Example: number of predicted steps

% Initialize original predictions
original_prediction_sm4 = sm4 + repmat(seasonal_component, ceil(num_steps/num_seasons), 1);
original_prediction_sm5 = sm5 + repmat(seasonal_component, ceil(num_steps/num_seasons), 1);
original_prediction_sm6 = sm6 + repmat(seasonal_component, ceil(num_steps/num_seasons), 1);

% Step 3: Plotting
feature_names = {'SII', 'Azad', 'Liquidity', 'Real GDP', 'BD/Nominal GDP', 'Inflation'};
num_features = size(Data1, 2);  % Number of features

figure;
for i = 1:num_features
    subplot(num_features, 1, i);
    hold on;
    % Plot predicted data (from Data2)
    plot(sm4(:, i), 'b', 'DisplayName', 'Predicted (s1)', 'LineWidth', 1.5);
    plot(sm5(:, i), 'g', 'DisplayName', 'Predicted (s2)', 'LineWidth', 1.5);
    plot(sm6(:, i), 'r', 'DisplayName', 'Predicted (s3)', 'LineWidth', 1.5);
    % Plot reconstructed original signals
    plot(original_prediction_sm4(:, i), 'k--', 'DisplayName', 'Prediction + seasonal term (s1)', 'LineWidth', 1.5);
    plot(original_prediction_sm5(:, i), 'm--', 'DisplayName', 'Prediction + seasonal term (s2)', 'LineWidth', 1.5);
    plot(original_prediction_sm6(:, i), 'c--', 'DisplayName', 'Prediction + seasonal term (s3)', 'LineWidth', 1.5);
    hold off;

    % Title and labels
    title(feature_names{i});
    xlabel('Time (Index)');
    ylabel('Value');
    legend('show');
    grid on;
end

% Overall figure title
sgtitle('Comparison of Predicted Signals and Reconstructed Original Signals: Stability of Sanctions (s1), Increase in Sanctions (s2), Decrease in Sanctions (s3)');