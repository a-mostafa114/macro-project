function [su_cf_rmse, sc_cf_rmse] = cf_doan_rmse(Data, num_maxlag, scf, num_steps, var_cons)
    % Check for sufficient input arguments
    if nargin < 4
        error('Not enough input')
    end
    
    % Constants
    ins = 40; % Increment step size
    [T, nn] = size(Data); % Get dimensions of Data
    
    % Initialize output matrices for RMSE calculations
    su_cf_rmse = zeros(num_steps, nn);
    sc_cf_rmse = zeros(num_steps, nn);
    cf_r = NaN(num_steps, nn); % Temporary storage for forecasted values
    
    % Preallocate arrays to hold actual and forecasted values
    actual_values_all = zeros(ins, nn); % Store the actual values
    sc_cforecast_all = zeros(ins, nn); % Store the forecasted values

    % Loop through time series for forecasting
    for n = num_steps:ins + num_steps
        % Subset the data for the current iteration
        var_temp = Data(1:end-n, :);
        
        % Generate the forecasted values based on the scf input
        for i = 1:nn
            if scf(1, i) == 1
                cf_r(:, i) = Data(end-n+1:end-n+num_steps, i);
            end
        end
        
        % Perform forecasting based on provided inputs
        if nargin == 5
            [su_cforecast, sc_cforecast] = cf_var_doan(var_temp, num_maxlag, cf_r, var_cons);
        elseif nargin == 4
            [su_cforecast, sc_cforecast] = cf_var_doan(var_temp, num_maxlag, cf_r);
        end
        
        % Update the RMSE calculations
        su_cf_rmse = su_cf_rmse + (su_cforecast - Data(end-n+1:end-n+num_steps, :)).^2;
        sc_cf_rmse = sc_cf_rmse + (sc_cforecast - Data(end-n+1:end-n+num_steps, :)).^2;

        % Store actual values and corresponding forecasted values for the current step
        actual_values_all(n-num_steps+1, :) = Data(end-n+1:end-n+num_steps, :); % Store actual values
        sc_cforecast_all(n-num_steps+1, :) = sc_cforecast; % Store forecasted values
    end
    
    % Calculate final RMSE results
    su_cf_rmse = (su_cf_rmse ./ ins).^0.5; % RMSE for the forecast
    sc_cf_rmse = (sc_cf_rmse ./ ins).^0.5; % RMSE for the scaled forecast
    
    variable_names = {'sii', 'azad growth', 'liquidity growth', 'real gdp growth', 'BD', 'inflation'};
    % Plot actual_values_all versus sc_cforecast_all for each variable
    figure;
    for i = 1:nn
        subplot(nn, 1, i); % Create a subplot for each variable
        plot(flip(actual_values_all(:, i)), 'b-', 'DisplayName', 'Actual Values'); hold on;
        plot(flip(sc_cforecast_all(:, i)), 'r--', 'DisplayName', 'Forecasted Values');
        title([variable_names{i}]);
        xlabel('Time Steps');
        ylabel('Values');
        xticks(linspace(1, 41, 41));
        xticklabels({'1393_1' '1393_2' '1393_3' '1393_4' '1394_1' '1394_2' '1394_3' '1394_4' '1395_1' '1395_2' '1395_3' '1395_4' '1396_1' '1396_2' '1396_3' '1396_4' '1397_1' '1397_2' '1397_3' '1397_4' '1398_1' '1398_2' '1398_3' '1398_4' '1399_1' '1399_2' '1399_3' '1399_4' '1400_1' '1400_2' '1400_3' '1400_4' '1401_1' '1401_2' '1401_3' '1401_4' '1402_1' '1402_2' '1402_3' '1402_4' '1403_1'});
        legend('show');
        grid on; % Optional: Add grid for better readability
    end
end