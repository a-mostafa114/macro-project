# Forecasting Inflation Using Macroeconomic Variables

## Overview
This project implements a forecasting model for inflation using macroeconomic variables. The implementation includes data preprocessing, model evaluation, and scenario-based forecasting.

## Files
- `main.py`: The main script that:
  - Reads and processes the data from an Excel file.
  - Performs seasonal adjustment.
  - Constructs the coefficient constraint matrix.
  - Evaluates models using RMSE.
  - Defines different scenarios for liquidity and sanctions.
  - Generates forecasts.
- `functions.py`: Contains helper functions used in `main.py` for data preprocessing, modeling, and evaluation, including:
  - **Data Loading & Processing**: Reads input data, handles missing values, and applies transformations.
  - **Kalman Filter Implementation**: Applies a state-space model for filtering and forecasting.
  - **RMSE Calculation**: Computes the root mean square error to assess model accuracy.
  - **Scenario Adjustment Functions**: Modifies macroeconomic variables under different conditions.
  - **Constraint Matrix Construction**: Builds the coefficient constraint matrix for inflation modeling.
  - **Forecasting Functions**: Generates inflation forecasts based on different macroeconomic scenarios.

## Key Steps
1. **Data Processing**
   - Reads inflation data and explanatory variables from an Excel file.
   - Performs seasonal and non-seasonal adjustments.
   - Organizes data into a structured format for model evaluation.

2. **Model Evaluation**
   - Uses RMSE to assess model accuracy.
   - Evaluates models using different lags and Kalman filtering approaches.

3. **Scenario-Based Forecasting**
   - Defines multiple scenarios for liquidity and sanctions.
   - Applies different shocks to macroeconomic variables.
   - Forecasts inflation under each scenario.

## Forecasting Output
The variables `um` and `sm` represent inflation forecasts under different economic scenarios. These forecasts provide insights into how macroeconomic shocks influence future inflation trends.
at the end you should embed um predictions into excel file to demonstrate excel plots.

