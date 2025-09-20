import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.graphics.tsaplots import plot_acf


def lineplot(df, column):

    # Plotting the time series
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[column], color='black', alpha=0.5, linewidth=1)

    title = 'Line Plot of' + " "+ column
    plt.title(title, fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel(column, fontsize=12)
    
    # Improve layout and display the plot
    plt.tight_layout()
    plt.show()


def seasonalplot(data, column, max_val=10):
    
    # Creating a seasonal plot
    # First, we need to add a 'year' and 'month' column
    df = data.copy()

    df['year'] = df.index.year
    df['month'] = df.index.month

    df = df[df['year']>(df['year'].max()-10)]

    df = df.groupby(['month', 'year']).mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='month', y=column, hue='year', legend='full', palette='tab10')

    title = 'Seasonal Plot of' + " "+ column
    plt.title(title, fontsize=16)
    plt.xlabel('Month')
    plt.ylabel(column, fontsize=12)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Year')
    
    # Improve layout and display the plot
    plt.tight_layout()
    plt.show()



def seasonalboxplot(data, column):
    """
    Creates a seasonal box plot to visualize data distribution across months, 
    with each year represented by a separate box.

    Args:
        data (pd.DataFrame): A DataFrame with a DatetimeIndex.
        column (str): The name of the column to plot.
    """
    # Make a copy of the original data to avoid modifying it
    df = data.copy()

    # Create 'year' and 'month' columns from the DatetimeIndex
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    # Filter the data to the most recent 10 years
    # This matches the behavior of the original function
    current_year = df['year'].max()
    # df = df[df['year'] > (current_year - 10)]

    # Set up the plot size
    plt.figure(figsize=(10, 6))
    
    # Create the box plot using Seaborn
    # The 'hue' parameter is used to create a separate box for each year within each month
    sns.boxplot(data=df, x='month', y=column, palette='tab10')
    
    # Customize the plot's appearance
    title = f'Seasonal Box Plot of {column}'
    plt.title(title, fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.xticks(range(12), 
               ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Improve layout and display the plot
    plt.tight_layout()
    plt.show()



def ma_plot(df, column, ma=7):

    data = df.copy()
    # Calculate the moving averages
    data['MA_'+str(ma)] = data[column].rolling(window=ma).mean()

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data[column], label=column, color='black', alpha=0.5, linewidth=1)
    plt.plot(data['MA_'+str(ma)], label=str(ma)+'-Month MA', linestyle=':', color='g')

    title = column + ' with ' + str(ma) + 'Month Moving Average'
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Improve layout and display the plot
    plt.tight_layout()
    plt.show()


def lowess_plot(df, column, frac=0.2):

    loess_smoothed = lowess(df[column], df.index, frac=0.2, it=0)
    loess_df = pd.DataFrame(loess_smoothed, columns=['date', 'loess_'+column])
    loess_df['date'] = pd.to_datetime(loess_df['date'])
    loess_df = loess_df.set_index('date')

    plt.figure(figsize=(10, 6))

    # Plot the original data
    plt.plot(df.index, df[column], label='Observed Data', color='black', alpha=0.5, linewidth=1)

    # Plot the LOESS smooth curve
    plt.plot(loess_df.index, loess_df['loess_'+column], label='LOESS Smooth', color='blue', linewidth=2)

    # Add plot 
    title = 'LOESS Smoothing on '+ column
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.legend(title='Series', loc='upper left')
    plt.grid(True)

    # Improve layout and display the plot
    plt.tight_layout()
    plt.show()


def decomp_plots(d):
    # plot the decomposition `sharex=True` argument ensures the x-axes are aligned.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # plot each component on its own subplot.
    # the decomposition object has attributes for 'observed', 'trend', 'seasonal', and 'resid'.
    d.observed.plot(ax=ax1, title='Observed', color='black')
    d.trend.plot(ax=ax2, title='Trend', color='blue')
    d.seasonal.plot(ax=ax3, title='Seasonal', color='green')
    d.resid.plot(ax=ax4, title='Residual', color='red')

    # add a main title to the figure.
    fig.suptitle('Classical Time Series Decomposition of co2', y=1.02, fontsize=16)

    # improve layout to prevent titles and labels from overlapping.
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # display the plot
    plt.show()


def resid_plot(residuals):
    # Create a figure with three subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Time plot of residuals
    axes[0].plot(residuals, color='black')
    axes[0].set_title('Residuals Time Plot', fontsize=14)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].grid(True)

    # Subplot 2: Histogram of residuals
    axes[1].hist(residuals, bins=15, color='gray', edgecolor='black')
    axes[1].set_title('Histogram of Residuals', fontsize=14)
    axes[1].set_xlabel('Residual Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)

    # Subplot 2: ACF plot of residuals
    plot_acf(residuals.dropna(), lags=12, title='ACF Plot of Residuals', color='black')
    # axes[1].set_xlabel('Lag', fontsize=12)
    # axes[1].set_ylabel('ACF', fontsize=12)

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()


import pandas as pd

def split_train_test(df, n=12):
    """
    Splits a DataFrame into a training and testing set.
    
    The function tries to use a training set that is 10 times the size of the test set,
    but falls back to a smaller train set if the data is too short.
    
    Args:
        df (pd.DataFrame): The DataFrame to split.
        n (int): The number of rows for the test set.
        
    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """
    # Calculate the number of rows for the training set.
    # The training set is 10 times the test set, but it can't be more than the total rows minus the test set.
    train_size = max(df.shape[0] - n, n)

    # Check if a training set of 10 * n is possible
    if df.shape[0] > (n * 30):
        train_data = df.iloc[len(df) - (30*5):-n]
        test_data = df.iloc[-n:]
    else:
        # If not, use a simple split
        train_data = df.iloc[:-n]
        test_data = df.iloc[-n:]

    return train_data, test_data


def simple_forecasts(train_data, test_data, column, n=12):

    # 3a. Mean Forecast
    mean_forecast_value = train_data[column].mean()
    mean_forecast = pd.Series([mean_forecast_value] * len(test_data), index=test_data.index)

    # 3b. Naive Forecast
    naive_forecast_value = train_data[column].iloc[-1]
    naive_forecast = pd.Series([naive_forecast_value] * len(test_data), index=test_data.index)

    # 3c. Seasonal Naive Forecast
    seasonal_naive_forecast_values = train_data[column].iloc[-n:].values
    seasonal_naive_forecast = pd.Series(
        seasonal_naive_forecast_values[:len(test_data)],
        index=test_data.index
    )
    return mean_forecast, naive_forecast, seasonal_naive_forecast


def plot_forecasts(train_data, test_data, column, n=12):

    mean_forecast, naive_forecast, seasonal_naive_forecast = simple_forecasts(train_data, test_data, column, n)

    plt.figure(figsize=(10, 6))

    # Plot the historical data
    plt.plot(train_data.index, train_data[column], label='Historical Data', color='black', alpha=0.7)

    # Plot the test data to show the actual values
    plt.plot(test_data.index, test_data[column], label='Actual Values', color='black', linestyle='--')

    # Plot the forecasts
    plt.plot(mean_forecast.index, mean_forecast, label='Mean Forecast', color='orange', linewidth=2)
    plt.plot(naive_forecast.index, naive_forecast, label='Naive Forecast', color='blue', linewidth=2)
    plt.plot(seasonal_naive_forecast.index, seasonal_naive_forecast, label='Seasonal Naive Forecast', color='green', linewidth=2)

    # Add plot labels and title
    title = 'Simple Forecasts for Monthly '+ column
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.legend(title='Forecast Method', loc='upper left')

    # Improve layout and display the plot
    plt.tight_layout()
    plt.show()


def calc_errors(train_data, test_data, column, n=12):

    mean_forecast, naive_forecast, seasonal_naive_forecast = simple_forecasts(train_data, test_data, column, n)

    # Create a dictionary to hold metrics
    metrics = {
        'Model': ['Mean', 'Naive', 'Seasonal Naive'],
        'MAE': [
            np.mean(np.abs(test_data[column] - mean_forecast)),
            np.mean(np.abs(test_data[column] - naive_forecast)),
            np.mean(np.abs(test_data[column] - seasonal_naive_forecast))
        ],
        'MSE': [
            np.mean((test_data[column] - mean_forecast)**2),
            np.mean((test_data[column] - naive_forecast)**2),
            np.mean((test_data[column] - seasonal_naive_forecast)**2)
        ],
        'Bias': [
            np.mean(test_data[column] - mean_forecast),
            np.mean(test_data[column] - naive_forecast),
            np.mean(test_data[column] - seasonal_naive_forecast)
        ]
    }
    # Create a DataFrame from the metrics dictionary
    results_df = pd.DataFrame(metrics).round(2)
    results_df = results_df.set_index('Model')

    return results_df

# def create_lag_plots(dataframe, column, max_lags=6):
#     """
#     Creates a grid of lag plots for a specified time series column.

#     Parameters:
#     - dataframe (pd.DataFrame): The DataFrame containing the time series data.
#     - column (str): The name of the column to plot.
#     - max_lags (int): The maximum number of lags to plot.
#     """
#     # Create a figure with subplots
#     fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
#     fig.suptitle(f'Lag Plots for {column}', fontsize=18, y=1.02)
    
#     # Flatten the axes array for easy iteration
#     axes = axes.flatten()

#     for i in range(1, max_lags + 1):
#         ax = axes[i-1]
        
#         # Create a new column with the lagged data
#         lagged_data = dataframe[column].shift(i)
        
#         # Plot the data
#         ax.scatter(lagged_data, dataframe[column], alpha=0.6)
        
#         # Add labels and title
#         ax.set_title(f'Lag {i}', fontsize=14)
#         ax.set_xlabel(f'{column} (t-{i})')
#         ax.set_ylabel(f'{column} (t)')
        
#         # Add a diagonal line for reference (x=y)
#         min_val = min(dataframe[column].min(), lagged_data.min())
#         max_val = max(dataframe[column].max(), lagged_data.max())
#         ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
#         ax.grid(True, linestyle='--', alpha=0.5)

#     # Remove any unused subplots
#     for j in range(len(axes)):
#         if j >= max_lags:
#             fig.delaxes(axes[j])
            
#     plt.tight_layout()
#     plt.show()