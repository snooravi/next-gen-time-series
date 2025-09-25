import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.graphics.tsaplots import plot_acf
from typing import Tuple


def lineplot(data, column):
    """
    Creates a line plot to visualize a time series

    Args:
        data (pd.DataFrame): A DataFrame with a DatetimeIndex.
        column (str): The name of the column to plot.
    """

    # Plot a time series 
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[column], color='black', alpha=0.5, linewidth=1)

    # Customize the plot's appearance
    title = 'Line Plot of' + " "+ column
    plt.title(title, fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel(column, fontsize=12)
    
    # Improve layout and display the plot
    plt.tight_layout()
    plt.show()


def seasonalplot(data, column, max_val=10):
    """
    Creates a seasonal plot to visualize data distribution across months, 
    with each year represented by a separate line.

    Args:
        data (pd.DataFrame): A DataFrame with a DatetimeIndex.
        column (str): The name of the column to plot.
        max_val (numeric): represents last X years
    """

    # Create a copy of the dataframe
    df = data.copy()

    # Create month and year variables
    df['year'] = df.index.year
    df['month'] = df.index.month

    # Filter the data to the most recent 10 years
    df = df[df['year']>(df['year'].max()-10)]

    # Aggregate data to month and year
    df = df.groupby(['month', 'year']).mean().reset_index()

    # Plot the time series, with hue='year' 
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='month', y=column, hue='year', legend='full', palette='tab10')

    # Customize the plot's appearance
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
    # current_year = df['year'].max()
    # df = df[df['year'] > (current_year - 10)]

    # Create the box plot using Seaborn
    plt.figure(figsize=(10, 6))
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



def ma_plot(data, column, ma=7):
    """
    Creates a line plot with moving average overlay.

    Args:
        data (pd.DataFrame): A DataFrame with a DatetimeIndex.
        column (str): The name of the column to plot.
        ma (number): Represents the number of data points for the moving average
    """

    # Make a copy of the original data to avoid modifying it
    df = data.copy()

    # Calculate the moving averages
    df['MA_'+str(ma)] = df[column].rolling(window=ma).mean()

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df[column], label=column, color='black', alpha=0.5, linewidth=1)
    plt.plot(df['MA_'+str(ma)], label=str(ma)+'-Month MA', linestyle=':', color='g')

    # Customize the plot's appearance
    title = column + ' with ' + str(ma) + 'Month Moving Average'
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Improve layout and display the plot
    plt.tight_layout()
    plt.show()


def lowess_plot(data, column, frac=0.2):
    """
    Creates a line plot with moving average overlay.

    Args:
        data (pd.DataFrame): A DataFrame with a DatetimeIndex.
        column (str): The name of the column to plot.
        frac (float): We use a fraction of the data for the span to control smoothness
    """

    # Calculate the LOESS smooth curve
    loess_smoothed = lowess(data[column], data.index, frac=frac, it=0)
    loess_df = pd.DataFrame(loess_smoothed, columns=['date', 'loess_'+column])
    loess_df['date'] = pd.to_datetime(loess_df['date'])
    loess_df = loess_df.set_index('date')

    # Plot the original data
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[column], label='Observed Data', color='black', alpha=0.5, linewidth=1)

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
    """
    Takes a time series and decomposes it into: trend, seaonality, and residual

    Args:
        d (DecomposeResult): Time series decomposition
    """

    # Plot the decomposition `sharex=True` argument ensures the x-axes are aligned.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Plot each component on its own subplot.
    # The decomposition object has attributes for 'observed', 'trend', 'seasonal', and 'resid'.
    d.observed.plot(ax=ax1, title='Observed', color='black')
    d.trend.plot(ax=ax2, title='Trend', color='blue')
    d.seasonal.plot(ax=ax3, title='Seasonal', color='green')
    d.resid.plot(ax=ax4, title='Residual', color='red')

    # Add a main title to the figure.
    fig.suptitle('Classical Time Series Decomposition of co2', y=1.02, fontsize=16)

    # Improve layout to prevent titles and labels from overlapping.
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def resid_plot(residuals):
    """
    Takes the residuals from the time series decomposition 
    and returns time plot, histogram and ACF plot

    Args:
        residuals (Series): Time series decomposition
    """

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


def train_test_split(df: pd.DataFrame, test_size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a time series dataframe into training and testing sets.
    
    The split is performed by reserving a specified percentage of the
    *latest* data points for the test set, which is crucial for
    time series analysis to prevent look-ahead bias.

    Args:
        df (pd.DataFrame): The input time series DataFrame.
        test_size (float): The proportion of the dataset to include in the 
                           test split (e.g., 0.25 for 25%). Defaults to 0.25.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the 
                                           (train_set, test_set) DataFrames.
    
    Raises:
        ValueError: If test_size is not between 0 and 1 (exclusive).
    """
    if not (0 < test_size < 1):
        raise ValueError("test_size must be a float between 0 and 1 (exclusive).")

    # Calculate the size of the test set
    test_rows = int(len(df) * test_size)
    
    # The split point index is the start of the test set
    split_index = len(df) - test_rows
    
    # Split the DataFrame
    train_set = df.iloc[:split_index]
    test_set = df.iloc[split_index:]
    
    return train_set, test_set


def simple_forecasts(train_data, test_data, column):
    """
    Takes the training and testing set of a time series 
    and generates simple forecasts (mean, naive and seasonal naive)

    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        column (str): The name of the column to forecast
    """

    n = len(test_data)

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


def plot_forecasts(train_data, test_data, column):
    """
    Plots the training and testing set of a time series 
    along with simple forecasts (mean, naive and seasonal naive)

    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        column (str): The name of the column to forecast
    """

    mean_forecast, naive_forecast, seasonal_naive_forecast = simple_forecasts(train_data, test_data, column)

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


def calc_errors(train_data, test_data, column):
    """
    Calculates errors related to simple forecasts

    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        column (str): The name of the column to forecast
    """

    mean_forecast, naive_forecast, seasonal_naive_forecast = simple_forecasts(train_data, test_data, column)

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