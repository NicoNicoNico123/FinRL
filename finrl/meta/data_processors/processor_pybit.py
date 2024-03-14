"""Reference: https://github.com/AI4Finance-LLC/FinRL"""

from __future__ import annotations

from datetime import datetime
import exchange_calendars as tc
import numpy as np
import pandas as pd
import talib
from pybit.unified_trading import HTTP

class PybitProcessor:
    """Provides methods for retrieving daily stock data from
    Bybit API
    For more info please nav https://bybit-exchange.github.io/docs/v5/intro
    Kline interval: 1,3,5,15,30,60,120,240,360,720,D,M,W
    """
    def __init__(self, 
                 api_key="JQ13dLjcZQvVIw1As3",  # Set your default API key
                 api_secret="NSmW06HEkZBTUD1DQEDv88iFPhwBRVsUXL7X",  # Set your default API secret
                 testnet=True, 
                 ticker_list=None, 
                 start_date=None, 
                 end_date=None, 
                 time_interval=None, 
                 indicators=None):
        self.session = HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)
        
        if ticker_list and start_date and end_date and time_interval:
            self.data_df = self.download_and_apply_technical_indicators(
                ticker_list, start_date, end_date, time_interval, indicators or [])
        else:
            self.data_df = pd.DataFrame()

    # def fetch_and_process_data(self, ticker_list, start_date, end_date, time_interval, indicators):
    #     # This method encapsulates the fetching and processing of data.
    #     # Convert date strings to timestamps in milliseconds
    #     start_timestamp = self.date_to_timestamp(start_date)
    #     end_timestamp = self.date_to_timestamp(end_date)

    #     data_df = pd.DataFrame()
        
    #     for tic in ticker_list:
    #         # Use Bybit API to fetch kline data
    #         raw_data = self.session.get_kline(
    #             symbol=tic,
    #             interval=time_interval,
    #             start=start_timestamp,
    #             end=end_timestamp,
    #             limit = 1000
    #         )
            
    #         if 'result' in raw_data and 'list' in raw_data['result']:
    #             # Create DataFrame from the list of data
    #             columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    #             temp_df = pd.DataFrame(raw_data['result']['list'], columns=columns)
                
    #             if not temp_df.empty:
    #                 # Convert 'Timestamp' from string UNIX (milliseconds) to datetime
    #                 temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'].astype(int), unit='ms')
                    
    #                 # Assign ticker symbol to 'tic' column
    #                 temp_df['tic'] = tic
                    
    #                 # Append to the main DataFrame
    #                 data_df = pd.concat([data_df, temp_df], ignore_index=True)

    #     # Process indicators if any are specified
    #     if indicators:
    #         self.apply_indicators(data_df, indicators)

    #     self.data_df = data_df  # Store the processed DataFrame in the instance attribute

    def apply_indicators(self, data_df, indicators):
        # Define a mapping from indicator names to their corresponding methods
        indicator_methods = {
            'RSI': self.RSI,
            'OBV': self.OBV,
            'MACD': self.MACD
        }

        selected_indicators = [indicator_methods[name] for name in indicators if name in indicator_methods]

        for indicator in selected_indicators:
            data_df = indicator(data_df)  # Apply each indicator

        self.data_df = data_df  # Update the instance attribute with the new DataFrame

    @staticmethod
    def date_to_timestamp(date_str, date_format='%Y-%m-%d', to_milliseconds=True):
        dt = datetime.strptime(date_str, date_format)
        timestamp = datetime.timestamp(dt)
        return int(timestamp * 1000) if to_milliseconds else int(timestamp)

    def download_data(
        self,
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        time_interval: str
    ) -> pd.DataFrame:
        # Convert date strings to timestamps in milliseconds
        start_timestamp = PybitProcessor.date_to_timestamp(start_date)
        end_timestamp = PybitProcessor.date_to_timestamp(end_date)
        
        data_df = pd.DataFrame()
        
        for tic in ticker_list:
            # Use Bybit API to fetch kline data
            raw_data = self.session.get_kline(
                symbol=tic,
                interval=time_interval,  # Assuming interval is already in Bybit format
                start = start_timestamp,
                end = end_timestamp,
                limit = 1000
            )
            
            if 'result' in raw_data and 'list' in raw_data['result']:
                # Create DataFrame from the list of data
                columns = ['Timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                temp_df = pd.DataFrame(raw_data['result']['list'], columns=columns)
                
                if not temp_df.empty:
                    # Convert 'Timestamp' from string UNIX (milliseconds) to datetime
                    temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'].astype(int), unit='ms')
                    
                    # Assign ticker symbol to 'tic' column
                    temp_df['tic'] = tic
                    
                    # Append to the main DataFrame
                    data_df = pd.concat([data_df, temp_df], ignore_index=True)
        
        # Convert columns to numeric types as appropriate
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        data_df[numeric_cols] = data_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        return data_df

    # def generate_custom_data(self, data_df):
    #     # Extract unique timestamps from data_df
    #     unique_timestamps = data_df['Timestamp'].unique()
        
    #     custom_data_rows = []
    #     for timestamp in unique_timestamps:
    #         custom_row = {
    #             'Timestamp': timestamp,
    #             'open': 1,
    #             'high': 1,
    #             'low': 1,
    #             'close': 1,
    #             'volume': 0,
    #             'turnover': 0,
    #             'tic': 'USD'
    #         }
    #         custom_data_rows.append(custom_row)
        
    #     # Create a new DataFrame with the custom data rows
    #     custom_data_df = pd.DataFrame(custom_data_rows)
        
    #     # Concatenate the original data_df with the custom_data_df to append the new rows
    #     combined_df = pd.concat([data_df, custom_data_df], ignore_index=True)
        
    #     return combined_df


    def RSI(self, data_df, column='close', period=14):
        # Ensure the column data is converted to float64 before passing to the RSI function
        real = data_df[column].values.astype('float64')  # Convert to float64
        data_df['rsi'] = talib.RSI(real, timeperiod=period)
        return data_df


    
    def OBV(self, data_df, price='close', volume='Volume'):
        # Ensure 'price' and 'volume' are column names in data_df, and calculate OBV
        data_df['obv'] = talib.OBV(data_df[price].values, data_df[volume].values)
        return data_df

    def ATR(self, data_df, price='close', volume='Volume'):
        # Ensure 'price' and 'volume' are column names in data_df, and calculate OBV
        data_df['obv'] = talib.OBV(data_df[price].values, data_df[volume].values)
        return data_df


    def MACD(data_df, column='close', fastperiod=12, slowperiod=26, signalperiod=9):
        macd, macdsignal, macdhist = talib.MACD(data_df[column].values, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        data_df['macd'] = macd
        data_df['macd_signal'] = macdsignal
        data_df['macd_hist'] = macdhist
        return data_df
    
    def add_technical_indicators(self, data_df, indicators):
        # print("Before applying indicators:", data_df.columns)  # Debugging aid
        for indicator in indicators:
            data_df = indicator(data_df)
            # print(f"After applying {indicator.__name__}:", data_df.columns)  # Debugging aid
        return data_df


    def download_and_apply_technical_indicators(
            self, 
            ticker_list, 
            start_date, 
            end_date, 
            time_interval,
            indicator_names
            ):
        """
        Downloads market data for the specified tickers and time range, and applies the given technical indicators.

        Parameters:
        - ticker_list (list[str]): List of ticker symbols to download data for.
        - start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        - end_date (str): End date for the data in 'YYYY-MM-DD' format.
        - time_interval (str): Time interval for the data.
        - indicator_names (list[str]): List of technical indicator names as strings to apply to the DataFrame.

        Returns:
        pd.DataFrame: DataFrame with the market data and added technical indicators.
        """

        # Download the data for the specified tickers within the given time frame.
        data_df = self.download_data(ticker_list, start_date, end_date, time_interval)

        # # Assuming the need to generate custom data based on the unique timestamps of data_df
        # data_with_custom_data = self.generate_custom_data(data_df)

        # # correct oreder
        # data_with_custom_data.sort_values(by='Timestamp', ascending=True, inplace=True)

        # Define a mapping from indicator names to their corresponding methods
        indicator_methods = {
            'RSI': self.RSI,
            'OBV': self.OBV,
            'MACD': self.MACD
        }

        # Filter the list of indicators to those specified by the user
        selected_indicators = [indicator_methods[name] for name in indicator_names if name in indicator_methods]

        # Apply the selected technical indicators to the downloaded data
        data_with_indicators = self.add_technical_indicators(data_df, selected_indicators)

        return data_with_indicators
