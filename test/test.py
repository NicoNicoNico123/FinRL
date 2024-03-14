import sys
import os

# Assuming your project's structure is mounted correctly in /workfile
project_root_abs_path = '/workfile/FinRL'
if project_root_abs_path not in sys.path:
    sys.path.insert(0, project_root_abs_path)

# print("Adjusted sys.path to include:", project_root_abs_path)
# print("\nCurrent Python sys.path:")
# for path in sys.path:
#     print(path)

from finrl.meta.data_processors.processor_pybit import PybitProcessor

def test_download_data():
    processor = PybitProcessor(
        ticker_list=[ "BTCUSDT"],  # Example ticker
        start_date="2023-01-01",  # Start date
        end_date="2023-12-31",    # End date
        time_interval="D",        # Daily interval
        indicators=['RSI']
    )

    # Assuming the processed DataFrame is stored in an attribute named `data_df` in your class
    portfolio_raw_df = processor.data_df

    print(portfolio_raw_df)

if __name__ == "__main__":
    test_download_data()

