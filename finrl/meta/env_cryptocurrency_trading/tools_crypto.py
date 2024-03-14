# Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Many platforms exist for simulated trading (paper trading) which can be used for building and developing the methods discussed. Please use common sense and always first consult a professional before trading or investing.
# -----------------------------------------------------------------------------------------------------------------------------------------
# Import related modules
from __future__ import annotations

import os
import time
from copy import deepcopy

import gym
import numpy as np
import numpy.random as rd
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from finrl.config import INDICATORS
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.plot import backtest_plot
from finrl.plot import backtest_stats
from finrl.plot import get_baseline
from finrl.plot import get_daily_return
from finrl.agents.elegantrl.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


# -----------------------------------------------------------------------------------------------------------------------------------------
# Train & Test Functions

from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processors.processor_pybit import PybitProcessor

def df_to_array( df, tech_indicator_list):
        unique_ticker = df.tic.unique()
        print(unique_ticker)
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                # price_ary = df[df.tic==tic]['close'].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        print("Successfully transformed into array")
        return price_array, tech_array

# construct environment

def train(
    start_date,
    end_date,
    ticker_list,
    time_interval,
    indicators,
    env,
    model_name,
    **kwargs,
):
    # download data
    pp = PybitProcessor()
    data = pp(ticker_list, start_date, end_date, time_interval, indicators)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))
    DRLAgent_erl = DRLAgent
    break_step = kwargs.get("break_step", 1e6)
    erl_params = kwargs.get("erl_params")
    agent = DRLAgent_erl(
    env=env)
    model = agent.get_model(model_name, model_kwargs=erl_params)
    trained_model = agent.train_model(
        model=model, cwd=cwd, total_timesteps=break_step
    )


# -----------------------------------------------------------------------------------------------------------------------------------------

from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    # import data processor
    from finrl.meta.data_processor import DataProcessor

    # fetch data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)

    if if_vix:
        data = dp.add_vix(data)
    else:
        data = dp.add_turbulence(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    if drl_lib == "elegantrl":
        DRLAgent_erl = DRLAgent
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
        )
        return episode_total_assets


# -----------------------------------------------------------------------------------------------------------------------------------------

import alpaca_trade_api as tradeapi
import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from datetime import datetime as dt
from finrl.plot import backtest_stats
import matplotlib.pyplot as plt


def get_trading_days(start, end):
    nyse = tc.get_calendar("NYSE")
    df = nyse.sessions_in_range(
        pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
    )
    trading_days = []
    for day in df:
        trading_days.append(str(day)[:10])

    return trading_days



# -----------------------------------------------------------------------------------------------------------------------------------------
