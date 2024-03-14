import gymnasium as gym
import numpy as np
from numpy import random as rd

class BybitTradingEnv(gym.Env):
    def __init__(self, config, gamma=0.99, min_stock_rate=0.1, max_stock=10, 
                 initial_cash=100000, buy_cost_pct=1e-3, sell_cost_pct=1e-3, 
                 reward_scaling=2**-11, leverage=10, initial_stocks=None):
        # environment information
        self.env_name = "StockEnv"

        self.day = 0
        self.gamma = gamma
        self.min_stock_rate = min_stock_rate
        self.max_stock = max_stock
        self.initial_cash = initial_cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.leverage = leverage
        self.THRESHOLD = self.max_stock * 0.1
        
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        if_train = config["if_train"]
        
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        stock_dim = self.price_ary.shape[1]
        
        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks
        self.stocks = self.initial_stocks.copy()  # Initialize stocks
        
        self.cash_balance = initial_cash
        self.total_asset = self.cash_balance
        self.previous_total_asset = self.total_asset
        self.position_value = 0

        self.gamma_reward = 0
        # amount + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 * stock_dim + self.tech_ary.shape[1]
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0
        
        self.observation_space = gym.spaces.Box(low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)


    def reset(self, 
              
              ):
        self.day = 0
        self.stocks = self.initial_stocks.copy()
        self.cash_balance = self.initial_cash
        self.total_asset = self.cash_balance
        self.previous_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.position_value = 0
        print(f"Resetting environment at Day {self.day}")
        print(f"Initial cash balance: {self.cash_balance}, Total asset: {self.total_asset}")

        return self.get_state(self.price_ary[0])



    def step(self, actions):
        actions = (np.array(actions) * self.max_stock).astype(int)
        self.day += 1
        profit_loss = 0
        today_prices_ary = self.price_ary[self.day]
        done = False

        for index, action in enumerate(actions):
            
            if action == 0: # Holding
                continue

            elif  action > 0:  # Buying shares, or in futures, going long
                # Determine the maximum number of contracts/shares you can buy with your current amount
                buy_num_shares = min(self.cash_balance // (today_prices_ary[index]), action)
                self.stocks[index] += buy_num_shares
                transaction_cost = today_prices_ary[index] * buy_num_shares
                self.cash_balance -= transaction_cost
                self.position_value += transaction_cost
                # print(f"After buying: Stock {index+1}, Cash Balance = {self.cash_balance}")

            elif action < 0:  # Short selling
                # Determine the maximum number of contracts/shares you can sell short with your current amount
                sell_num_shares = min(self.cash_balance // (today_prices_ary[index]), abs(action))
                self.stocks[index] -= sell_num_shares
                transaction_proceeds = today_prices_ary[index] * sell_num_shares
                self.cash_balance -= transaction_proceeds
                self.position_value += transaction_proceeds
                # print(f"After short selling: Stock {index+1}, Cash Balance = {self.cash_balance}")
                    

        for index in range(len(self.stocks)):
            price_change = today_prices_ary[index] - self.price_ary[self.day - 1][index]
            if self.stocks[index] > 0:  # Long position
                profit_loss += price_change * self.stocks[index]
            elif self.stocks[index] < 0:  # Short position, invert price change effect
                profit_loss -= price_change * abs(self.stocks[index])
                
        profit_loss *= self.leverage
        # print(f"Profit Loss after price change {profit_loss}")
        self.cash_balance += profit_loss

        # Check if cash balance falls below $10 after any trading action
        if self.cash_balance < 10:
            done = True  # Stop the episode if cash balance is too low

        # Assest Calculate
        updated_assets_value = self.cash_balance + self.position_value # Start with the current cash balance

        # Reward
        reward = (updated_assets_value - self.previous_total_asset) * self.reward_scaling

        # State
        state = self.get_state(today_prices_ary)
        
        # Update assets value
        self.previous_total_asset = updated_assets_value
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        
        # If not already terminating due to low cash, check if it's the last day
        done = self.day == self.max_step

        if done:
            reward = self.gamma_reward
            self.episode_return = updated_assets_value / self.initial_cash
            print(f"""
                    Episode finished at Day {self.day}: Cash balance: 
                    {self.cash_balance}, Total asset: {updated_assets_value}, 
                    Episode return: {self.episode_return}
                    """)

        
        print(f"Day {self.day}:")
        print(f"Actions taken: {actions}")
        print(f"Cash balance after actions: {self.cash_balance}, Position value: {self.position_value}")

        return state, reward, done,  dict()



    def get_state(self, price):
        amount_scaled = np.array([self.cash_balance * (2**-12)], dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
    (
        amount_scaled,
        price * scale,
        self.stocks * scale,
        self.tech_ary[self.day],
    ))


    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5
        return sigmoid(ary / thresh) * thresh
