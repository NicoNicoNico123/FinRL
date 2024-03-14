"""
DRL models from ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL
"""

from __future__ import annotations

import torch
from elegantrl.agents import *
from ppo import Config, train_agent

MODELS = {
    "ddpg": AgentDDPG,
    "td3": AgentTD3,
    "sac": AgentSAC,
    "ppo": AgentPPO,
    "a2c": AgentA2C,
}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]
# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
#
# NOISE = {
#     "normal": NormalActionNoise,
#     "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
# }


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array

    def get_model(self, model_name, model_kwargs):
        self.env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "if_train": True,
        }
        self.model_kwargs = model_kwargs
        self.gamma = model_kwargs.get("gamma", 0.985)

        env = self.env
        env.env_num = 1
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        stock_dim = self.price_array.shape[1]
        self.state_dim = 1 + 2 * stock_dim + self.tech_array.shape[1]
        self.action_dim = stock_dim
        self.env_args = {
            "env_name": "StockEnv",
            "config": self.env_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "if_discrete": False,
            "max_step": self.price_array.shape[0] - 1,
        }

        model = Config(agent_class=agent, env_class=env, env_args=self.env_args)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.break_step = int(
                    2e5
                )  # break training if 'total_step > break_step'
                model.net_dims = (
                    128,
                    64,
                )  # the middle layer dimension of MultiLayer Perceptron
                model.gamma = self.gamma  # discount factor of future rewards
                model.horizon_len = model.max_step
                model.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
                model.learning_rate = model_kwargs.get("learning_rate", 1e-4)
                model.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
                model.eval_times = model_kwargs.get("eval_times", 2**5)
                model.eval_per_step = int(2e4)
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        agent_class = MODELS[model_name]
        environment.env_num = 1
        agent = agent_class(net_dimension, environment.state_dim, environment.action_dim)
        actor = agent.act
        # load agent
        try:  
            cwd = cwd + '/actor.pth'
            print(f"| load actor from: {cwd}")
            actor.load_state_dict(torch.load(cwd, map_location=lambda storage, loc: storage))
            act = actor
            device = agent.device
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = environment.step(action)

                total_asset = (
                    environment.amount
                    + (
                        environment.price_ary[environment.day] * environment.stocks
                    ).sum()
                )
                episode_total_assets.append(total_asset)
                episode_return = total_asset / environment.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)
        return episode_total_assets
