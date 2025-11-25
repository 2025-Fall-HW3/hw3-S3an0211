"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        ret = self.returns[assets]

        # parameters
        look_mom = 252
        look_vol = 20
        look_ma = 200
        mom_threshold = 0.10      # 10% over past 252 days
        target_vol = 0.20         # 20% annual volatility
        ann = np.sqrt(252)

        spy = self.price[self.exclude]
        start = max(look_mom, look_vol, look_ma)

        for t in range(start, len(self.price)):
            # ===== RISK FILTER =====
            spy_window = spy.iloc[t - look_ma:t]
            if spy.iloc[t] < spy_window.mean():
                continue

            # ===== MOMENTUM & VOLATILITY =====
            mom_window = ret.iloc[t - look_mom:t]
            vol_window = ret.iloc[t - look_vol:t]

            if mom_window.isnull().values.any() or vol_window.isnull().values.any():
                continue

            momentum = (1 + mom_window).prod() - 1
            vol = vol_window.std().replace(0, np.nan)

            score = (momentum / vol).replace([np.inf, -np.inf], np.nan).fillna(0)

            # Require positive trends
            score = score[score > 0]
            if score.empty:
                continue

            # Also require momentum > threshold
            strong = score[momentum[score.index] > mom_threshold]
            if strong.empty:
                continue

            # ===== PICK TOP 2 =====
            k = min(2, len(strong))
            picks = strong.nlargest(k)
            raw_weights = picks / picks.sum()   # equal-weight within top 2

            # ===== PORTFOLIO VOLATILITY TARGETING =====
            # approximate portfolio volatility from weighted vol
            port_vol_daily = (vol[raw_weights.index] * raw_weights).sum()
            port_vol_ann = port_vol_daily * ann

            scale = target_vol / port_vol_ann
            scale = max(0, min(scale, 1.0))

            w = pd.Series(0.0, index=self.price.columns)
            for a in raw_weights.index:
                w[a] = raw_weights[a] * scale

            self.portfolio_weights.iloc[t] = w.values
 
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
