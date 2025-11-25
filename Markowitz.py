"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
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

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust = False)
    df[asset] = raw['Adj Close']

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""


class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 1 Below
        """

        # number of investable assets (exclude SPY)
        n_assets = len(assets)
        if n_assets > 0:
            # equal weight for each non-excluded asset
            equal_weight = 1.0 / n_assets

            # assign equal weights for all dates
            self.portfolio_weights.loc[:, assets] = equal_weight

        # excluded asset (e.g., SPY) always has weight 0
        if self.exclude in self.portfolio_weights.columns:
            self.portfolio_weights.loc[:, self.exclude] = 0.0

        """
        TODO: Complete Task 1 Above
        """
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""


class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 2 Below
        """

        # loop over time, starting after we have enough lookback data
        for i in range(self.lookback + 1, len(df)):
            # past "lookback" days of returns for non-excluded assets
            window_returns = df_returns[assets].iloc[i - self.lookback : i]

            # compute volatilities (standard deviation)
            vol = window_returns.std()

            # avoid division by zero
            vol_replaced = vol.replace(0, np.nan)
            inv_vol = 1.0 / vol_replaced

            # if all vols are zero (very unlikely), fall back to equal weights
            if np.isfinite(inv_vol).sum() == 0:
                weights = np.ones(len(assets)) / len(assets)
            else:
                inv_vol = inv_vol.fillna(0.0)
                weights = inv_vol / inv_vol.sum()

            # assign weights for this date
            self.portfolio_weights.loc[df.index[i], assets] = weights.values

        # excluded asset always has weight 0
        if self.exclude in self.portfolio_weights.columns:
            self.portfolio_weights.loc[:, self.exclude] = 0.0

        """
        TODO: Complete Task 2 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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


"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]
            self.portfolio_weights.loc[df.index[i], assets] = self.mv_opt(
                R_n, self.gamma
            )

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                """
                TODO: Complete Task 3 Below
                """
                # decision variable: portfolio weights w_i >= 0
                w = model.addMVar(n, lb=0.0, name="w")

                # budget constraint: sum_i w_i = 1
                model.addConstr(w.sum() == 1.0, name="budget")

                # ---- build quadratic risk term w^T Σ w manually ----
                risk_term = 0
                for i in range(n):
                    for j in range(n):
                        coeff = Sigma[i, j]
                        if coeff != 0:
                            # Gurobi overloads * so this creates a QuadExpr term
                            risk_term += coeff * w[i] * w[j]
                # ----------------------------------------------------

                # objective: maximize w^T µ - (γ/2) * w^T Σ w
                obj = mu @ w - (gamma / 2.0) * risk_term
                model.setObjective(obj, gp.GRB.MAXIMIZE)
                """
                TODO: Complete Task 3 Above
                """
                model.optimize()

                # Check if the status is INF_OR_UNBD (code 4)
                if model.status == gp.GRB.INF_OR_UNBD:
                    print(
                        "Model status is INF_OR_UNBD. Reoptimizing with DualReductions set to 0."
                    )
                elif model.status == gp.GRB.INFEASIBLE:
                    # Handle infeasible model
                    print("Model is infeasible.")
                elif model.status == gp.GRB.INF_OR_UNBD:
                    # Handle infeasible or unbounded model
                    print("Model is infeasible or unbounded.")

                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    # Extract the solution
                    solution = []
                    for i in range(n):
                        var = model.getVarByName(f"w[{i}]")
                        # print(f"w {i} = {var.X}")
                        solution.append(var.X)

        return solution

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )
    """
    NOTE: For Assignment Judge
    """
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

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader.py
    judge.run_grading(args)
