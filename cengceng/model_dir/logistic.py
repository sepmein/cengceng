"""
Logistic Model
"""
# import system libraries and frameworks
import math
import os
import torch
import numpy as np
import pandas as pd


class Logistic_Model(object):

    """Docstring for Logistic_Model. """

    def __init__(self, df):
        """
        df is the dataframe of the logistic model to be fitted
        """
        self.df = df
        self.xm = torch.tensor(
            confirmed_cases.to_numpy()[:, -1].reshape(-1, 1),
            dtype=torch.float64,
            requires_grad=True,
        )

        # set x0 as the first column of the data
        self.x0 = torch.tensor(
            confirmed_cases.to_numpy()[:, 1].reshape(-1, 1),
            dtype=torch.float64,
            requires_grad=True,
        )

        # randomize ro
        self.ro_init = np.zeros((j, 1)) + 0.1
        self.ro = torch.tensor(ro_init, dtype=torch.float64, requires_grad=True)
        self.t = torch.arange(0, t_length, dtype=torch.float64).view(1, -1)

    def fit(self):
        """
        Fit the model using pytorch model
        """
        # set t, the shape of t is (1, t)
        xm = self.xm
        x0 = self.x0
        ro = self.ro
        optimization_fn = torch.optim.Adam(
            [
                # {"params": delta_t, "lr": 1e-1},
                {"params": xm, "lr": 1e-3},
                {"params": x0, "lr": 1e-3},
                {"params": ro},
            ],
            lr=1e-6,
        )
        training_steps = 10000
        # training
        for i in range(j):
            # training line by line
            for k in range(training_steps):
                optimization_fn.zero_grad()
                x_t = xm[i] / (
                    1
                    + (xm[i] / (x0[i] + 1e-9) - 1) * torch.pow(math.e, -1 * ro[i] * (t))
                )
                loss = torch.sum((x_t - true_t[i, :]).pow(2)) / t_length
                loss.backward()
                optimization_fn.step()
                if k % 999 == 0:
                    print(i, "-", k, "-", math.sqrt(loss))
            results["loss"][i] = math.sqrt(loss)

    def save(self):
        # save results into one file

        # highest speed point by the record
        xm = self.xm
        x0 = self.xo
        ro = self.ro
        t_highest_speed = torch.log(xm / x0 - 1) / ro
        # t1
        t_turning_point_1 = (torch.log(xm / x0 - 1) - 1.317) / ro
        # t2
        t_turning_point_2 = (torch.log(xm / x0 - 1) + 1.317) / ro
        results = pd.DataFrame(index=confirmed_cases.index)
        results["loss"] = 0.0
        results["xm"] = xm.detach().numpy()
        results["x0"] = x0.detach().numpy()
        results["ro"] = ro.detach().numpy()
        results["t1"] = t_turning_point_1.detach().numpy()
        results["t"] = t_highest_speed.detach().numpy()
        results["t2"] = t_turning_point_2.detach().numpy()
        results["delta_t"] = delta_t.detach().numpy()
        results.to_csv("results.csv")
