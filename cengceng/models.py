"""
The model library
"""
import torch


class Model(object):

    """
    Docstring for Model. The basic model.
    """

    def __init__(self):
        """init the model with preferences"""
        # Use Adam as the default optimizer
        self.optimizer = torch.optim.Adam


class Sir(Model):

    """
    Sir Model
    Parameters:
    - beta
    - gama
    """

    def __init__(self, beta, gama):
        super().__init__()
        self.beta = beta
        self.gama = gama
        self.r_o = self.beta / self.gama

    def forward(self, t, y):
        si = y[0]
        ii = y[1]
        ri = y[2]
        dsdt = -self.beta * si * ii
        didt = self.beta * si * ii - self.gama * ii
        drdt = self.gama * ii
        return torch.cat((dsdt, didt, drdt))