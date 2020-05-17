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

    def fit(self, optimizer=torch.optim.Adam, learning_rate=0.001):
        pass


class Compartmental(Model):

    """Docstring for Compartmental. """

    def __init__(self):
        """TODO: to be defined. """
        super().__init__()

    def forward(self):
        pass


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


class Seir(Sir):
    """
    Inherited from sir model
    """

    def __init__(self, lamba):
        super().__init__()

    def forward(self, y):
        si = y[0]
        si = y[0]
        ii = y[1]
        ri = y[2]
        dsdt = -self.beta * si * ii
        didt = self.beta * si * ii - self.gama * ii
        drdt = self.gama * ii
        return torch.cat((dsdt, didt, drdt))


class Logistic(Model):

    """Docstring for Logistic. """

    def __init__(self):
        """TODO: to be defined. """
        super().__init__()