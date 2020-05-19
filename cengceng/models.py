"""
The model library
"""
import torch
import numpy as np


class Model(object):

    """
    Docstring for Model. The basic model.
    """

    def __init__(self) -> None:
        """init the model with preferences"""
        # Use Adam as the default optimizer
        self.optimizer = torch.optim.Adam
        self.parameters = {}

    def fit(self, optimizer=torch.optim.Adam, learning_rate=0.001) -> None:
        pass

    def load(self, data) -> None:
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            raise Exception(
                "The Model is Loading data, please use numpy array as the datatype"
            )

    def add_trainable_parameter(self, name: str, parameter: int) -> Model:
        self.parameters[name] = torch.tensor(parameter, requires_grad=True)
        return self

    def save(self, file_name: str, path: str) -> None:
        pass


class Compartmental(Model):

    """Docstring for Compartmental. """

    def __init__(self):
        """TODO: to be defined. """
        super().__init__()

    def forward(self):
        pass

    def fit(self, y: np.ndarray, learning_rate: int = 1e-3, training_steps: int = 1e5):
        """

        """
        # get t length of the real data
        t_length = y.shape[0]
        t = torch.arange(1.0, tlength)

        # construct parameters list object
        trainable_parameters = []
        for name, para in self.parameters:
            trainable_parameters.append(para)

        optimizer = self.optimizer(trainable_parameters, lr=learning_rate)

        for iteration in range(1, training_steps):
            optimizer.zero_grad()
            yhat = odeint(self.forward, y[0], t)
            loss = torch.sum(torch.abs(yhat - y))
            loss.backward()
            optimizer.step()


class Sir(Model):

    """
    Sir Model
    Parameters:
    - beta
    - gama
    """

    def __init__(self, beta=5e-1, gama=4e-1):
        super().__init__()
        self.add_trainable_parameter("beta", beta)
        self.add_trainable_parameter("gama", gama)
        self.r_o = self.beta / self.gama

    def forward(self, t, y):
        si = y[0]
        ii = y[1]
        ri = y[2]
        dsdt = -self.beta * si * ii
        didt = self.beta * si * ii - self.gama * ii
        drdt = self.gama * ii
        return torch.cat((dsdt, didt, drdt))

    def fit(self, y: np.ndarray) -> None:
        pass


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