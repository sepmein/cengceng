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

    def add_trainable_parameter(self, name: str, parameter: int) -> object:
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
        beta = self.add_trainable_parameter("beta", beta)
        gama = self.add_trainable_parameter("gama", gama)

    def forward(self, t, y):
        beta = self.parameters["beta"]
        gama = self.parameters["gama"]
        si = y[0]
        ii = y[1]
        ri = y[2]
        dsdt = -beta * si * ii
        didt = beta * si * ii - gama * ii
        drdt = gama * ii
        return torch.cat((dsdt, didt, drdt))

    def fit(self, y: np.ndarray) -> None:
        pass

    def r_o(self) -> int:
        beta = self.parameters["beta"]
        gama = self.parameters["gama"]
        r_o = beta / gama
        return r_o.detach().numpy()


class Seir(Sir):
    """
    Inherited from sir model
    """

    def __init__(self, lamba):
        super().__init__()

    def forward(self, y):
        si = y[0]
        ii = y[1]
        ri = y[2]
        dsdt = -self.beta * si * ii
        didt = self.beta * si * ii - self.gama * ii
        drdt = self.gama * ii
        return torch.cat((dsdt, didt, drdt))


class Malaria_Transmission_Model(object):

    """
    Docs for Malaria_Transmission_Model.

    references: TODO add references later.
    """

    def __init__(self):
        """
        Model init:
        - STATS vars
            S - susceptible
            T - treated clinical disease
            D - untreated clinical disease
            A - asymptomatic infection which may be detected by microscopy
            U - sub-patent infection
            P - protected by a period of prophylaxis from prior treatment
        - Model parameters
            lamb    -   force of the infection
            fai     -   probability of clinical disease upon infection
            ft      -   probability that clinical malaria is effectively treated
            dt      -   duration of T state
            dd      -   duration of clinical disease state
            da      -   duration of asymptomatic infection state
            dp      -   duration of P state
            du      -   duration of U state
        - Probabilities
            ps      -   probability of S -> T
            1 - ps  -   Probability of S -> A
            pt      -   probability of T -> T
            1 - pt  -   probability of T -> A
            pd      -   probability of D -> T
            1 - pd  -   probability of D -> A
            pa      -   probalility of A -> T
            1 - pa  -   probability of A -> A
            pu      -   probability of U -> T
            1 - pu  -   probability of U -> A
            pp      -   probability of P -> T
            1 - pp  -   probability of P -> A
        """
        # states
        self.S = 0
        self.T = 0
        self.D = 0
        self.A = 0
        self.U = 0
        self.P = 0
        # model parameters
        self.lamb = 0
        self.fai = 0
        self.ft = 0
        self.dt = 0
        self.dd = 0
        self.da = 0
        self.dp = 0
        self.du = 0
        # probalilities
        self.ps = 0
        self.pt = 0
        self.pd = 0
        self.pa = 0
        self.pu = 0
        self.pp = 0

    def forward(self):
        S = self.S
        T = self.T
        D = self.D
        A = self.A
        U = self.U
        P = self.P
        lamb = self.lamb
        fai = self.fai
        ft = self.ft
        dt = self.dt
        dd = self.dd
        da = self.da
        dp = self.dp
        du = self.du
        ps = self.ps
        pt = self.pt
        pd = self.pd
        pa = self.pa
        pu = self.pu
        pp = self.pp

        dsdt = ps * (-lamb * S + P / dp + U / du)
        dsda = (1 - ps) * (-lamb * S + P / dp + U / du)
        dtdt = pt * fai * ft * lamb * (S + A + U) - T / dt
        dtda = (1 - pt) * fai * ft * lamb * (S + A + U) - T / dt
        dddt = pd * fai * (1 - ft) * lamb * (S + A + U) - D / dd
        ddda = (1 - pd) * fai * (1 - ft) * lamb * (S + A + U) - D / dd
        dadt = pa * (1 - fai) * lamb * (S + U) + D / dd - fai * lamb * A - A / da
        dada = (1 - pa) * (1 - fai) * lamb * (S + U) + D / dd - fai * lamb * A - A / da
        dudt = pu * A / da - U / du - lamb * U
        duda = (1 - pu) * A / da - U / du - lamb * U
        dpdt = dp * T / dt - P / dp
        dpda = (1 - dp) * T / dt - P / dp
        return torch.cat(
            (
                dsdt,
                dsda,
                dtdt,
                dtda,
                dddt,
                ddda,
                dadt,
                dada,
                dudt,
                dudt,
                duda,
                dpdt,
                dpda,
            )
        )


class Logistic(Model):

    """Docstring for Logistic. """

    def __init__(self):
        """TODO: to be defined. """
        super().__init__()