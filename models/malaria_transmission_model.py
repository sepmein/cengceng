import torch


class Malaria_Transmission_Model(object):

    """Docstring for Malaria_Transmission_Model. """

    def __init__(self):
        """
        Model init:
        - STATS vars
            susceptible ( S ), treated clinical disease ( T ), untreated clinical disease ( D ), asymptomatic infection which may be detected by microscopy ( A ), sub-patent infection (U ) and protected by a period of prophylaxis from prior treatment ( P ).
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
        S = 0
        T = 0
        D = 0
        A = 0
        U = 0
        P = 0
        # model parameters
        lamb = 0
        fai = 0
        ft = 0
        dt = 0
        dd = 0
        da = 0
        dp = 0
        du = 0

    def forward(self):
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
