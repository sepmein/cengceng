import torch


class SirModel:
    def __init__(self, beta, gama):
        super(SirModel, self).__init__()
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
