import torch
from main import positive_semidefinite, hermitian,antisymmetrize,test_antisymmetrize, test_hermitian, normalize_eq_4_5,\
    normalize_eq_6_abs, calc_E, calc_1RDM
from torch import nn

class TwoRDM(nn.Module):
    def __init__(self, n_sites, N, N_up, N_down):
        super().__init__()
        self.upup = nn.Parameter(torch.rand(n_sites, n_sites,n_sites, n_sites), requires_grad=True).rename('i','j','k','l')
        self.downdown = nn.Parameter(torch.rand(n_sites, n_sites,n_sites, n_sites), requires_grad=True).rename('i','j','k','l')
        self.downup = nn.Parameter(torch.rand(n_sites, n_sites,n_sites, n_sites), requires_grad=True).rename('i','j','k','l')
        self.N = N
        self.N_up = N_up
        self.down = N_down
        self.N_alpha = torch.tensor([N_down, N_up])
    def forward(self)
        upup = positive_semidefinite(hermitian(self.upup))
        downdown = positive_semidefinite(hermitian(self.downdown))
        downdown = antisymmetrize(downdown)
        upup = antisymmetrize(upup)
        downup = positive_semidefinite(hermitian(self.downup))
        D1 = torch.stack([downdown, downup])
        D2 = torch.stack([downup, upup])
        return


#N_up = 2.0
#N_down = 3.0
#M  = torch.rand(2,2,3,3,3,3)
#M = hermitian(M)
#M = positive_semidefinite(M)
#M = antisymmetrize(M)