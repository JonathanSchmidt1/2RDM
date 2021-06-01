import torch
from main import positive_semidefinite, hermitian,antisymmetrize,test_antisymmetrize, test_hermitian, normalize_eq_4_5, normalize_eq_4_5_old,\
    normalize_eq_7, calc_E, calc_1RDM
from torch import nn
torch.set_default_dtype(torch.float64)
class TwoRDM(nn.Module):
    def __init__(self, n_sites, N, N_up, N_down, S):
        super().__init__()
        self.upup = nn.Parameter(antisymmetrize(torch.rand(n_sites, n_sites,n_sites, n_sites)), requires_grad=True) #('i','j','k','l')
        self.downdown = nn.Parameter(antisymmetrize(torch.rand(n_sites, n_sites,n_sites, n_sites)), requires_grad=True)
        self.downup = nn.Parameter(torch.rand(n_sites, n_sites,n_sites, n_sites), requires_grad=True)
        self.N = N
        self.N_up = N_up
        self.N_down = N_down
        self.N_alpha = torch.tensor([N_down, N_up])
        self.mse = torch.nn.MSELoss()
        self.S = S
    def calc_2rdm(self):
        #upup = positive_semidefinite(hermitian(antisymmetrize(self.upup))
        upup = positive_semidefinite(hermitian(self.upup))
        #downdown = positive_semidefinite(hermitian(antisymmetrize(self.downdown)))
        downdown = positive_semidefinite(hermitian(self.downdown))
        downup = positive_semidefinite(hermitian(self.downup))
        downup = (downup + downup.permute(1,0,3,2))/2.
        # print(torch.eig(upup.view(9,9)),torch.eig(downdown.view(9,9)),torch.eig(downup.view(9,9)))
        D1 = torch.stack([downdown, downup])
        D2 = torch.stack([downup.rename(None).permute(1, 0, 3, 2), upup])
        D = torch.stack([D1, D2])
        norm = normalize_eq_4_5_old(D, self.N_up, self.N_down)
        #c_6_true, c_6 = normalize_eq_4_5(D, self.N_up, self.N_down)
        D = D * norm
        return D.rename('alpha', 'beta', 'i', 'k', 'l', 'j')
    def forward(self,t, U):
        D = self.calc_2rdm()
        #print(test_hermitian(D.rename(None)))
        #print(test_antisymmetrize(D.rename(None)))
        c_7_true, c_7 = normalize_eq_7(D, self.N_up, self.N_down, self.S)
        c_6_true, c_6 = normalize_eq_4_5(D, self.N_up, self.N_down)
        E = calc_E(D, t, U, self.N_alpha)
        return E, torch.sum((c_6_true - c_6)**2) , torch.sum((c_7_true - c_7)**2)

"""
E_0 = -3.
t = 2
U= E_0-4*t**2/E_0
D = TwoRDM(2, 2., 1., 1., 0.5)
test = D.forward(t, U)
print(test)
n_sites=2
a = torch.tensor([E_0 ** 2, -2 * t * E_0, -2 * t * E_0, E_0 ** 2]).view(4, 1)
b = torch.tensor([-2 * t * E_0, 4 * t ** 2, 4 * t ** 2, -2 * t * E_0]).view(4, 1)
delta_alphabeta = torch.eye(2).view(2, 2, 1, 1, 1, 1).repeat(1, 1, n_sites, n_sites, n_sites, n_sites).rename(
            'alpha', 'beta', 'i', 'k', 'l', 'j')
E = (torch.stack((a, b, b, a), axis=-1).view(1, 1, 2, 2, 2, 2).repeat(2, 2, 1, 1, 1, 1) - delta_alphabeta * torch.stack(
    (a, b, b, a), axis=-1).
     view(1, 1, 2, 2, 2, 2).repeat(2, 2, 1, 1, 1, 1)) \
    / (8 * t ** 2 + 2 * E_0 ** 2)
print('eq_6', normalize_eq_6(D, 1.0, 1.0, 0.5))
print('eq4', normalize_eq_4_5(D, 1.0,1.0))
#print(calc_E(D, t, U, torch.tensor([1.0,1.0])))
"""
torch.set_printoptions(20)
E_0 = -3.
t = 2
U= E_0-4*t**2/E_0
#U=2
D = TwoRDM(2, 4., 2., 2., 0.)
optimizer = torch.optim.Adam(D.parameters(), lr=0.01)
for i in range(1000000):
    optimizer.zero_grad()
    E, constrain_6, constrain_7 = D.forward(t, U)
    print(E, constrain_6, constrain_7)#, constrain_10)
    loss = 10e16 * constrain_6 + 10e16 * constrain_7 + E
    loss.backward()
    optimizer.step()

# for i in range(1000000):
#     optimizer.zero_grad()
#     E, constrain_10 = D.forward(t, U)
#     print('E', E, constrain_10)
#     loss = E + 1e16*constrain_10
#     loss.backward()
#     optimizer.step()


n_sites = 2
# a = torch.tensor([E_0 ** 2, -2 * t * E_0, -2 * t * E_0, E_0 ** 2]).view(4, 1)
# b = torch.tensor([-2 * t * E_0, 4 * t ** 2, 4 * t ** 2, -2 * t * E_0]).view(4, 1)
delta_alphabeta = torch.eye(2).view(2, 2, 1, 1, 1, 1).repeat(1, 1, n_sites, n_sites, n_sites, n_sites).rename(
             'alpha', 'beta', 'i', 'k', 'l', 'j')
# E = (torch.stack((a, b, b, a), axis=-1).view(1, 1, 2, 2, 2, 2).repeat(2, 2, 1, 1, 1, 1) - delta_alphabeta * torch.stack(
#     (a, b, b, a), axis=-1).
#      view(1, 1, 2, 2, 2, 2).repeat(2, 2, 1, 1, 1, 1)) \
#     / (8 * t ** 2 + 2 * E_0 ** 2)

#
a = torch.eye(4).repeat(2,2,1,1).view(2,2,2,2,2,2)
d= a - delta_alphabeta * a
d = d.rename(None).permute(0, 1, 2, 3, 5, 4).rename('alpha', 'beta', 'i', 'k', 'l', 'j')
d2 = torch.zeros(2,2,2,2)
d2[0,1,1,0]=1
d2[0,1,0,1]=-1
d2[1,0,1,0]=-1
d2[1,0,0,1]=1
d = d+ delta_alphabeta*d2.view(1,1,2,2,2,2)

#print(normalize_eq_6(E,D.N_up, D.N_down, D.S), E)
#print(test.shape)
#N_up = 2.0
#N_down = 3.0
#M  = torch.rand(2,2,3,3,3,3)
#M = hermitian(M)
#M = positive_semidefinite(M)
#M = antisymmetrize(M)