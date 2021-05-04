import torch
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)
""" 
Notes:
Spin dimensions: alpha, beta 0:=down, 1:=up
Site dimensions: i, j, k, l
"""

def positive_semidefinite(M, n_sites = None):
    if n_sites == None:
        n_sites = M.shape[-1]
    else:
        pass
    return torch.matmul(M.permute(0, 2, 4, 1, 3, 5).reshape(2*n_sites**2, 2*n_sites**2),
                        M.permute(0, 2, 4, 1, 3, 5).reshape(2*n_sites**2, 2*n_sites**2))\
        .reshape(2, n_sites, n_sites, 2, n_sites, n_sites).permute(0, 3, 1, 4, 2, 5)

def hermitian(M):
    return (M + M.rename(None).permute(0,1,4,5,2,3))/2

def antisymmetrize(M):
    M_b = M.clone()
    M = M.rename(None)
    M_b[0, 0] = M[0, 0] - M[0, 0].permute(0, 1, 3, 2) - M[0, 0].permute(1, 0, 2, 3)\
              + M[0, 0].permute(1, 0, 3, 2)
    M_b[1, 1] = M[1, 1] - M[1, 1].permute(0, 1, 3, 2) - M[1, 1].permute(1, 0, 2, 3)\
              + M[1, 1].permute(1, 0, 3, 2)
    M_b[0, 1] = M[0, 1] + M.permute(1, 0, 3, 2, 5, 4)[0, 1]
    M_b[1, 0] = M[1, 0] + M.permute(1, 0, 3, 2, 5, 4)[1, 0]
    return M_b.rename('alpha', 'beta', 'i', 'k', 'l', 'j')

def test_antisymmetrize(M):
    aM = antisymmetrize(M).rename(None)
    one = torch.allclose(aM, aM.permute(1, 0, 3, 2, 5, 4), rtol=1e-06)
    two = torch.allclose(aM, aM.permute(1, 0, 3, 2, 5, 4), rtol=1e-06)
    three = torch.allclose(aM[0 ,0], -aM.permute(0, 1, 2, 3, 5, 4)[0,0], rtol=1e-06)
    four = torch.allclose(aM[1, 1], -aM.permute(0, 1, 3, 2, 4, 5)[1, 1], rtol=1e-06)
    if one and two and three and four:
        pass
    else:
        print(one, two, three, four)
        raise ValueError

def test_hermitian(M):
    if torch.allclose(M, hermitian(M).permute(0,1,4,5,2,3)):
        pass
    else:
        raise ValueError

def normalize_eq_4_5(M, N_up, N_down):
    #dim ('alpha','beta', 'i', 'k', 'l', 'j')
    M = torch.diagonal(M.rename(None), dim1=2, dim2=5) #dim alpha beta k l i
    M = torch.diagonal(M, dim1=2, dim2=3) #dim alpha beta i k
    M = M.rename('alpha', 'beta', 'i', 'k')
#    print(M.sum(['i','k']))
#    print('spinmatrix 45',torch.tensor([[N_down*(N_down-1), N_up*N_down],[N_up*N_down, N_up*(N_up-1)]]))
    norm = torch.tensor([[N_down*(N_down-1), N_up*N_down],[N_up*N_down, N_up*(N_up-1)]]) / M.sum(['i','k'])
    return norm.rename(None).view(2, 2, 1, 1, 1, 1).rename('alpha','beta', 'i', 'k', 'l', 'j')

def normalize_eq_6_abs(M, N_up, N_down):
    #dim ('alpha','beta', 'i', 'k', 'l', 'j')
    M = torch.diagonal(M.rename(None), dim1=2, dim2=5).rename(None) #dim alpha beta k j i
    M = torch.diagonal(M, dim1=2, dim2=3) #dim alpha beta i k
    M = M.rename('alpha', 'beta', 'i', 'k')
    print('spinmatrix 6', torch.tensor([[N_down, 1/2. * (N_up + N_down) + (N_down - N_up)**2 + abs(N_up - N_down) * (abs(N_up - N_down) + 1)],
                         [1/2. * (N_up + N_down) + (N_up - N_down)**2 + abs(N_up - N_down) * (abs(N_up - N_down) + 1), N_up]]))
    norm = torch.tensor([[N_down, 1/2. * (N_up + N_down) + (N_down - N_up)**2 + abs(N_up - N_down) * (abs(N_up - N_down) + 1)],
                         [1/2. * (N_up + N_down) + (N_up - N_down)**2 + abs(N_up - N_down) * (abs(N_up - N_down) + 1), N_up]]) / M.sum(['i','k'])
    return norm.rename(None).view(2, 2, 1, 1, 1, 1).rename('alpha','beta', 'i', 'k', 'l', 'j')

def normalize_eq_6(M, N_up, N_down, S):
    #dim ('alpha','beta', 'i', 'k', 'l', 'j')
    M = torch.diagonal(M.rename(None), dim1=2, dim2=5).rename(None) #dim alpha beta k j i
    M = torch.diagonal(M, dim1=2, dim2=3) #dim alpha beta i k
    M = M.rename('alpha', 'beta', 'i', 'k')
    constrain = (torch.tensor([[N_down - S * (S + 1), 1/2. * (N_up + N_down) + (N_down - N_up)**2 - S * (S + 1)],
                         [1/2. * (N_up + N_down) + (N_up - N_down)**2 + - S * (S + 1), N_up - S * (S + 1)]]).rename('alpha', 'beta'), M.sum(['i','k']))
    return constrain



def calc_E(D, t, U, N_beta):
    # dim ('alpha','beta', 'i', 'k', 'l', 'j')
    N_beta = N_beta.view(2,1).rename('alpha','off_diag_i')
    U_term = torch.diagonal(D.rename(None), dim1=2, dim2=5) #dim alpha beta k l i
    U_term = torch.diagonal(U_term, dim1=2, dim2=3) #dim alpha beta i k
    U_term = torch.diagonal(U_term, dim1=2, dim2=3) # dim alpha beta i
    U_term =  U_term.rename('alpha', 'beta', 'i')
    U_term = U*torch.sum(U_term[1][0], ['i'])
    rdm = calc_1RDM(D, torch.sum(N_beta))
    T_term = torch.diagonal(rdm.rename(None), dim1=1, dim2=2, offset=1) + torch.diagonal(rdm.rename(None), dim1=1, dim2=2, offset=-1) #dim alpha beta k l i,i=i+1
    T_term = T_term.rename('alpha','off_diag_i') #dim alpha beta i k
    T_term = -t*T_term
    T_term = torch.sum(T_term, ['alpha','off_diag_i'])
    return T_term + U_term

def calc_1RDM(D,N):
        # dim ('alpha','beta', 'i', 'k', 'l', 'j')
        D = torch.diagonal(D.rename(None), dim1=3, dim2=4).rename('alpha','beta', 'i', 'j', 'k')  # alpha beta  i j k
        return torch.sum(D, ['k', 'beta'])

def constraint_eq_12(D, N_alpha, n_sites = None):
    if n_sites == None:
        n_sites = D.shape[-1]
    else:
        pass
    delta_alphabeta = torch.eye(2).view(2, 2, 1, 1, 1, 1).repeat(1, 1, n_sites, n_sites, n_sites, n_sites).rename(
        'alpha', 'beta', 'i', 'k', 'l', 'j')
    delta_il = torch.eye(n_sites).view(1, 1, n_sites, 1, n_sites, 1).repeat(2, 2, 1, n_sites, 1, n_sites).rename(
        'alpha', 'beta', 'i', 'k', 'l', 'j')
    delta_jk = torch.eye(n_sites).view(1, 1, 1, n_sites, 1, n_sites).repeat(2, 2, n_sites, 1, n_sites, 1).rename(
        'alpha', 'beta', 'i', 'k', 'l', 'j')
    delta_lk = torch.eye(n_sites).view(1, 1, 1, n_sites, n_sites, 1).repeat(2, 2, n_sites, 1, 1, n_sites).rename(
        'alpha', 'beta', 'i', 'k', 'l', 'j')
    delta_ij = torch.eye(n_sites).view(1, 1, n_sites, 1, 1, n_sites).repeat(2, 2, 1, n_sites, n_sites, 1).rename(
        'alpha', 'beta', 'i', 'k', 'l', 'j')
    rdm = calc_1RDM(D)
    rdm_alpha_k_j = rdm.rename(None).view(2, 1, 1, n_sites, n_sites, 1).rename('alpha', 'beta', 'i', 'k', 'l', 'j' )
    rdm_alpha_i_l = rdm.rename(None).view(2, 1, n_sites, 1, 1, n_sites).rename('alpha', 'beta', 'i', 'k', 'l', 'j')
    rdm_alpha_i_j = rdm.rename(None).view(2, 1, n_sites, 1, n_sites, 1).rename('alpha', 'beta', 'i', 'k', 'l', 'j')
    rdm_alpha_k_l = rdm.rename(None).view(2, 1, 1, n_sites, 1, n_sites).rename('alpha', 'beta', 'i', 'k', 'l', 'j' )
    N_alpha = N_alpha.view(2, 1, 1, 1, 1, 1).rename('alpha','beta', 'i', 'k', 'l', 'j')
    #D dim ('alpha','beta', 'i', 'k', 'l', 'j')
    constraint = delta_ij * delta_lk - delta_alphabeta * delta_il * delta_jk\
                 + delta_alphabeta / (N_alpha - 1) \
                 * (delta_il * rdm_alpha_k_j + delta_jk * rdm_alpha_i_l) \
                 +  1 / (N_alpha - 1) \
                 * (delta_jk*rdm_alpha_i_j + delta_ij*rdm_alpha_k_l) + D
    return constraint

def constraint_eq_13(D, N_alpha, n_sites = None):
    rdm = calc_1RDM(D)
    rdm_alpha_jj = torch.diagonal(rdm.rename(None), dim1=-2, dim2=-1).view(2, 1, 1, 1, n_sites, 1)\
        .rename('alpha', 'beta', 'i', 'k', 'l', 'j')

    delta_lk = torch.eye(n_sites).view(1, 1, 1, n_sites, n_sites, 1).repeat(2, 2, n_sites, 1, 1, n_sites).rename(
        'alpha', 'beta', 'i', 'k', 'l', 'j')
    N_alpha = N_alpha.view(2, 1, 1, 1, 1, 1).rename('alpha','beta', 'i', 'k', 'l', 'j')
    constraint = delta_lk/(N_alpha-1) * rdm_alpha_jj + D


#if __name__ == "main":
n_sites = 3
delta_il = torch.eye(n_sites).view(1, 1, n_sites, 1, n_sites, 1).repeat(2, 2, 1, n_sites, 1, n_sites).rename(
    'alpha', 'beta', 'i', 'k', 'l', 'j')
delta_jk = torch.eye(n_sites).view(1, 1, 1, n_sites, 1, n_sites).repeat(2, 2, n_sites, 1, n_sites, 1).rename(
    'alpha', 'beta', 'i', 'k', 'l', 'j')
delta_lk = torch.eye(n_sites).view(1, 1, 1, n_sites, n_sites, 1).repeat(2, 2, n_sites, 1, 1, n_sites).rename(
    'alpha', 'beta', 'i', 'k', 'l', 'j')
delta_ij = torch.eye(n_sites).view(1, 1, n_sites, 1, 1, n_sites).repeat(2, 2, 1, n_sites, n_sites, 1).rename(
    'alpha', 'beta', 'i', 'k', 'l', 'j')
delta_alphabeta = torch.eye(2).view(2, 2, 1, 1, 1, 1).repeat(1, 1, n_sites, n_sites, n_sites, n_sites).rename(
        'alpha', 'beta', 'i', 'k', 'l', 'j')
N_up = 2.0
N_down = 3.0
M  = torch.rand(2,2,3,3,3,3)
M = hermitian(M)
#M = positive_semidefinite(M)
M = antisymmetrize(M)
a = normalize_eq_4_5(delta_ij*delta_lk*M, N_up = N_up, N_down = N_down)
E_0 = -3.
t =2
a= torch.tensor([E_0**2, -2*t*E_0,-2*t*E_0, E_0**2]).view(4,1)
b= torch.tensor([-2*t*E_0, 4*t**2, 4*t**2, -2*t*E_0]).view(4,1)

D = torch.stack((a,b,b,a), axis=-1).view(1,1,2,2,2,2).repeat(2,2,1,1,1,1)-delta_alphabeta*torch.stack((a,b,b,a), axis=-1).view(1,1,2,2,2,2).repeat(2,2,1,1,1,1)
1rdm = torch.tensor([[4*t**2+E_0**2, -4*t*E_0],[-4*t*E_0,  4*t**2+E_0**2]])

#print(M_WF)
#print(M_WF[:,0]+M_WF[:,:,0])
#print(torch.where(M_WF>0.0001))