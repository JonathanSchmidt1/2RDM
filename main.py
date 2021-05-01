import torch
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)
""" 
Notes:
Spin dimensions: alpha, beta 0:=down, 1:=up
Site dimensions: i, j, k, l
"""

def positive_semidefinite(M):
    return torch.matmul(M.permute(0, 2,4,1,3,5).reshape(2*n_sites**2,2*n_sites**2),
                        M.permute(0, 2,4,1,3,5).reshape(2*n_sites**2,2*n_sites**2)).reshape(2,3,3,2,3,3).permute(0,3,1,4,2,5)

def hermitian(M):
    return (M + M.permute(0,1,4,5,2,3))

#def antisymmetrize(M):
#    M = M.permute(0, 1, 2, 3) - M.permute(0, 1, 3, 2) - M.permute(0, 2, 1, 3) + M.permute(0, 2, 3, 1)\
#        + M.permute(0, 3, 1, 2) - M.permute(0, 3, 2, 1) - M.permute(1, 0, 2, 3) + M.permute(1, 0, 3, 2)\
#        + M.permute(1, 2, 0, 3) - M.permute(1, 2, 3, 0) - M.permute(1, 3, 0, 2) + M.permute(1, 3, 2, 0)\
#        + M.permute(2, 0, 1, 3) - M.permute(2, 0, 3, 1) - M.permute(2, 1, 0, 3) + M.permute(2, 1, 3, 0)\
#        + M.permute(2, 3, 0, 1) - M.permute(2, 3, 1, 0) - M.permute(3, 0, 1, 2) + M.permute(3, 0, 2, 1)\
#        + M.permute(3, 1, 0, 2) - M.permute(3, 1, 2, 0) - M.permute(3, 2, 0, 1) + M.permute(3, 2, 1, 0)
#    return M

#def antisymmetrize(M):
#    M = M.rename(None).permute(0, 1, 2, 3) - M.rename(None).permute(0, 1, 3, 2) - M.rename(None).permute(1, 0, 2, 3)\
#        + M.rename(None).permute(1, 0, 3, 2)
#    return M

#def antisymmetrize(M):
#    M = M.rename(None).permute(0 , 1, 2, 3, 4, 5) - M.rename(None).permute(0, 1, 2, 3, 5, 4) - M.rename(None).permute(0, 2, 1, 3, 4, 5)\
#        + M.rename(None).permute(0, 2, 1, 3, 5, 4)
#    return M


def antisymmetrize(M):
    M[0, 0] = M[0, 0] - M[0, 0].rename(None).permute(0, 1, 3, 2) - M[0, 0].rename(None).permute(1, 0, 2, 3)\
              + M[0, 0].rename(None).permute(1, 0, 3, 2)
    M[1, 1] = M[1, 1] - M[1, 1].rename(None).permute(0, 1, 3, 2) - M[1, 1].rename(None).permute(1, 0, 2, 3)\
              + M[1, 1].rename(None).permute(1, 0, 3, 2)
    M[0, 1] = M[0, 1] + M[1, 0].rename(None).permute(1, 0, 3, 2)
    M[1, 0] = M[1, 1] + M[0, 1].rename(None).permute(1, 0, 3, 2)
    return M.rename('alpha', 'beta', 'i', 'k', 'l', 'j')

# make D 4 dimensional 2M*2M*M*M
def view_alpha_i_j_beta_k_l( M, n_sites = None):
    if n_sites == None:
        n_sites = int(np.sqrt( M.shape[0]/2))
    else:
        pass
    M=  M.view(2, n_sites, n_sites, 2, n_sites, n_sites)
    M =  M.rename('alpha','beta', 'i', 'k', 'l', 'j')
    return  M

#def view_i_j_k_l(M, n_sites = None):
#    if n_sites == None:
#        n_sites = int(np.sqrt(M.shape[0] / 2))
#    else:
#        pass
#    M =  M.rename(None).view(2, 2, n_sites, n_sites,n_sites, n_sites)
#    M =  M.rename('alpha_i','j','beta_k','l')
#    return M

def normalize_eq_4_5(M, N_up, N_down):
    if len(M.shape)==6:
        pass
    else:
        M = view_alpha_i_j_beta_k_l(M)
    #dim ('alpha','beta', 'i', 'k', 'l', 'j')
    M = torch.diagonal(M, dim1=2, dim2=5).rename(None) #dim alpha beta k l i
    M = torch.diagonal(M, dim1=2, dim2=3) #dim alpha beta i k
    M = M.rename('alpha', 'beta', 'i', 'k')
    norm = torch.tensor([[N_up*(N_up-1), N_up*N_down],[N_up*N_down, N_down*(N_down-1)]]) / M.sum(['i','k'])
    return norm.rename(None).view(2, 2, 1, 1, 1, 1).rename('alpha','beta', 'i', 'k', 'l', 'j')

def normalize_eq_6(M, N_up, N_down):
    if len(M.shape)==6:
        pass
    else:
        M = view_alpha_i_j_beta_k_l(M)
    #dim ('alpha','beta', 'i', 'k', 'l', 'j')
    M = torch.diagonal(M, dim1=2, dim2=4).rename(None) #dim alpha beta k j i
    M = torch.diagonal(M, dim1=2, dim2=3) #dim alpha beta i k
    M = M.rename('alpha', 'beta', 'i', 'k')
    norm = torch.tensor([[N_up, 1/2. * (N_up + N_down) + (N_up - N_down) * (N_up - N_down + 1)],
                         [1/2. * (N_up + N_down) + (N_up - N_down) * (N_up - N_down + 1), N_down]]) / M.sum(['i','k'])
    return norm.rename(None).view(2, 2, 1, 1, 1, 1).rename('alpha','beta', 'i', 'k', 'l', 'j')

def calc_E(D, t, U, N_alpha):
    # dim ('alpha','beta', 'i', 'k', 'l', 'j')
    U_term = torch.diagonal(M, dim1=2, dim2=5) #dim alpha beta k l i
    U_term = torch.diagonal(U_term, dim1=2, dim2=3) #dim alpha beta i k
    U_term = U * torch.diagonal(U_term, dim1=2, dim2=3) # dim alpha beta i
    U_term = torch.sum(U_term[1][0], ['i'])

    T_term = torch.diagonal(M, dim1=2, dim2=5, offset=1) #dim alpha beta k l i,i=i+1
    T_term = torch.diagonal(T_term, dim1=2, dim2=3) #dim alpha beta i k
    T_term = -2 * t * 1 / (N_alpha - 1).view(2,1,1).rename('alpha', 'i', 'k')\
             * torch.diagonal(U_term,dim1=0,dim2=1) #dim alpha i k
    T_term = torch.sum(T_term, 'i', 'j', 'alpha')

    return T_term + U_term

def calc_1RDM(D):
    # dim ('alpha','beta', 'i', 'k', 'l', 'j')
    D = torch.diagonal(D, dim1=3, dim2=4)  # alpha beta  i j k
    D = torch.diagonal(D, dim1=0, dim2=1).permute(3,0,1,2).rename('alpha', 'i', 'j', 'k')  # alpha beta  i j
    return torch.sum(D, 'k')



#i, k = np.ogrid[:n_sites, :n_sites]
#delta_ij_kl = np.zeros((n_sites, n_sites, n_sites, n_sites), int)
#delta_ij_kl[i, k, k, i] = 1
#delta_ij_kl = torch.tensor(delta_ij_kl).repeat(2, 2, 1, 1, 1, 1).rename('alpha', 'beta', 'i', 'k', 'l', 'j')

#delta_il_kj = np.zeros((2, 2, n_sites, n_sites, n_sites, n_sites), int)
#delta_il_kj[0, 0][i, k, i, k] = 1
#delta_il_kj[1, 1][i, k, i, k] = 1
#delta_alphabeta_il_kj = torch.tensor(delta_il_kj).rename('alpha', 'beta', 'i', 'k', 'l', 'j')


def constraint_eq_12(D, n_sites, N_alpha):
    if n_sites == None:
        n_sites = int(np.sqrt( D.shape[0]/2))
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

#if __name__ == "main":
M  = torch.rand(4,4,4,4)
M_WF = antisymmetrize(M)
print(M_WF)
print(M_WF[:,0]+M_WF[:,:,0])
print(torch.where(M_WF>0.0001))