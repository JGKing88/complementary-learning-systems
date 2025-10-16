# corrupt_p01() and topk() from assoc_utils.py
# haven't been converted to numpy so they are missing 
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from numpy.random import rand
from numpy.random import randn 
from numpy.random import randint
from cls.vectorhash.assoc_utils_np_2D import module_wise_NN_2d


def relu(x, thresh=0):
    return x * (x > thresh)


def nonlin(x, thresh=2.5):
    #return relu(x, 0)
    return relu(x-thresh, 0)
    #return np.sign(x)


# compute pattern correlations
# codebook = gbook/pbook/sbook
def correlation(codebook):
    return np.corrcoef(codebook, rowvar=False)


def extend_gbook(gbook, discretize):
    return np.repeat(gbook,discretize,axis=1)


def colvolve_1d(codebook, std):
    return gaussian_filter1d(codebook, std, mode="constant")


def cont_gbook(gbook, discretize=10, std=1):
    gbook = extend_gbook(gbook, discretize)
    gbook = colvolve_1d(gbook, std)
    return gbook


# generate modular grid code
def gen_gbook(lambdas, Ng, Npos):
    ginds = list(np.cumsum(lambdas))
    ginds = [0] + ginds[:-1]
    # if len(lambdas) == 2:
        # ginds = [0,lambdas[0]] 
    # elif len(lambdas) == 3:
        # ginds = [0,lambdas[0],lambdas[0]+lambdas[1]] 
    # elif len(lambdas) == 4:
        # ginds = [0,lambdas[0],lambdas[0]+lambdas[1],lambdas[0]+lambdas[1]+lambdas[2]]
    # elif len(lambdas) == 5:
        # ginds = [0,lambdas[0],lambdas[0]+lambdas[1],lambdas[0]+lambdas[1]+lambdas[2],
                 # lambdas[0]+lambdas[1]+lambdas[2]+lambdas[3]]  
    # elif len(lambdas) == 6:
        # ginds = [0,lambdas[0],lambdas[0]+lambdas[1],lambdas[0]+lambdas[1]+lambdas[2],
                 # lambdas[0]+lambdas[1]+lambdas[2]+lambdas[3],
                 # lambdas[0]+lambdas[1]+lambdas[2]+lambdas[3]+lambdas[4]]                   
    gbook=np.zeros((Ng,Npos))
    for x in range(Npos):
        phis = np.mod(x,lambdas) 
        gbook[phis+ginds,x]=1    
    return gbook


def train_hopfield(pbook, Npatts):
    return (1/Npatts)*np.einsum('ijk, ilk->ijl', pbook[:,:,:Npatts], pbook[:,:,:Npatts])


# -------------------- device / dtype helpers --------------------

def _dev():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def _to_torch(x, device=None, dtype=torch.float32):
    return torch.as_tensor(x, device=device or _dev(), dtype=dtype)

def _to_numpy(x_t):
    return x_t.detach().cpu().numpy()

# -------------------- GEMM helpers (torch, shape-agnostic) --------------------

def _ij_klj_to_kil_torch(G_t, T_t):
    """
    torch equivalent of einsum('ij,klj->kil', G, T) via GEMM.
    G_t: (I, J)
    T_t: (K, L, J)
    -> (K, I, L)
    """
    I, J = G_t.shape
    K, L, J2 = T_t.shape
    assert J == J2, "Dimension mismatch on J"

    T2 = T_t.reshape(K * L, J)          # (K*L, J)
    out = (T2 @ G_t.T).reshape(K, L, I) # (K, L, I)
    out = out.transpose(1, 2)           # (K, I, L)
    return out

def _ij_kli_to_klj_torch(G_t, T_t):
    """
    torch equivalent of einsum('ij,kli->klj' with G: (J, I), T: (K, L, I)) via GEMM.
    G_t: (J, I)
    T_t: (K, L, I)
    -> (K, L, J)
    """
    J, I = G_t.shape
    K, L, I2 = T_t.shape
    assert I == I2, "Dimension mismatch on I"

    T2 = T_t.reshape(K * L, I)          # (K*L, I)
    out = (T2 @ G_t.T).reshape(K, L, J) # (K, L, J)
    return out

# -------------------- EQUATION 3 --------------------

def train_gcpc(pbook, gbook, Npatts):
    """
    NumPy in -> NumPy out, uses GPU if available.
    gbook: (I, J)
    pbook: (K, L, J)  or (L, J)
    Returns:
      if 3D pbook -> (K, I, L)
      if 2D pbook -> (I, L)
    Implements (1/Npatts) * einsum('ij,klj->kil', ...) or (1/Npatts)*einsum('ij,lj->il').
    """
    device = _dev()
    # Slice first to reduce transfers
    G_np = np.asarray(gbook)[:, :Npatts]   # (I, J’)
    if pbook.ndim == 3:
        P_np = np.asarray(pbook)[:, :, :Npatts]  # (K, L, J’)

        G_t = _to_torch(G_np, device)
        P_t = _to_torch(P_np, device)

        out_t = _ij_klj_to_kil_torch(G_t, P_t)   # (K, I, L)
        out_np = _to_numpy(out_t) / float(Npatts)
        return out_np
    else:
        P_np = np.asarray(pbook)[:, :Npatts]     # (L, J’)

        G_t = _to_torch(G_np, device)            # (I, J’)
        PT_t = _to_torch(P_np.T, device)         # (J’, L)
        out_t = G_t @ PT_t                       # (I, L)
        out_np = _to_numpy(out_t) / float(Npatts)
        return out_np

# -------------------- train_pbook (NumPy I/O, GEMM) --------------------

def train_pbook(Wpg, gbook):
    """
    NumPy in -> NumPy out, GPU-accelerated GEMM.
    Wpg:   (J, K)
    gbook: (K, L, M)
    Returns: (Wpg @ gbook.reshape(K, L*M)).reshape(J, L, M), applied in NumPy.
    """
    device = _dev()
    W_np = np.asarray(Wpg)
    G_np = np.asarray(gbook)

    J, K = W_np.shape
    K2, *rest = G_np.shape
    assert K == K2, "Inner dims must match: Wpg[:,K] vs gbook[K,...]"

    W_t = _to_torch(W_np, device)
    Gflat_t = _to_torch(G_np.reshape(K, -1), device)   # (K, L*M)

    P_t = W_t @ Gflat_t                                # (J, L*M)
    P_np = _to_numpy(P_t).reshape(J, *rest)            # back to NumPy for nonlin
    return P_np

# -------------------- EQUATION 7 --------------------

def pseudotrain_Wsp(sbook, ca1book, Npatts):
    """
    NumPy in -> NumPy out, GPU-accelerated.
    sbook:   (I, J_full)
    ca1book: (K, L, J_full)  or (L, J_full)
    Returns:
      if 3D ca1book -> (K, I, L)  [einsum('ij, kjl -> kil')]
      if 2D ca1book -> (I, L)     [einsum('ij, jl  -> il')]
    Uses torch.linalg.pinv (batched) for the pseudoinverse.
    """
    device = _dev()
    S_np = np.asarray(sbook)[:, :Npatts]     # (I, J’)

    if ca1book.ndim == 3:
        C_np = np.asarray(ca1book)[:, :, :Npatts]   # (K, L, J’)
        C_t  = _to_torch(C_np, device)
        # pinv over last two dims: (K, L, J’) -> (K, J’, L)
        Cinv_t = torch.linalg.pinv(C_t)
        S_t = _to_torch(S_np, device)              # (I, J’)
        out_t = _ij_klj_to_kil_torch(S_t, Cinv_t)  # (K, I, L)
        return _to_numpy(out_t)

    else:
        C_np = np.asarray(ca1book)[:, :Npatts]     # (L, J’)
        C_t  = _to_torch(C_np, device)
        Cinv_t = torch.linalg.pinv(C_t)            # (J’, L)
        S_t = _to_torch(S_np, device)              # (I, J’)
        out_t = S_t @ Cinv_t                       # (I, L)
        return _to_numpy(out_t)

# -------------------- EQUATION 6 --------------------

def pseudotrain_Wps(ca1book, sbook, Npatts):
    """
    NumPy in → NumPy out (GPU if available).
    Original intent:
      sbookinv = pinv(sbook[:, :Npatts])  # (J', I) with J'=Npatts
      3D: einsum('ij, kli -> klj', sbookinv[:Npatts,:], ca1book[:,:,:Npatts])
      2D: einsum('ij, li  -> lj', sbookinv[:Npatts,:], ca1book[:,:Npatts])
    """
    device = _dev()

    # S[:, :Npatts] has shape (I, I') with I' = Npatts
    S_np = np.asarray(sbook)[:, :Npatts]
    S_t  = _to_torch(S_np, device)
    S_inv_t = torch.linalg.pinv(S_t)          # (I', I)  ≡ (J', I)

    if ca1book.ndim == 3:
        # C: (K, L, I') so dims match the 'i' index in einsum
        C_np = np.asarray(ca1book)[:, :, :Npatts]
        C_t  = _to_torch(C_np, device)        # (K, L, I')
        out_t = _ij_kli_to_klj_torch(S_inv_t, C_t)  # (K, L, J')
        return _to_numpy(out_t)
    else:
        # C: (L, I')  and S_inv_t: (I', I)  → (L, I)
        C_np = np.asarray(ca1book)[:, :Npatts]
        C_t  = _to_torch(C_np, device)        # (L, I')
        out_t = C_t @ S_inv_t                 # (L, I)   <<< no transpose
        return _to_numpy(out_t)


# #EQUATION 3
# def train_gcpc(pbook, gbook, Npatts):
#     if len(pbook.shape) == 3:
#         return (1/Npatts)*np.einsum('ij, klj -> kil', gbook[:,:Npatts], pbook[:,:,:Npatts])  
#     else:
#         return (1/Npatts)*np.einsum('ij, lj -> il', gbook[:,:Npatts], pbook[:,:Npatts])  
    
# #EQUATION 7
# def pseudotrain_Wsp(sbook, ca1book, Npatts):
#     if len(sbook.shape) == 3:
#         ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
#         return np.einsum('ij, kjl -> kil', sbook[:,:Npatts], ca1inv[:,:Npatts,:]) 
#     else:
#         ca1inv = np.linalg.pinv(ca1book[:, :Npatts])
#         return np.einsum('ij, jl -> il', sbook[:,:Npatts], ca1inv[:Npatts,:]) 

# #EQUATION 6
# def pseudotrain_Wps(ca1book, sbook, Npatts):
#     sbookinv = np.linalg.pinv(sbook[:, :Npatts])    
#     if len(ca1book.shape) == 3:
#         return np.einsum('ij, kli -> klj', sbookinv[:Npatts,:], ca1book[:,:,:Npatts]) 
#     else:
#         return np.einsum('ij, li -> lj', sbookinv[:Npatts,:], ca1book[:,:Npatts]) 
    
#NOT SURE WHY ANY OF THESE ARE NEEDED
def pseudotrain_Wpp(ca1book, Npatts):
    if len(ca1book.shape) == 3:
        ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
        return np.einsum('ijk, ikl -> ijl', ca1book[:,:,:Npatts], ca1inv[:,:Npatts,:]) 
    else:
        ca1inv = np.linalg.pinv(ca1book[:, :Npatts])
        return np.einsum('ij, ijl -> ijl', ca1book[:,:Npatts], ca1inv[:Npatts,:]) 

def pseudotrain_Wgp(ca1book, gbook, Npatts):
    if len(ca1book.shape) == 3:
        ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
        return np.einsum('ij, ljk -> lik', gbook[:,:Npatts], ca1inv[:,:Npatts,:]) 
    else:
        ca1inv = np.linalg.pinv(ca1book[:, :Npatts])
        return np.einsum('ij, jk -> ik', gbook[:,:Npatts], ca1inv[:Npatts,:]) 


def pseudotrain_Wpg(gbook, ca1book, Npatts):
    ginv = np.linalg.pinv(gbook[:,:Npatts])
    if len(ca1book.shape) == 3:
        return np.einsum('ijk, kl -> ijl', ca1book[:,:,:Npatts], ginv[:Npatts,:])
    else:
        return np.einsum('ij, l -> il', ca1book[:Npatts,:], ginv[:Npatts,:])

def pseudotrain_Wgg(gbook, Npatts):
    gbookinv = np.linalg.pinv(gbook)
    return np.einsum('ij, jk -> ik', gbook[:,:Npatts], gbookinv[:Npatts,:])     


def corrupt_pmask(Np, pflip, ptrue, nruns):
    #flipmask = rand(nruns, Np)>(1-pflip)
    flipmask = rand(*ptrue.shape)>(1-pflip)
    ind = np.argwhere(flipmask == True)  # find indices of non zero elements 
    pinit = np.copy(ptrue) 
    return pinit, ind


# corrupts p when its -1/1 code
def corrupt_p(Np, pflip, ptrue, nruns):
    if pflip == 0:
        return ptrue
    pinit, ind = corrupt_pmask(Np, pflip, ptrue, nruns)
    pinit[ind[:,0], ind[:,1]] = -1*pinit[ind[:,0], ind[:,1]] 
    return pinit 


def corrupt_pcont(pflip, ptrue):
    if pflip == 0:
        return ptrue
    pinit = ptrue + pflip*randn(*ptrue.shape)
    return pinit   


def hopfield(pinit, ptrue, Niter, W):
    p = pinit
    for i in range (Niter):
        p = np.sign(W@p)
    return np.sum(np.abs(p-ptrue), axis=(1,2))/np.sum(np.abs(pinit-ptrue), axis=(1,2))


def gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh):
    module_sizes = np.square(lambdas)
    module_gbooks = [np.eye(i) for i in module_sizes]
    m = len(lambdas)
    p = pinit
    for i in range(Niter):
        gin = Wgp@p;
        # g = topk_binary(gin, m)         # non modular net
        g = module_wise_NN_2d(gin, module_gbooks, module_sizes)  # modular net
        p = nonlin(Wpg@g, thresh); 
    return np.linalg.norm(p-ptrue, axis=(1,2))/Np

def gcpc_randomized(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh):
    module_sizes = np.square(lambdas)
    module_gbooks = [np.eye(i) for i in module_sizes]
    m = len(lambdas)
    p = pinit
    for i in range(Niter):
        gin = Wgp@p;
        # g = topk_binary(gin, m)         # non modular net
        g = module_wise_NN_2d(gin, module_gbooks, module_sizes)  # modular net
        p = Wpg@g
        #p = nonlin(Wpg@g, thresh); 
    return np.linalg.norm(p-ptrue, axis=(1,2))/Np
    
def gcpc_randomized_both(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh):
    p = pinit
    gin = Wgp@p
    g = nonlin(Wgp@p,0)
    p = Wpg@g
    return np.linalg.norm(p-ptrue, axis=(1,2))/Np

def find_g_in_gbook(g,gbook):
    return np.where((gbook.T==g.T).all(axis=1))[0][0]

def gridCAN_2d(gs,lambdas):
    #gs.shape == nruns,Ng,Npatts
    if len(gs.shape) == 2:
        nruns, Ng = gs.shape
    else:
        nruns,Ng,Npatts = gs.shape
    ls = [l**2 for l in lambdas]
    i=0
    gout = np.zeros(gs.shape)
    for j in ls:
        if len(gs.shape) == 2:
            gmod = gs[:,i:i+j]
        else:
            gmod = gs[:,i:i+j,:]
        #print(gmod.shape)
        maxes = gmod.argmax(axis=1)
        #print(maxes.shape)
        for ru in range(nruns):
            if len(gs.shape) == 2:
                gout[ru][maxes[ru]+i] = 1
            else:
                gout[ru][maxes[ru]+i,np.arange(Npatts)] = 1
        i=i+j
    return gout

# module wise nearest neighbor
def module_wise_NN(gin, gbook, lambdas):
    size = gin.shape
    g = np.zeros(size)               #size is (Ng, 1)
    print("gin.shape", gin.shape)
    i = 0
    for j in lambdas:
        gin_mod = gin[:, i:i+j]           # module subset of gin
        gbook_mod = gbook[i:i+j]
        print("gin_mod.shape", gin_mod.shape)
        print("gbook_mod.shape", gbook_mod.shape)
        g_mod = nearest_neighbor(gin_mod, gbook_mod)
        g[:, i:i+j, 0] = g_mod
        i = i+j
    return g  


# global nearest neighbor
def nearest_neighbor(gin, gbook):
    est = np.einsum('ijk, jl -> ikl', gin, gbook)
    maxm = np.amax(est, axis=2)       #(nruns,1)
    g = np.zeros((len(maxm), len(gbook)))
    for r in range(len(maxm)):
        a = np.argwhere(est[r] == maxm[r])
        idx = np.random.choice(a[:,1])
        g[r,:] = gbook[:,idx]; 
    return g


# return topk sparse code by setting 
# topk to 1 and all other values to zero
def topk_binary(gin, k):
    idx = np.argsort(gin, axis=1)
    idx = idx[:,-k:]
    idx = np.squeeze(idx)   # nruns x k
    g = np.zeros_like(gin) 
    nruns = gin.shape[0]   
    if k==1:
        g[np.arange(nruns),idx] = 1 
    else:               
        for i in range(k):
            g[np.arange(nruns),idx[:,i]] = 1
    return g


def default_model(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns):
    # avg error over Npatts
    err_hop = -1*np.ones((len(Npatts_lst), nruns))
    err_gcpc = -1*np.ones((len(Npatts_lst), nruns))
    
    Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        W = np.zeros((nruns, Np, Np));      # plastic pc-pc weights
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights

        # Learning patterns 
        W = train_hopfield(pbook, Npatts)
        Wgp = train_gcpc(pbook, gbook, Npatts)

        # Testing
        sum_hop = 0
        sum_gcpc = 0 
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
            pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
            cleanup = hopfield(pinit, ptrue, Niter, W)      # pc-pc autoassociative cleanup  
            sum_hop += cleanup
            cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas)   # pc-gc autoassociative cleanup
            sum_gcpc += cleanup
        err_hop[k] = sum_hop/Npatts
        err_gcpc[k] = sum_gcpc/Npatts
        k += 1   

    return err_hop, err_gcpc    

def sparse_rand(nruns, Np, Ng, sparsity):
    W = -1*np.ones((nruns, Np,Ng))
    W[:, :, :sparsity] = 1
    #shuffles at each position in-place 
    # (random shuffling across neurons) 
    for j in range(nruns):
        for i in range(Np):
            np.random.shuffle(W[j,i,:])     
    return W


from cls.vectorhash.theory_utils import *
from tqdm import tqdm
def capacity_gcpc(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns,Npatts_lst):
    # avg error over Npatts
    #Npatts_lst = np.arange(1,Npos+1)
    #Npatts_lst = [21]
    err_gcpc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    num_correct = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    l=0
    for Np in (Np_lst):
        
        Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
        c = 0.60     # connection probability
        prune = int((1-c)*Np*Ng)
        mask = np.ones((Np, Ng))
        mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
        Wpg = np.multiply(mask, Wpg)
        
        thresh = 0.5
        pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)  
        # print("pbook shape"+str(pbook.shape))
        k=0
        for Npatts in tqdm(Npatts_lst):
            #print("k = ",k)
            Wgp = np.zeros((nruns, Ng, Np));        # plastic pc-to-gc weights

            #Wgp = train_gcpc(pbook_grid, gbook_grid, np.prod(lambdas))
            Wgp = train_gcpc(pbook, gbook, Npatts)  # Training
            #Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)

            # Testing
            sum_gcpc = 0 
            num_corr=0
            # for x in range(Npatts): 
            # sampledpatt = np.random.choice(Npos,100)
            for x in range(Npos):
            # for x in sampledpatt:
                ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
                pinit = corrupt_pcont(pflip, ptrue)      # make corrupted pc pattern
                cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                #if cleanup[0] > 0:
                    #print(cleanup[0], x)
                    #print(gbook[:,x,None].T)
                sum_gcpc += cleanup
                num_corr += (cleanup<0.003).astype('int')
            err_gcpc[l,k] = sum_gcpc/Npatts
            num_correct[l,k] = num_corr
            k += 1   
        l += 1    
    return err_gcpc, num_correct                 



def capacity_gcpc_random_sparse_p(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns,Npatts_lst):
    # avg error over Npatts
    #Npatts_lst = np.arange(1,Npos+1)
    #Npatts_lst = [21]
    err_gcpc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    num_correct = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    l=0
    for Np in (Np_lst):
        print("l = ",l)

        Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
        c = 0.60     # connection probability
        prune = int((1-c)*Np*Ng)
        mask = np.ones((Np, Ng))
        mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
        Wpg = np.multiply(mask, Wpg)
        
        thresh = 0.5
        pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook[:,:10000]), thresh)  # (nruns, Np, Npos)  
        for i in range(nruns):
            pbook_i = pbook[i].flatten()
            np.random.shuffle(pbook_i)
            pbook[i] = pbook_i.reshape((Np,Npos))#10000))#Npos))
        print("pbook shape"+str(pbook.shape))
        k=0
        
        for Npatts in tqdm(Npatts_lst):
            #print("k = ",k)
            Wgp = np.zeros((nruns, Ng, Np));        # plastic pc-to-gc weights

            #Wgp = train_gcpc(pbook_grid, gbook_grid, np.prod(lambdas))
            #Wgp = train_gcpc(pbook, gbook, Npatts)  # Training
            Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)
            
            #Wpg = train_gcpc(gbook, pbook, Npatts)  # Training
            Wpg = pseudotrain_Wpg(gbook, pbook, Npatts)

            # Testing
            sum_gcpc = 0 
            num_corr=0
            #sampledpatt = np.random.choice(Npos,1000)
            for x in range(Npatts): 
            #for x in range(Npos):
            #for x in sampledpatt:
                ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
                pinit = corrupt_pcont(pflip, ptrue)      # make corrupted pc pattern
                #cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                cleanup = gcpc_randomized(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                #print(cleanup.mean())
                #if cleanup[0] > 0:
                    #print(cleanup[0], x)
                    #print(gbook[:,x,None].T)
                sum_gcpc += cleanup
                num_corr += (cleanup<0.03).astype('int')
            #print(sum_gcpc.mean())
            err_gcpc[l,k] = sum_gcpc/Npatts
            num_correct[l,k] = num_corr
            k += 1   
        l += 1    
    return err_gcpc, num_correct                 




def capacity_gcpc_shuffled(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns,Npatts_lst):
    # avg error over Npatts
    #Npatts_lst = np.arange(1,Npos+1)
    #Npatts_lst = [21]
    
    err_gcpc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    num_correct = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    l=0
    for Np in (Np_lst):
        print("l = ",l)

        Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
        c = 0.60     # connection probability
        prune = int((1-c)*Np*Ng)
        mask = np.ones((Np, Ng))
        mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
        Wpg = np.multiply(mask, Wpg)
        
        permutation = np.random.permutation(Npos)
        gbook = gbook[:,permutation]
        print("shuffled")
        thresh = 0.5
        pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)  
        print("pbook shape"+str(pbook.shape))
        k=0
        
        
        for Npatts in tqdm(Npatts_lst):
            #print("k = ",k)
            
            Wgp = np.zeros((nruns, Ng, Np));        # plastic pc-to-gc weights

            #Wgp = train_gcpc(pbook_grid, gbook_grid, np.prod(lambdas))
            Wgp = train_gcpc(pbook, gbook, Npatts)  # Training
            #Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)

            # Testing
            sum_gcpc = 0 
            num_corr=0
            sampledpatt = np.random.choice(Npos,100)
            #for x in range(Npatts): 
            #for x in range(Npos):
            for x in sampledpatt:
                ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
                pinit = corrupt_pcont(pflip, ptrue)      # make corrupted pc pattern
                cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                #if cleanup[0] > 0:
                    #print(cleanup[0], x)
                    #print(gbook[:,x,None].T)
                sum_gcpc += cleanup
                num_corr += (cleanup<0.003).astype('int')
            err_gcpc[l,k] = sum_gcpc/Npatts
            num_correct[l,k] = num_corr
            k += 1   
        l += 1    
    return err_gcpc, num_correct     

def capacity_gcpc_spiral(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns,Npatts_lst):
    # avg error over Npatts
    #Npatts_lst = np.arange(1,Npos+1)
    #Npatts_lst = [21]
    err_gcpc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    num_correct = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    l=0
    
    for Np in (Np_lst):
        print("l = ",l)

        Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
        c = 0.60     # connection probability
        prune = int((1-c)*Np*Ng)
        mask = np.ones((Np, Ng))
        mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
        Wpg = np.multiply(mask, Wpg)
        

        
        thresh = 0.5
        
        pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)  
        print("pbook shape"+str(pbook.shape))
        k=0
        
        
        for Npatts in tqdm(Npatts_lst):
            #print("k = ",k)
            
            Wgp = np.zeros((nruns, Ng, Np));        # plastic pc-to-gc weights

            #Wgp = train_gcpc(pbook_grid, gbook_grid, np.prod(lambdas))
            Wgp = train_gcpc(pbook, gbook, Npatts)  # Training
            #Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)

            # Testing
            sum_gcpc = 0 
            num_corr=0
            sampledpatt = np.random.choice(Npos,1000)
            #for x in range(Npatts): 
            #for x in range(Npos):
            for x in sampledpatt:
                ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
                pinit = corrupt_pcont(pflip, ptrue)      # make corrupted pc pattern
                cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                #if cleanup[0] > 0:
                    #print(cleanup[0], x)
                    #print(gbook[:,x,None].T)
                sum_gcpc += cleanup
                num_corr += (cleanup<0.003).astype('int')
            err_gcpc[l,k] = sum_gcpc/Npatts
            num_correct[l,k] = num_corr
            k += 1   
        l += 1    
    return err_gcpc, num_correct         
    
    
def make_spiral(lambda_prod):
    Npos=int(lambda_prod**2)
    x,y=[(lambda_prod-1)//2],[(lambda_prod-1)//2]
    stepsize=1
    sgn=1
    flag=0
    for i in range(Npos):
        if flag==0:
            for j in range(stepsize):
                x.append(x[-1]+sgn)
                y.append(y[-1])
            for j in range(stepsize):
                x.append(x[-1])
                y.append(y[-1]+sgn)
            sgn=-sgn
            stepsize=stepsize+1
            if stepsize==lambda_prod:
                flag=1
        else:
            for j in range(stepsize-1):
                x.append(x[-1]+sgn)
                y.append(y[-1])
            break
    return np.array(x),np.array(y)
    
    
def capacity_gcpc_vectorized(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns,Npatts_lst,test_generalization='no'):
    # avg error over Npatts
    #Npatts_lst = np.arange(1,Npos+1)
    #Npatts_lst = [21]
    err_gcpc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    num_correct = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    l=0
    for Np in (Np_lst):
        
        Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
        c = 0.60     # connection probability
        prune = int((1-c)*Np*Ng)
        mask = np.ones((Np, Ng))
        mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
        Wpg = np.multiply(mask, Wpg)
        
        thresh = 0.5
        pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)  
        # print("pbook shape"+str(pbook.shape))
        k=0
        for Npatts in tqdm(Npatts_lst):
            # print("k = ",k)
            Wgp = np.zeros((nruns, Ng, Np));        # plastic pc-to-gc weights

            #Wgp = train_gcpc(pbook, gbook, Npatts)  # Training
            Wgp = train_gcpc(pbook+np.random.normal(0,pflip,pbook.shape), gbook, Npatts)  # Training with noise
            #Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)

            # Testing
            sum_gcpc = 0 
            num_corr=0
            
            if test_generalization=='no':
                test_patts = np.arange(Npatts)
            else:
                test_patts = np.arange(Npos)#np.random.choice(Npos,100)
            # 
            # for x in range(Npatts): 
            # sampledpatt = np.random.choice(Npos,100)
            #for x in range(Npos):
            ptrue = pbook[:,:,test_patts]
            pinit = corrupt_pcont(pflip,ptrue)
            cleanup_vectorized = gcpc_vectorized(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
            # print(cleanup_vectorized.shape)
            err_gcpc[l,k] = np.average(cleanup_vectorized,axis=1)
            num_correct[l,k]= np.sum((cleanup_vectorized<0.003).astype('int'), axis=1)
            
            # for x in sampledpatt:
                # ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
                # pinit = corrupt_pcont(pflip, ptrue)      # make corrupted pc pattern
                # cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                # sum_gcpc += cleanup
                # num_corr += (cleanup<0.003).astype('int')
            # err_gcpc[l,k] = sum_gcpc/Npatts
            # num_correct[l,k] = num_corr
            k += 1   
        l += 1    
    return err_gcpc, num_correct    
    

def gcpc_vectorized(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh):
    p = np.copy(pinit)
    for i in range(Niter):
        gin = Wgp@p
        g = gridCAN_2d(gin,lambdas)
        p = nonlin(Wpg@g, thresh)
    # print(p.shape)
    return np.linalg.norm(p-ptrue, axis=(1))/Np


def capacity_gcpc_random_sparse_p_both_random(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns,Npatts_lst):
    # avg error over Npatts
    #Npatts_lst = np.arange(1,Npos+1)
    #Npatts_lst = [21]
    err_gcpc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    num_correct = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    l=0
    print("both random")
    for Np in (Np_lst):
        print("l = ",l)

        Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
        c = 0.60     # connection probability
        prune = int((1-c)*Np*Ng)
        mask = np.ones((Np, Ng))
        mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
        Wpg = np.multiply(mask, Wpg)
        
        thresh = 0.5
        pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)  
        for i in range(nruns):
            pbook_i = pbook[i].flatten()
            np.random.shuffle(pbook_i)
            pbook[i] = pbook_i.reshape((Np,Npos))#10000))#Npos))
        print("pbook shape"+str(pbook.shape))

        
        Wgp = randn(nruns, Ng, Np);           # fixed random gc-to-pc weights
        c = 0.60     # connection probability
        prune = int((1-c)*Np*Ng)
        mask = np.ones((Ng, Np))
        mask[randint(low=0, high=Ng, size=prune), randint(low=0, high=Np, size=prune)] = 0
        Wgp = np.multiply(mask, Wgp)
        new_gbook = nonlin(np.einsum('ijk,kl->ijl', Wgp, pbook[0]), 0)  # (nruns, Np, Npos)  
        
        
        k=0
        
        for Npatts in tqdm(Npatts_lst):
            #print("k = ",k)
            
            #Wpg = train_gcpc(gbook, pbook, Npatts)  # Training
            Wpg = pseudotrain_Wpg(new_gbook[0], pbook, Npatts)
            
            

            # Testing
            sum_gcpc = 0 
            num_corr=0
            #sampledpatt = np.random.choice(Npos,1000)
            for x in range(Npatts): 
            #for x in range(Npos):
            #for x in sampledpatt:
                ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
                pinit = corrupt_pcont(pflip, ptrue)      # make corrupted pc pattern
                #cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                # cleanup = gcpc_randomized(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                cleanup = gcpc_randomized_both(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                #print(cleanup.mean())
                #if cleanup[0] > 0:
                    #print(cleanup[0], x)
                    #print(gbook[:,x,None].T)
                sum_gcpc += cleanup
                num_corr += (cleanup<0.03).astype('int')
            #print(sum_gcpc.mean())
            err_gcpc[l,k] = sum_gcpc/Npatts
            num_correct[l,k] = num_corr
            k += 1   
        l += 1    
    return err_gcpc, num_correct  
    

def capacity_gcpc_random_sparse_p_both_random_2(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns,Npatts_lst):
    # avg error over Npatts
    #Npatts_lst = np.arange(1,Npos+1)
    #Npatts_lst = [21]
    err_gcpc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    num_correct = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    l=0
    print("both random")
    for Np in (Np_lst):
        print("l = ",l)

        Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
        c = 0.60     # connection probability
        prune = int((1-c)*Np*Ng)
        mask = np.ones((Np, Ng))
        mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
        Wpg = np.multiply(mask, Wpg)
        
        thresh = 0.5
        pbook = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)  
        for i in range(nruns):
            pbook_i = pbook[i].flatten()
            np.random.shuffle(pbook_i)
            pbook[i] = pbook_i.reshape((Np,Npos))#10000))#Npos))
        print("pbook shape"+str(pbook.shape))

        
        Wgp = randn(nruns, Ng, Np);           # fixed random gc-to-pc weights
        c = 0.60     # connection probability
        prune = int((1-c)*Np*Ng)
        mask = np.ones((Ng, Np))
        mask[randint(low=0, high=Ng, size=prune), randint(low=0, high=Np, size=prune)] = 0
        Wgp = np.multiply(mask, Wgp)
        #new_gbook = nonlin(np.einsum('ijk,kl->ijl', Wgp, pbook[0]), 0)  # (nruns, Np, Npos)  
        new_gbook = gridCAN_2d(np.einsum('ijk,kl->ijl', Wgp, pbook[0]),lambdas)
        
        k=0
        
        for Npatts in tqdm(Npatts_lst):
            #print("k = ",k)
            
            #Wpg = train_gcpc(gbook, pbook, Npatts)  # Training
            Wpg = pseudotrain_Wpg(new_gbook[0], pbook, Npatts)
            
            

            # Testing
            sum_gcpc = 0 
            num_corr=0
            #sampledpatt = np.random.choice(Npos,1000)
            for x in range(Npatts): 
            #for x in range(Npos):
            #for x in sampledpatt:
                ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
                pinit = corrupt_pcont(pflip, ptrue)      # make corrupted pc pattern
                #cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                # cleanup = gcpc_randomized(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                cleanup = gcpc_randomized_both(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                cleanup = gcpc_randomized(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
                #print(cleanup.mean())
                #if cleanup[0] > 0:
                    #print(cleanup[0], x)
                    #print(gbook[:,x,None].T)
                sum_gcpc += cleanup
                num_corr += (cleanup<0.03).astype('int')
            #print(sum_gcpc.mean())
            err_gcpc[l,k] = sum_gcpc/Npatts
            num_correct[l,k] = num_corr
            k += 1   
        l += 1    
    return err_gcpc, num_correct  