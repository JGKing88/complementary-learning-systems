import numpy as np
from cls.vectorhash.seq_utils import *
from cls.vectorhash.assoc_utils_np import *
from cls.vectorhash.senstranspose_utils import *
from cls.vectorhash.assoc_utils_np_2D import gen_gbook_2d, path_integration_Wgg_2d, module_wise_NN_2d
    
class VectorHash:
    def __init__(self, Np, lambdas, size):
        self.thresh = 2.0
        self.c = 1
        self.Np = Np
        self.lambdas = lambdas
        self.size = size
        self.envs = []
    
    def grid_onehot_to_indices(self, g):
        ls = [l**2 for l in self.lambdas]
        indices = np.zeros(len(ls)*2, dtype=int)
        start = 0
        for l_idx, n in enumerate(self.lambdas):
            size = n**2
            onehot = g[start:start+size].reshape(n, n)
            y, x = np.argwhere(onehot == 1)[0]
            indices[2*l_idx : 2*l_idx+2] = (y, x)
            start += size
        return indices

    def setup_scaffold(self, Np, lambdas, thresh, c):
        Ng = np.sum(np.square(lambdas))
        Npos = np.prod(lambdas)
        gbook = gen_gbook_2d(lambdas, Ng, Npos)
        gbook.shape     # (Ng, Npos, Npos)

        module_sizes = np.square(lambdas)
        module_gbooks = [np.eye(i) for i in module_sizes]

        Wpg = randn(Np, Ng) 

        prune = int((1-c)*Np*Ng)
        mask = np.ones((Np, Ng))
        mask[randint(low=0, high=Np, size=prune), randint(low=0, high=Ng, size=prune)] = 0
        Wpg = np.multiply(mask, Wpg)
        
        pbook = nonlin(np.einsum('jk,klm->jlm', Wpg, gbook), thresh=thresh)  # (Np, Npos, Npos) 
        gbook_flattened = gbook.reshape(Ng, Npos*Npos)  #order='F'
        pbook_flattened = pbook.reshape(Np, Npos*Npos)

        Wgp = train_gcpc(pbook_flattened, gbook_flattened,Npatts=Npos*Npos)

        return pbook, pbook_flattened, gbook, gbook_flattened, Wpg, Wgp, module_sizes, module_gbooks, Npos, Ng
    
    def overlaps(self, x, y, px, py, size, touch_ok=True):
        if touch_ok:
            # touching edges allowed
            return not (x + size <= px or px + size <= x or y + size <= py or py + size <= y)
        else:
            # touching counts as overlap
            return not (x + size <  px or px + size <  x or y + size <  py or py + size <  y)

    def setup_envs(self, envs, size, n_envs, Npos, Ng, pbook, gbook):
        # Choose non-overlapping bottom-left corners for size x size grids in Npos x Npos grid
        used = []  # store placed (x, y)
        C_pairs = []
        max_tries = 10_000  # guard against infinite loops
        touch_ok = True      # set False to forbid touching
        tries = 0
        while len(C_pairs) < n_envs and tries < max_tries:
            x = np.random.randint(0, Npos - size + 1)
            y = np.random.randint(0, Npos - size + 1)
            if all(not self.overlaps(x, y, px, py, size, touch_ok) for (px, py) in used):
                used.append((x, y))
                C_pairs.append((x, y))
            tries += 1

        if len(C_pairs) < n_envs:
            raise RuntimeError(f"Could only place {len(C_pairs)}/{n_envs} squares; try fewer envs or smaller size.")

        all_path_locations = []
        all_observations = []
        abook = []

        for env_idx, env in enumerate(envs):
            pos_obs_head = env.fully_explore_random()
            path_locations = np.array([poh[0] for poh in pos_obs_head]) # if poh[2] == (1, 0)])
            observations = np.array([poh[1] for poh in pos_obs_head]) # if poh[2] == (1, 0)])
            C_X, C_Y = C_pairs[env_idx]
            path_locations[:,0] = path_locations[:,0] + C_X
            path_locations[:,1] = path_locations[:,1] + C_Y

            all_path_locations.append(path_locations)
            all_observations.append(observations)

        all_path_locations = np.concatenate(all_path_locations, axis=0)
        all_observations = np.concatenate(all_observations, axis=0)
        path_sbook = all_observations.T

        #pbook.shape: (Np, Npos, Npos)
        Npatts = len(all_path_locations)
        Np = pbook.shape[0]
        path_pbook = np.zeros((Np, Npatts))
        path_gbook = np.zeros((Ng, Npatts))
        k = 0
        for i in all_path_locations:
            path_pbook[:,k] = pbook[:,i[0],i[1]]
            path_gbook[:,k] = gbook[:,i[0],i[1]]
            k = k+1

        Wsp = pseudotrain_Wsp(path_sbook, path_pbook, Npatts)
        Wps = pseudotrain_Wps(path_pbook, path_sbook, Npatts)

        return path_sbook, path_pbook, path_gbook, Wsp, Wps


    def initiate_vectorhash(self, envs):
        """
        Initializes vector hash representations for a set of environments.

        Args:
            envs: List of environment instances.
            size (int): Grid size.
            speed (int): Movement speed.
            n_envs (int): Number of environments.
            lambdas (list): List of module sizes.
            Np (int): Number of patterns.
            thresh (float): Threshold for nonlinearity.
            c (int): Unused parameter (reserved for future use).

        Returns:
            tuple: (path_sbook, path_pbook, path_gbook, Wsp, Wps)
        """

        Np = self.Np
        lambdas = self.lambdas
        thresh = self.thresh
        c = self.c
        size = self.size
        n_envs = len(envs)

        # Setup scaffold and environment encodings
        self.pbook, self.pbook_flattened, self.gbook, self.gbook_flattened, self.Wpg, self.Wgp, self.module_sizes, self.module_gbooks, self.Npos, self.Ng = self.setup_scaffold(Np, lambdas, thresh, c)

        self.path_sbook, self.path_pbook, self.path_gbook, self.Wsp, self.Wps = self.setup_envs(
            envs, size, n_envs, self.Npos, self.Ng, self.pbook, self.gbook
        )

        self.Ns = self.path_sbook.shape[0]
        self.Np = self.pbook.shape[0]
        self.Ng = self.gbook.shape[0]

        for env in envs:
            env.initiate_vectorhash(self)
        
        self.envs = envs
    
    def recall(self, obs):

        Ns = self.path_sbook.shape[0]
        Np = self.pbook.shape[0]
        Ng = self.gbook.shape[0]

        pin = nonlin(self.Wps@obs, thresh=0)

        p = np.copy(pin)
        gin = self.Wgp@p

        Ng = gin.shape[0]
        ls = [l**2 for l in self.lambdas]
        i=0
        gout = np.zeros(gin.shape)
        for j in ls:
            gmod = gin[i:i+j]
            maxes = gmod.argmax()
            gout[maxes+i] = 1
            i=i+j

        pout = nonlin(self.Wpg@gout, thresh=self.thresh)
        pout = np.copy(pout)
        gout = np.copy(gout)
        sout = (self.Wsp@pout > 0).astype(float)
        
        return sout,pout,gout

    def test_vectorhash(self):

        Ns = self.path_sbook.shape[0]
        Np = self.pbook.shape[0]
        Ng = self.gbook.shape[0]

        pin = nonlin(self.Wps@self.path_sbook, thresh=0)

        p = np.copy(pin)
        gin = self.Wgp@p

        Ng,Npatts = gin.shape
        ls = [l**2 for l in self.lambdas]
        i=0
        gout = np.zeros(gin.shape)
        for j in ls:
            gmod = gin[i:i+j,:]
            maxes = gmod.argmax(axis=0)
            gout[maxes+i,np.arange(Npatts)] = 1
            i=i+j

        pout = nonlin(self.Wpg@gout, thresh=self.thresh)
        pout = np.copy(pout)
        gout = np.copy(gout)
        sout = (self.Wsp @ pout > 0).astype(float)

        strue=self.path_sbook
        ptrue=self.path_pbook
        gtrue=self.path_gbook

        p2_l2_err = np.linalg.norm(pout-ptrue,axis=0)
        p1_l2_err = np.linalg.norm(pin-pout,axis=0)

        g_error = []
        for i in range(gout.shape[1]):
            if not np.all(gout[:,i] == gtrue[:,i]):
                g_error.append(0)
            else:
                g_error.append(1)
        
        traversal = []
        for g_idx in range(gout.shape[1]):
            g = gout[:,g_idx]
            matches = np.all(np.isclose(self.gbook, g[:, None, None], rtol=1e-6, atol=1e-8), axis=0)
            idxs = np.argwhere(matches)
            traversal.append(idxs)
        traversal = np.array(traversal)    
    
        return g_error,p1_l2_err, p2_l2_err,traversal
