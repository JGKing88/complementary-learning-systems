import numpy as np
import torch
from typing import Optional, TYPE_CHECKING
from cls.vectorhash.seq_utils import *
from cls.vectorhash.assoc_utils_np import *
from cls.vectorhash.senstranspose_utils import *
from cls.vectorhash.assoc_utils_np_2D import gen_gbook_2d, path_integration_Wgg_2d, module_wise_NN_2d
from cls.hopfield import Hopfield

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import copy


def onehot2d_to_gaussian_np(x: np.ndarray,
                            sigma: float,
                            sigmay: float | None = None,
                            wrap: bool = True,
                            normalize: str = "none") -> np.ndarray:
    """
    Turn a one-hot 2D map into a wrapped 2D Gaussian bump.

    Args:
        x         : 2D array (H, W), assumed one-hot (but any single nonzero works).
        sigma     : std dev along x (columns), in pixels.
        sigmay    : std dev along y (rows); if None, uses sigma.
        wrap      : wrap distances horizontally and vertically (toroidal).
        normalize : "none" | "max" | "sum"

    Returns:
        2D float array (H, W)
    """
    x = x.astype(float, copy=False)
    H, W = x.shape
    if H == 0 or W == 0 or sigma <= 0:
        return np.zeros_like(x, dtype=float)

    pts = np.argwhere(x != 0)
    if pts.size == 0:
        return np.zeros_like(x, dtype=float)

    sx = float(sigma)
    sy = float(sigmay if sigmay is not None else sigma)

    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]

    out = np.zeros((H, W), dtype=float)

    for (cy, cx) in pts:
        dy = np.abs(rows - cy)
        dx = np.abs(cols - cx)
        if wrap:
            dy = np.minimum(dy, H - dy)
            dx = np.minimum(dx, W - dx)

        g = np.exp(-0.5 * ((dy / sy) ** 2 + (dx / sx) ** 2))
        out += x[cy, cx] * g

    if normalize == "max":
        m = out.max()
        if m > 0: out /= m
    elif normalize == "sum":
        s = out.sum()
        if s > 0: out /= s

    return out


def smooth_g(g: np.ndarray, lambdas: list, fwhm_ratio: float) -> np.ndarray:
    """
    Smooth a one-hot grid vector by converting each module to a Gaussian bump.
    
    Args:
        g: Grid vector (concatenated one-hots across modules)
        lambdas: List of module periods
        fwhm_ratio: Ratio of FWHM to lambda (e.g., 0.25 means FWHM is 1/4 of lambda)
    
    Returns:
        Smoothed grid vector
    """
    if fwhm_ratio <= 0:
        return g.copy()
    
    gin = g.copy()
    gout = np.zeros_like(gin, dtype=np.float32)
    i = 0
    for l in lambdas:
        fwhm = l * fwhm_ratio
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        two_d_gin = gin[i:i+l**2].reshape(l, l)
        gout[i:i+l**2] = onehot2d_to_gaussian_np(two_d_gin, sigma).flatten()
        i += l**2
    return gout

def smooth_gbook(gbook, lambdas, fwhm_ratio):
    smooth_gbook = np.zeros_like(gbook)
    for i in range(gbook.shape[1]):
        for j in range(gbook.shape[2]):
            smooth_gbook[:, i, j] = smooth_g(gbook[:, i, j], lambdas, fwhm_ratio)
    return smooth_gbook

def overlaps(x, y, px, py, size, touch_ok=True):
    if touch_ok:
        # touching edges allowed
        return not (x + size <= px or px + size <= x or y + size <= py or py + size <= y)
    else:
        # touching counts as overlap
        return not (x + size <  px or px + size <  x or y + size <  py or py + size <  y)


class VectorHash:
    def __init__(self, Np, lambdas, size, Npos = None, use_hopfield: bool = False, hopfield_gain: float = 2.0, hopfield_alpha: float = 1.0, hopfield_steps: int = 1, thresh: float = 2.0, use_headings: bool = False):
        self.thresh = thresh
        self.c = 0.5
        self.Np = Np
        self.lambdas = lambdas
        self.Ng = np.sum(np.square(lambdas))
        self.Npos = Npos if Npos is not None else np.prod(lambdas)
        self.size = size
        self.use_headings = use_headings
        self.envs = []
        # Hopfield settings
        self.use_hopfield = use_hopfield
        self.hopfield_gain = hopfield_gain
        self.hopfield_alpha = hopfield_alpha
        self.hopfield_steps = hopfield_steps
        self.hopfield: Optional["Hopfield"] = None
    
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
    
    def setup_scaffold(self, Np, lambdas, thresh):
        
        print("      gen_gbook_2d")
        gbook = gen_gbook_2d(lambdas, self.Ng, self.Npos)
        print(f"gbook shape: {gbook.shape}")     # (Ng, Npos, Npos)

        module_sizes = np.square(lambdas)
        module_gbooks = [np.eye(i) for i in module_sizes]

        Wpg = randn(self.Np, self.Ng) 

        prune = int((1-self.c)*self.Np*self.Ng)
        mask = np.ones((self.Np, self.Ng))
        mask[randint(low=0, high=Np, size=prune), randint(low=0, high=self.Ng, size=prune)] = 0
        Wpg = np.multiply(mask, Wpg)

        print(f"Wpg shape: {Wpg.shape}")
        print("      pbook")
        # pbook = nonlin(np.einsum('jk,klm->jlm', Wpg, gbook), thresh=thresh)  # (Np, Npos, Npos) 

        pbook = nonlin(train_pbook(Wpg, gbook), thresh=thresh) # (Np, Npos, Npos) 
        print(f"pbook shape: {pbook.shape}")
        gbook_flattened = gbook.reshape(self.Ng, self.Npos*self.Npos)  #order='F'
        pbook_flattened = pbook.reshape(self.Np, self.Npos*self.Npos)

        print("      train_gcpc")
        Wgp = train_gcpc(pbook_flattened, gbook_flattened,Npatts=self.Npos*self.Npos)
        print(f"Wgp shape: {Wgp.shape}")

        return pbook, pbook_flattened, gbook, gbook_flattened, Wpg, Wgp, module_sizes, module_gbooks
    
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
            if all(not overlaps(x, y, px, py, size, touch_ok) for (px, py) in used):
                used.append((x, y))
                C_pairs.append((x, y))
            tries += 1

        if len(C_pairs) < n_envs:
            raise RuntimeError(f"Could only place {len(C_pairs)}/{n_envs} squares; try fewer envs or smaller size.")

        all_path_locations = []
        all_observations = []
        abook = []

        self.env_locations = []

        for env_idx, env in enumerate(envs):
            pos_obs_head = env.fully_explore_random()
            if not self.use_headings:
                # Filter to single heading (1,0) - multiple headings break Wps pseudoinverse
                pos_obs_head = [poh for poh in pos_obs_head if poh[2] == (1, 0)]
            path_locations = np.array([poh[0] for poh in pos_obs_head])
            observations = np.array([poh[1] for poh in pos_obs_head])
            C_X, C_Y = C_pairs[env_idx]
            self.env_locations.append((C_X, C_Y))
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

    def get_loc_from_grid_state(self, g):
        return self.g_to_location[tuple(g)]

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
        # Validate use_headings consistency
        for env in envs:
            if hasattr(env, 'use_headings') and env.use_headings and not self.use_headings:
                raise ValueError(
                    f"Env has use_headings=True but VectorHash has use_headings=False. "
                    f"VectorHash cannot handle heading-dependent observations without use_headings=True."
                )

        Np = self.Np
        lambdas = self.lambdas
        thresh = self.thresh
        c = self.c
        size = self.size
        n_envs = len(envs)

        # Setup scaffold and environment encodings
        print("   setup scaffold")
        self.pbook, self.pbook_flattened, self.gbook, self.gbook_flattened, self.Wpg, self.Wgp, self.module_sizes, self.module_gbooks = self.setup_scaffold(Np, lambdas, thresh)

        #for debugging
        self.g_to_location = {}
        width,height = self.gbook.shape[1],self.gbook.shape[2]
        gbook_copy = self.gbook.copy()
        for x in range(width):
            for y in range(height):
                self.g_to_location[tuple(gbook_copy[:, x, y])] = (x, y)

        print("   setup envs")
        self.path_sbook, self.path_pbook, self.path_gbook, self.Wsp, self.Wps = self.setup_envs(
            envs, size, n_envs, self.Npos, self.Ng, self.pbook, self.gbook
        )

        self.Ns = self.path_sbook.shape[0]
        self.Np = self.pbook.shape[0]
        self.Ng = self.gbook.shape[0]

        print("   initialize envs vh")
        for env in envs:
            env.initiate_vectorhash(self)
        
        self.envs = envs
        
        # Initialize Hopfield network if requested
        if self.use_hopfield:
            self._init_hopfield(envs)
        
        # Verify scaffold integrity
        self.test_vectorhash()
    
    def recall(self, obs):

        Ns = self.path_sbook.shape[0]
        Np = self.pbook.shape[0]
        Ng = self.gbook.shape[0]

        pin = nonlin(self.Wps@obs, thresh=self.thresh)

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

    def _init_hopfield(self, envs):
        """Initialize Hopfield network and store goal patterns from all environments."""
        
        
        # Get pattern dimension from first env's observation
        pattern_dim = envs[0].get_input_size()
        print(f"   initializing Hopfield network (units={pattern_dim}, gain={self.hopfield_gain})")
        
        self.hopfield = Hopfield(
            num_units=pattern_dim,
            beta=self.hopfield_gain,
            device=device,
        )
        
        # Store goal patterns from all environments
        print("   storing goal patterns in Hopfield network...")
        for env in envs:
            goal_obs = env.obs_at_goal()
            goal_t = torch.from_numpy(goal_obs).float()
            self.hopfield.input_memory(goal_t)
        
        print(f"   stored {self.hopfield.num_memories} goal patterns")
    
    def hopfield_recall(self, obs: np.ndarray) -> np.ndarray:
        """Recall from Hopfield network for a single observation.
        
        Args:
            obs: Observation array of shape (D,)
            
        Returns:
            Recalled pattern of shape (D,)
        """
        if self.hopfield is None:
            raise ValueError("Hopfield network not initialized. Set use_hopfield=True.")
        
        obs_t = torch.from_numpy(obs).float().to(device)
        recalled, _ = self.hopfield.recall(obs_t, steps=self.hopfield_steps, alpha=self.hopfield_alpha)
        return recalled.cpu().numpy()
    
    def hopfield_recall_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Recall from Hopfield network for a batch of observations.
        
        Args:
            obs_batch: Observation array of shape (B, D)
            
        Returns:
            Recalled patterns of shape (B, D)
        """
        if self.hopfield is None:
            raise ValueError("Hopfield network not initialized. Set use_hopfield=True.")
        
        recalled_list = []
        for i in range(obs_batch.shape[0]):
            obs_t = torch.from_numpy(obs_batch[i]).float().to(device)
            recalled, _ = self.hopfield.recall(obs_t, steps=self.hopfield_steps, alpha=self.hopfield_alpha)
            recalled_list.append(recalled.cpu().numpy())
        return np.stack(recalled_list, axis=0)

    def test_vectorhash(self):
        """Test that recall() correctly recovers grid state from sensory input.
        
        For every explored position, verifies:
            sensory → recall() → grid_recovered == grid_true
        
        Raises:
            RuntimeError: If any position fails to recover the correct grid state.
        """
        n_samples = self.path_sbook.shape[1]
        n_correct = 0
        n_p_correct = 0  # Track HPC accuracy separately
        failures = []
        
        print(f"   testing vectorhash recall on {n_samples} samples...")
        print(f"   shapes: path_sbook={self.path_sbook.shape}, path_pbook={self.path_pbook.shape}, path_gbook={self.path_gbook.shape}")
        print(f"   Wps={self.Wps.shape}, Wgp={self.Wgp.shape}, thresh={self.thresh}")
        
        # Check if sensory observations are unique
        n_unique_s = len(set(tuple(self.path_sbook[:, i]) for i in range(n_samples)))
        n_unique_p = len(set(tuple(self.path_pbook[:, i]) for i in range(n_samples)))
        print(f"   unique s: {n_unique_s}/{n_samples}, unique p: {n_unique_p}/{n_samples}")
        
        n_pin_correct = 0  # Direct s→p (before g correction)
        n_pin_argmax = 0   # Does argmax match?
        for i in range(n_samples):
            # Get sensory input for this position
            s = self.path_sbook[:, i]
            
            # Check DIRECT s→p (before g→p correction)
            p_raw = self.Wps @ s  # Before nonlin
            pin = nonlin(p_raw, thresh=self.thresh)
            p_true = self.path_pbook[:, i]
            
            if np.array_equal(pin, p_true):
                n_pin_correct += 1
            # Check if at least the argmax matches (direction is right)
            if np.argmax(p_raw) == np.argmax(p_true):
                n_pin_argmax += 1
            
            # Print first few examples for debugging
            if i < 3:
                print(f"   sample {i}: p_raw max={p_raw.max():.3f} min={p_raw.min():.3f} argmax={np.argmax(p_raw)} | p_true argmax={np.argmax(p_true)} nnz={np.sum(p_true>0)}")
            
            # Use the actual recall function
            s_out, p_out, g_recovered = self.recall(s)
            
            # Compare recovered grid to ground truth
            g_true = self.path_gbook[:, i]
            
            # Check round-trip p (s→p→g→p)
            if np.array_equal(p_out, p_true):
                n_p_correct += 1
            
            if np.array_equal(g_recovered, g_true):
                n_correct += 1
            else:
                failures.append(i)
        
        accuracy = n_correct / n_samples
        pin_accuracy = n_pin_correct / n_samples
        pin_argmax_acc = n_pin_argmax / n_samples
        p_accuracy = n_p_correct / n_samples
        print(f"   s→p exact: {n_pin_correct}/{n_samples} ({pin_accuracy*100:.1f}%)")
        print(f"   s→p argmax: {n_pin_argmax}/{n_samples} ({pin_argmax_acc*100:.1f}%)")
        print(f"   s→p→g→p roundtrip: {n_p_correct}/{n_samples} ({p_accuracy*100:.1f}%)")
        print(f"   grid recovery: {n_correct}/{n_samples} ({accuracy*100:.1f}%)")
        
        # Additional diagnostic: test p→g pathway directly (bypassing sensory)
        n_pg_correct = 0
        n_gp_correct = 0  # g→p direction
        for i in range(n_samples):
            p_true = self.path_pbook[:, i]
            g_true = self.path_gbook[:, i]
            
            # Direct p→g (same logic as recall)
            gin = self.Wgp @ p_true
            ls = [l**2 for l in self.lambdas]
            idx = 0
            gout = np.zeros(gin.shape)
            for j in ls:
                gmod = gin[idx:idx+j]
                maxes = gmod.argmax()
                gout[maxes+idx] = 1
                idx += j
            
            if np.array_equal(gout, g_true):
                n_pg_correct += 1
            
            # Direct g→p (Wpg @ g_true should give p_true)
            p_from_g = nonlin(self.Wpg @ g_true, thresh=self.thresh)
            if np.allclose(p_from_g, p_true):
                n_gp_correct += 1
        
        pg_accuracy = n_pg_correct / n_samples
        gp_accuracy = n_gp_correct / n_samples
        print(f"   p→g direct: {n_pg_correct}/{n_samples} correct ({pg_accuracy*100:.1f}%)")
        print(f"   g→p direct: {n_gp_correct}/{n_samples} correct ({gp_accuracy*100:.1f}%)")
        
        if failures:
            raise RuntimeError(
                f"VectorHash scaffold test FAILED: {len(failures)}/{n_samples} positions "
                f"({len(failures)/n_samples*100:.1f}%) did not recover correct grid state. "
                f"First 10 failed indices: {failures[:10]}"
            )
