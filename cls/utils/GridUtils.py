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
    """Smooth all gbook positions. Vectorized over positions per module."""
    if fwhm_ratio <= 0:
        return gbook.copy()

    Ng, Npos1, Npos2 = gbook.shape
    out = np.zeros_like(gbook, dtype=np.float32)

    offset = 0
    for l in lambdas:
        n = l * l
        fwhm = l * fwhm_ratio
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # Extract module block: (n, Npos1, Npos2) -> reshape to (Npos1*Npos2, l, l)
        block = gbook[offset:offset + n].reshape(l, l, Npos1 * Npos2)  # (l, l, N)
        block = block.transpose(2, 0, 1)  # (N, l, l)

        # Find active pixel per position (each is one-hot)
        flat = block.reshape(block.shape[0], -1)  # (N, l*l)
        active = np.argmax(flat, axis=1)  # (N,)
        cy = active // l  # row
        cx = active % l   # col

        # Compute wrapped distances for all positions at once
        # cy: (N,), rows_1d: (l,) -> dy: (N, l, 1)
        rows_1d = np.arange(l)
        cols_1d = np.arange(l)
        dy = np.abs(rows_1d[None, :, None] - cy[:, None, None])  # (N, l, 1)
        dy = np.minimum(dy, l - dy)
        dx = np.abs(cols_1d[None, None, :] - cx[:, None, None])  # (N, 1, l)
        dx = np.minimum(dx, l - dx)

        bumps = np.exp(-0.5 * ((dy / sigma) ** 2 + (dx / sigma) ** 2))  # (N, l, l)

        # Reshape back: (N, l, l) -> (l, l, N) -> (l*l, Npos1, Npos2)
        bumps = bumps.transpose(1, 2, 0).reshape(n, Npos1, Npos2)
        out[offset:offset + n] = bumps
        offset += n

    return out

def gram_schmidt_2d_batch(d_forward, d_right):
    """Batched Gram-Schmidt: compute 2D projection matrices from forward/right displacement vectors.

    Args:
        d_forward: (B, D) array — forward (North) displacement vectors
        d_right:   (B, D) array — right (East) displacement vectors

    Returns:
        W: (B, 2, D) projection matrices. Row 0 = "right" (East) basis, Row 1 = "forward" (North) basis.
    """
    # Normalize forward to get e1
    norms_f = np.linalg.norm(d_forward, axis=1, keepdims=True)
    norms_f = np.maximum(norms_f, 1e-12)
    e1 = d_forward / norms_f

    # Orthogonalize right against forward
    dots = np.sum(d_right * e1, axis=1, keepdims=True)
    e2 = d_right - dots * e1
    norms_r = np.linalg.norm(e2, axis=1, keepdims=True)
    norms_r = np.maximum(norms_r, 1e-12)
    e2 = e2 / norms_r

    # Stack: row 0 = e2 (right/East), row 1 = e1 (forward/North)
    W = np.stack([e2, e1], axis=1)  # (B, 2, D)
    return W


def classify_direction_batch(q):
    """Batched angle classification of 2D projected vectors into compass directions.

    Args:
        q: (B, 2) array — projected 2D vectors [x, y]

    Returns:
        direction_idx: (B,) int array — 0=N, 1=E, 2=S, 3=W
    """
    angles = np.arctan2(q[:, 1], q[:, 0])  # (B,)

    direction_idx = np.full(angles.shape, 3, dtype=np.int32)  # default W (|angle| >= 3pi/4)
    direction_idx[(-np.pi/4 <= angles) & (angles < np.pi/4)] = 1       # E (right)
    direction_idx[(np.pi/4 <= angles) & (angles < 3*np.pi/4)] = 0      # N (forward)
    direction_idx[(-3*np.pi/4 <= angles) & (angles < -np.pi/4)] = 2    # S (backward)

    return direction_idx


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
        return self.g_to_location[np.asarray(g, dtype=np.float64).tobytes()]

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

        #for debugging — use tobytes() for fast hashing instead of tuple()
        self.g_to_location = {}
        width, height = self.gbook.shape[1], self.gbook.shape[2]
        for x in range(width):
            for y in range(height):
                self.g_to_location[self.gbook[:, x, y].tobytes()] = (x, y)

        print("   setup envs")
        self.path_sbook, self.path_pbook, self.path_gbook, self.Wsp, self.Wps = self.setup_envs(
            envs, size, n_envs, self.Npos, self.Ng, self.pbook, self.gbook
        )

        self.Ns = self.path_sbook.shape[0]
        self.Np = self.pbook.shape[0]
        self.Ng = self.gbook.shape[0]

        print("   initialize envs vh")
        for env_idx, env in enumerate(envs):
            env.initiate_vectorhash(self, env_idx=env_idx)
        
        self.envs = envs
        
        # Initialize Hopfield network if requested
        # For hopfield_onehot/hopfield_proj, defer to init_hopfield_encoded (after precompute_encoded_phi)
        input_type = getattr(envs[0], 'input_type', None) if envs else None
        if self.use_hopfield and input_type not in ("hopfield_onehot", "hopfield_proj"):
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

    def precompute_encoded_phi(self, encoder, fwhm_ratio, device):
        """Encode all gbook positions through the encoder and store as encoded_Phi.

        Args:
            encoder: GridEncoder model (already on device, in eval mode)
            fwhm_ratio: Smoothing ratio (0 = no smoothing)
            device: torch.device for encoder forward pass
        """
        Npos = self.gbook.shape[1]

        if fwhm_ratio > 0:
            sgb = smooth_gbook(self.gbook, self.lambdas, fwhm_ratio)
        else:
            sgb = self.gbook.copy()

        # Reshape from (Ng, Npos, Npos) -> (Npos*Npos, Ng)
        flat = sgb.reshape(self.Ng, Npos * Npos).T.astype(np.float32)  # (Npos*Npos, Ng)

        # Encode in batches
        encoded_parts = []
        batch_size = 1000
        with torch.no_grad():
            for start in range(0, flat.shape[0], batch_size):
                chunk = torch.from_numpy(flat[start:start + batch_size]).to(device)
                enc = encoder(chunk).cpu().numpy()
                encoded_parts.append(enc)

        encoded_flat = np.concatenate(encoded_parts, axis=0)  # (Npos*Npos, embed_dim)
        embed_dim = encoded_flat.shape[1]
        self.encoded_Phi = encoded_flat.reshape(Npos, Npos, embed_dim)
        print(f"   precomputed encoded_Phi: shape={self.encoded_Phi.shape}")

    def init_hopfield_encoded(self, envs):
        """Initialize Hopfield network in encoded space using encoded_Phi and goal positions.

        Must be called after precompute_encoded_phi. Stores encoded goal patterns
        looked up from encoded_Phi using each env's goal location + offset.
        """
        embed_dim = self.encoded_Phi.shape[2]
        print(f"   initializing Hopfield network in encoded space (units={embed_dim}, gain={self.hopfield_gain})")

        self.hopfield = Hopfield(
            num_units=embed_dim,
            beta=self.hopfield_gain,
            device=device,
        )

        print("   storing encoded goal patterns in Hopfield network...")
        for env_idx, env in enumerate(envs):
            offset = self.env_locations[env_idx] if env_idx < len(self.env_locations) else (0, 0)
            gx = env.goal_location[0] + offset[0]
            gy = env.goal_location[1] + offset[1]
            # Clamp to valid range
            gx = min(max(gx, 0), self.encoded_Phi.shape[0] - 1)
            gy = min(max(gy, 0), self.encoded_Phi.shape[1] - 1)
            goal_encoded = self.encoded_Phi[gx, gy]  # (embed_dim,)
            goal_t = torch.from_numpy(goal_encoded).float()
            self.hopfield.input_memory(goal_t)

        print(f"   stored {self.hopfield.num_memories} encoded goal patterns")

    def compute_hopfield_direction_batch(
        self,
        local_positions,
        env_offset,
        cached_W=None,
        recompute_mask=None,
        return_proj=False,
    ):
        """Compute Hopfield-based direction classification for a batch of positions.

        Args:
            local_positions: (B, 2) local coords within env
            env_offset:      (C_X, C_Y) tuple — env's offset in global grid
            cached_W:        (B, 2, embed_dim) or None — cached projection matrices
            recompute_mask:  (B,) bool — which slots need fresh W computation
            return_proj:     whether to also return the 2D projected vectors

        Returns:
            onehot:   (B, 4) one-hot direction vectors
            proj:     (B, 2) projected vectors or None (if return_proj=False)
            W:        (B, 2, embed_dim) updated projection matrices
        """
        B = local_positions.shape[0]
        C_X, C_Y = env_offset
        Npos = self.encoded_Phi.shape[0]
        embed_dim = self.encoded_Phi.shape[2]

        # Convert local -> global coordinates
        gx = local_positions[:, 0] + C_X
        gy = local_positions[:, 1] + C_Y

        # Clamp to valid range (need neighbors at +1 for E and N)
        gx = np.clip(gx, 1, Npos - 2)
        gy = np.clip(gy, 1, Npos - 2)

        # Look up current encoded state
        current = self.encoded_Phi[gx, gy]  # (B, embed_dim)

        # Initialize or copy W
        if cached_W is None:
            W = np.zeros((B, 2, embed_dim), dtype=np.float32)
            recompute_mask = np.ones(B, dtype=bool)
        else:
            W = cached_W.copy()

        if recompute_mask is None:
            recompute_mask = np.ones(B, dtype=bool)

        # Recompute projection matrices where needed
        rc_idx = np.where(recompute_mask)[0]
        if len(rc_idx) > 0:
            rc_gx = gx[rc_idx]
            rc_gy = gy[rc_idx]
            rc_current = current[rc_idx]

            # North neighbor: (gx, gy+1) — forward
            north = self.encoded_Phi[rc_gx, rc_gy + 1]
            d_forward = north - rc_current  # (rc, embed_dim)

            # East neighbor: (gx+1, gy) — right
            east = self.encoded_Phi[rc_gx + 1, rc_gy]
            d_right = east - rc_current  # (rc, embed_dim)

            W_new = gram_schmidt_2d_batch(d_forward, d_right)  # (rc, 2, embed_dim)
            W[rc_idx] = W_new

        # Hopfield recall
        recalled = self.hopfield_recall_batch(current)  # (B, embed_dim)

        # Project displacement
        displacement = recalled - current  # (B, embed_dim)
        q = np.einsum('bij,bj->bi', W, displacement)  # (B, 2)

        # Classify direction
        direction_idx = classify_direction_batch(q)  # (B,)

        # Build one-hot
        onehot = np.zeros((B, 4), dtype=np.float32)
        onehot[np.arange(B), direction_idx] = 1.0

        proj = q if return_proj else None
        return onehot, proj, W

    def _recall_batched(self, S):
        """Batched recall: S is (Ns, N) — each column is a sensory vector.

        Returns (s_out, p_out, g_out) each of shape (dim, N).
        """
        # S: (Ns, N) — columns are observations
        pin = nonlin(self.Wps @ S, thresh=self.thresh)  # (Np, N)
        gin = self.Wgp @ pin  # (Ng, N)

        # Module-wise argmax (vectorized)
        ls = [l**2 for l in self.lambdas]
        gout = np.zeros_like(gin)
        idx = 0
        for j in ls:
            gmod = gin[idx:idx+j]  # (j, N)
            maxes = gmod.argmax(axis=0)  # (N,)
            gout[maxes + idx, np.arange(gin.shape[1])] = 1
            idx += j

        pout = nonlin(self.Wpg @ gout, thresh=self.thresh)  # (Np, N)
        sout = (self.Wsp @ pout > 0).astype(float)  # (Ns, N)
        return sout, pout, gout

    def test_vectorhash(self):
        """Test that recall() correctly recovers grid state from sensory input.

        For every explored position, verifies:
            sensory → recall() → grid_recovered == grid_true

        Raises:
            RuntimeError: If any position fails to recover the correct grid state.
        """
        n_samples = self.path_sbook.shape[1]

        print(f"   testing vectorhash recall on {n_samples} samples...")
        print(f"   shapes: path_sbook={self.path_sbook.shape}, path_pbook={self.path_pbook.shape}, path_gbook={self.path_gbook.shape}")
        print(f"   Wps={self.Wps.shape}, Wgp={self.Wgp.shape}, thresh={self.thresh}")

        # Uniqueness check (vectorized with hashing)
        s_bytes = set(self.path_sbook[:, i].tobytes() for i in range(n_samples))
        p_bytes = set(self.path_pbook[:, i].tobytes() for i in range(n_samples))
        print(f"   unique s: {len(s_bytes)}/{n_samples}, unique p: {len(p_bytes)}/{n_samples}")

        # --- Batched s→p check ---
        S = self.path_sbook  # (Ns, N)
        P_true = self.path_pbook  # (Np, N)
        G_true = self.path_gbook  # (Ng, N)

        P_raw = self.Wps @ S  # (Np, N)
        Pin = nonlin(P_raw, thresh=self.thresh)

        n_pin_correct = int(np.all(Pin == P_true, axis=0).sum())
        n_pin_argmax = int((P_raw.argmax(axis=0) == P_true.argmax(axis=0)).sum())

        # Print first 3 samples
        for i in range(min(3, n_samples)):
            p_raw_i = P_raw[:, i]
            p_true_i = P_true[:, i]
            print(f"   sample {i}: p_raw max={p_raw_i.max():.3f} min={p_raw_i.min():.3f} argmax={np.argmax(p_raw_i)} | p_true argmax={np.argmax(p_true_i)} nnz={np.sum(p_true_i>0)}")

        # --- Batched full recall ---
        s_out, p_out, g_recovered = self._recall_batched(S)

        # Compare
        p_match = np.all(p_out == P_true, axis=0)  # (N,)
        g_match = np.all(g_recovered == G_true, axis=0)  # (N,)
        n_p_correct = int(p_match.sum())
        n_correct = int(g_match.sum())
        failures = np.where(~g_match)[0].tolist()

        pin_accuracy = n_pin_correct / n_samples
        pin_argmax_acc = n_pin_argmax / n_samples
        p_accuracy = n_p_correct / n_samples
        accuracy = n_correct / n_samples
        print(f"   s→p exact: {n_pin_correct}/{n_samples} ({pin_accuracy*100:.1f}%)")
        print(f"   s→p argmax: {n_pin_argmax}/{n_samples} ({pin_argmax_acc*100:.1f}%)")
        print(f"   s→p→g→p roundtrip: {n_p_correct}/{n_samples} ({p_accuracy*100:.1f}%)")
        print(f"   grid recovery: {n_correct}/{n_samples} ({accuracy*100:.1f}%)")

        # --- Batched p→g and g→p direct tests ---
        Gin_direct = self.Wgp @ P_true  # (Ng, N)
        Gout_direct = np.zeros_like(Gin_direct)
        ls = [l**2 for l in self.lambdas]
        idx = 0
        for j in ls:
            gmod = Gin_direct[idx:idx+j]
            maxes = gmod.argmax(axis=0)
            Gout_direct[maxes + idx, np.arange(n_samples)] = 1
            idx += j
        n_pg_correct = int(np.all(Gout_direct == G_true, axis=0).sum())

        P_from_g = nonlin(self.Wpg @ G_true, thresh=self.thresh)
        n_gp_correct = int(np.all(np.isclose(P_from_g, P_true), axis=0).sum())

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
