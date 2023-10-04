import numpy as np
import itertools
import quimb.tensor as qtn
from quimb.tensor import TensorNetwork1D, MatrixProductState
from quimb.tensor.tensor_1d import TensorNetwork1DVector
from quimb.utils import progbar, oset
from quimb.tensor.tensor_dmrg import get_default_opts, MovingEnvironment
from quimb.experimental.tensor_1d_mpo_gate_methods import mps_gate_with_mpo_lazy

class ALS_fit_1d:
    def __init__(self, tn_fit, init_guess, max_bond, bsz=1):

        if not isinstance(tn_fit, TensorNetwork1DVector):
            ValueError("tn_fit should be TensorNetwork1DVector")
        if not isinstance(init_guess, MatrixProductState):
            ValueError("init_guess should be MatrixProductState")

        self.L = tn_fit.L
        self.phys_dim = tn_fit.phys_dim()
        self.bsz = bsz
        self.max_bond = max_bond

        self.tn_fit = tn_fit.copy()
        self.tn_fit.add_tag("_TARGET")

        if init_guess is not None:
            self._b = init_guess.conj()
            self._b.site_ind_id = tn_fit.site_ind_id
            self._b.add_tag("_BRA")
        else:
            self._b = qtn.MPS_rand_state(L=self.L, bond_dim=max_bond, phys_dim=self.phys_dim, site_ind_id=tn_fit.site_ind_id)
            self._b.add_tag("_BRA")

        self.TN_ovlp = (self._b | self.tn_fit)

        self.opts = get_default_opts()

    @property
    def state(self):
        copy = self._b.H
        copy.drop_tags('_BRA')
        return copy
    
    # -------------------- standard DMRG update methods --------------------- #

    def _canonize_after_1site_update(self, direction, i):
        """Compress a site having updated it. Also serves to move the
        orthogonality center along.
        """
        if (direction == 'right') and (i < self.L - 1):
            self._b.left_canonize_site(i)
        elif (direction == 'left') and (i > 0):
            self._b.right_canonize_site(i)

    def _update_loc_state_1site(self,i, direction):

        self.ME_eff_ovlp.move_to(i)
        ME_ovlp = self.ME_eff_ovlp()
        loc_t = (ME_ovlp ^ '_TARGET')['_TARGET']
        loc_t.transpose_(*self._b[i].inds)
        self._b[i].modify(data=loc_t.data.conj())

        self._canonize_after_1site_update(direction, i)

    def sweep(self, direction, canonize=True, verbosity=1):

        if canonize:
            {'R': self._b.right_canonize,
             'L': self._b.left_canonize}[direction]()
            
        n, bsz = self.L, self.bsz

        direction, begin, sweep = {
            'R': ('right', 'left', range(0, n - bsz + 1)),
            'L': ('left', 'right', range(n - bsz, -1, -1)),
        }[direction]

        if verbosity:
            sweep = progbar(sweep, ncols=80, total=len(sweep))

        self.ME_eff_ovlp = MovingEnvironment(self.TN_ovlp, begin, bsz)

        for i in sweep :
            self._update_loc_state_1site(i, direction)
        
        if verbosity:
            sweep.close()

        self.ME_eff_ovlp = None

    def solve(self, max_bond=None, sweep_sequence=None, max_sweeps=10, tol=1e-4, verbosity=1):
        if max_bond is not None:
            self.max_bond = max_bond
        if sweep_sequence is None:
            sweep_sequence = self.opts['default_sweep_sequence']

        RLs = itertools.cycle(sweep_sequence)
        previous_LR = '0'

        self._b.expand_bond_dimension(self.max_bond, rand_strength=self.opts['bond_expand_rand_strength'])
        self._b.normalize()

        for sw in range(max_sweeps):
            if verbosity :
                print("SWEEP-"+str(sw))
            LR = next(RLs)
            bprev = self._b.copy()

            canonize = False if LR + previous_LR in {'LR', 'RL'} else True

            self.sweep(direction=LR, canonize=canonize, verbosity=verbosity)
        
            self._b.normalize()

            ov = bprev.H @ self._b
            if abs(ov) > 1.0 - tol :
                break
        return sw+1
    
def mps_gate_with_mpo_als1dfit(
    mps,
    mpo,
    max_bond,
    cutoff=0.0,
    init_guess=None,
    verbosity = 0      
):
    if cutoff != 0.0:
        raise ValueError("cutoff must be zero for fitting")
    
    target = mps_gate_with_mpo_lazy(mps, mpo)

    if init_guess is None:
        init_guess = mps.copy()
        init_guess.expand_bond_dimension_(max_bond)
    
    alsfit = ALS_fit_1d(target, init_guess, max_bond)
    alsfit.solve(verbosity=verbosity)

    return alsfit.state
