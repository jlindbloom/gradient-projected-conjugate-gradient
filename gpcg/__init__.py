# Check if CuPy installed or not
CUPY_INSTALLED = False
try:
    import cupy
    CUPY_INSTALLED = True
except:
    pass

# Imports
from .util import determine_bound_and_free, build_Zk_subsampling_op, project_onto_feasible, compute_mu, determine_binding_set
from .examples import make_1d_signal, make_shepp_logan_image, build_1d_first_order_grad, build_2d_first_order_grad
from .gpcg import GPCGSolver





