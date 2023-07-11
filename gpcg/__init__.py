# Check if CuPy installed or not
CUPY_INSTALLED = False
try:
    import cupy
    CUPY_INSTALLED = True
except:
    pass

# Imports
from gpcg.util import determine_bound_and_free, build_Zk_subsampling_op, project_onto_feasible, compute_mu, determine_binding_set
from gpcg.gpcg import GPCGSolver





