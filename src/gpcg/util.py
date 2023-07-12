import numpy as np
from scipy.sparse.linalg import LinearOperator

from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator as CuPyLinearOperator



def determine_bound_and_free(x, l, u, tol=1e-8):
    """Given input vector $x$ and bounding vectors $l$ and $u$, returns indices where the constraint is active.
    Returns indices as (bound, free)."""
    
    if CUPY_INSTALLED:
        xp = cp.get_array_module(x)
    else:
        xp = np
    
    check = xp.logical_or(  xp.abs(x - l) < tol, xp.abs( x - u) < tol  )

    return xp.argwhere(check)[:,0], xp.argwhere( xp.logical_not(check) )[:,0]



def build_Zk_subsampling_op(n, free, xp):
    """
    Builds the operator Zk that subsamples to the free variables.
    """
    
    n_free = len(free)
    
    def _rmatvec(x):
        return x[free]
    
    def _matvec(x):
        output = xp.zeros(n)
        output[free] = x
        return output
    
    if xp == np:
        linop = LinearOperator( (n, n_free), _matvec, _rmatvec )
    else:
        linop = CuPyLinearOperator( (n, n_free), _matvec, _rmatvec )
    
    return linop



def project_onto_feasible(x, l, u):
    """Given $x$, $l$, and $u$, computes projection onto feasible set using $P(x) = mid(l, u, x)$.
    """
    if CUPY_INSTALLED:
        xp = cp.get_array_module(x)
    else:
        xp = np

    return xp.median(np.vstack([l, x, u]).T, axis=1)

    

def determine_binding_set(A, b, x, l, u, tol=1e-8):
    """Given A, b, x, l, and u, computes the (indices of) the binding set B(x). """
    
    if CUPY_INSTALLED:
        xp = cp.get_array_module(x)
    else:
        xp = np
    
    # Compute grad
    qgrad = (A @ x) - b

    # Check
    check = xp.logical_or(   
                    xp.logical_and( xp.abs(x - l) < tol, qgrad >= 0.0 ), 
                    xp.logical_and( xp.abs( x - u) < tol, qgrad <= 0.0 )  
                )
    
    return xp.argwhere( check )



def compute_mu(x, d, l, u, free_vars):
    """Computes mu, the maximum alpha s.t. x + alpha*d is still inside the constraint set."""

    if CUPY_INSTALLED:
        xp = cp.get_array_module(x)
    else:
        xp = np
    
    # Subsample the variables
    x = x[free_vars]
    d = d[free_vars]
    l = l[free_vars]
    u = u[free_vars]
    n = len(x)

    # Compute alphas
    upper_end = (u - x)/d
    lower_end = (l - x)/d

    # New, much FASTER method
    condlist = [d >= 0, d < 0]
    choicelist = [upper_end, lower_end]
    alphas = xp.select(condlist, choicelist, 0)
    alpha = xp.amin(alphas)

    return alpha    



def cupy_mod_select(condlist, choicelist, default=0):
    """
    This is a modified version of the function cupy.select, which is a quick fix to a bug I ran into
    with bools not having a .dtype attribute.
    
    See https://docs.cupy.dev/en/stable/reference/generated/cupy.select.html.
    """

    if len(condlist) != len(choicelist):
        raise ValueError(
            'list of cases must be same length as list of conditions')

    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")

    if not cp.isscalar(default):
        raise TypeError("default only accepts scalar values")

    for i in range(len(choicelist)):
        if not isinstance(choicelist[i], cp.ndarray):
            raise TypeError("choicelist only accepts lists of cupy ndarrays")
        cond = condlist[i]

    dtype = cp.result_type(*choicelist)

    condlist = cp.broadcast_arrays(*condlist)
    choicelist = cp.broadcast_arrays(*choicelist, default)

    if choicelist[0].ndim == 0:
        result_shape = condlist[0].shape
    else:
        result_shape = cp.broadcast_arrays(condlist[0],
                                             choicelist[0])[0].shape

    result = cp.empty(result_shape, dtype)
    cp.copyto(result, default)

    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        cp.copyto(result, choice, where=cond)

    return result


