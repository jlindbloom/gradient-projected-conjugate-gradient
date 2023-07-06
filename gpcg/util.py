import numpy as np
from scipy.sparse.linalg import LinearOperator


def determine_bound_and_free(x, l, u, tol=1e-8):
    """Given input vector $x$ and bounding vectors $l$ and $u$, returns indices where the constraint is active.
    Returns indices as (bound, free)."""
    
    check = np.logical_or(  np.abs(x - l) < tol, np.abs( x - u) < tol  )

    return np.argwhere(check)[:,0], np.argwhere( np.logical_not(check) )[:,0]



def build_Zk_subsampling_op( n, free ):
    """
    Builds the operator Zk that subsamples to the free variables.
    """
    
    n_free = len(free)
    
    def _rmatvec(x):
        return x[free]
    
    def _matvec(x):
        output = np.zeros(n)
        output[free] = x
        return output
    
    linop = LinearOperator( (n, n_free), _matvec, _rmatvec )
    
    return linop



def project_onto_feasible(x, l, u):
    """Given $x$, $l$, and $u$, computes projection onto feasible set using $P(x) = mid(l, u, x)$.
    """

    return np.median(np.vstack([l, x, u]).T, axis=1)

    

def determine_binding_set(A, b, x, l, u, tol=1e-8):
    """Given A, b, x, l, and u, computes the (indices of) the binding set B(x). """

    # Compute grad
    qgrad = (A @ x) - b

    # Check
    check = np.logical_or(   
                    np.logical_and( np.abs(x - l) < tol, qgrad >= 0.0 ), 
                    np.logical_and( np.abs( x - u) < tol, qgrad <= 0.0 )  
                )
    
    return np.argwhere( check )



def compute_mu(x, d, l, u, free_vars):
    """Computes mu, the maximum alpha s.t. x + alpha*d is still inside the constraint set."""

    # Subsample the variables
    x = x[free_vars]
    d = d[free_vars]
    l = l[free_vars]
    u = u[free_vars]
    n = len(x)

    # Compute alphas
    upper_end = (u - x)/d
    lower_end = (l - x)/d

    alphas = []
    for j in range(n):
        if d[j] >= 0:
            alphas.append(upper_end[j])
        else:
            alphas.append(lower_end[j])

    # Take the minimum
    alpha = np.amin(alphas)

    return alpha


