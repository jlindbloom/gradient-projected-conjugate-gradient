import numpy as np

from .util import determine_bound_and_free, determine_bound_and_free, build_Zk_subsampling_op, project_onto_feasible, compute_mu, determine_binding_set



class GPCGSolver:
    """Class for the GPCG solver."""

    def __init__(self, A, b, lower_bounds=None, upper_bounds=None):
        
        # Bind
        self.A = A
        self.b = b
        self.n = len(b)

        # Handle bound constraints
        if lower_bounds is None: lower_bounds = -np.inf*np.ones(n)
        self.lower_bounds = lower_bounds

        if upper_bounds is None: upper_bounds = np.inf*np.ones(n)
        self.upper_bounds = upper_bounds
    
        # Other setup
        self.binf = np.linalg.norm(self.b, ord=np.inf)
        self.qfunc = lambda x: 0.5*(x.T @ (self.A @ x) ) - (self.b.T @ x)
        self.grad_qfunc = lambda x: (self.A @ x) - b
    

    def solve(self, maxits=100,  tol=1e-4, eta1=0.1, eta2=0.25, mu=0.1, x0=None, bound_tol=1e-14):
        """Computes solution using the GPCG algorithm."""

        
        # Initialization
        if x0 is None:
            x = np.ones(self.n)
        else:
            x = x0

        # Convergence flag
        converged = False

        # Iterate
        for j in range(maxits):

            ### Step 1: Gradient projection iterates
            x = self.gradient_projection_substep(y0=x, mu=mu, eta2=eta2)

            ### Step 2: Conjugate gradient iterates
            x = self.conjugate_gradient_substep(x, mu=mu, bound_tol=bound_tol, eta1=eta1)
            
            ### Check stopping criterion
            proj_grad = self.eval_projected_gradient(x, tol=bound_tol)
            if np.linalg.norm(proj_grad) < tol*np.linalg.norm(self.grad_qfunc(np.zeros_like(x))):
                converged = True
                break

        data = {
            "x": x,
            "converged": converged,
        }
                
        return data
    

    def conjugate_gradient_substep(self, x, bound_tol=1e-8, mu=0.1, eta1=0.1):
        """Carries out the conjugate gradient substep."""
        
        # Compute active set
        bound, free = determine_bound_and_free(x, self.lower_bounds, self.upper_bounds, tol=bound_tol)
        Z = build_Zk_subsampling_op(self.n, free)
        
        # Compute Ak and rk (for reduced problem)
        w = Z.rmatvec(x)
        Amod = Z.T @ self.A @ Z # Ak from paper
        bmod = Z.rmatvec(self.A.matvec(x)) - Z.rmatvec(self.b) # rk from paper
    
        # Compute binding set
        binding_set = determine_binding_set(self.A, self.b, x, self.lower_bounds, self.upper_bounds, tol=bound_tol)

        # Repeat conjugate gradient method until binding set no longer equals active set
        while np.array_equal( np.sort(binding_set), np.sort(free) ):

            # Solve subproblem for the next descent direction
            descent_direction_mod = self.conjugate_gradient_subproblem_solve(Amod, -bmod, x0=np.zeros(len(bmod)), eta1=eta1)
            descent_direction = Z.matvec(descent_direction_mod)

            ## Determine stepsize alpha using line search
            alpha = 1.0
            beta1 = compute_mu(x, descent_direction, self.lower_bounds, self.upper_bounds, free)
            phi_zero = self.qfunc( project_onto_feasible(x, self.lower_bounds, self.upper_bounds) ) # \phi( P(x) )
            phi_prime_zero = ( descent_direction.T @ (self.A @ x) ) - (self.b.T @ descent_direction ) # d/dalpha \phi(P(x + alpha*d)) at alpha = 0
            proj_step = project_onto_feasible( x + alpha*descent_direction , self.lower_bounds, self.upper_bounds) # P( x + \alpha d )
            phi_alpha = self.qfunc( proj_step ) # \phi( P(x + \alpha d)  )
        
            # Compute new alpha until sufficient decrease condition satisfied
            while phi_alpha > phi_zero + mu*( ( self.grad_qfunc(x)  ).T @ ( proj_step - x )  ):

                # Compute coefficients of interpolating polynomial, and minimizer
                c = phi_zero
                b = phi_prime_zero
                a = ( phi_alpha - phi_prime_zero*alpha - phi_zero ) / (alpha**2)
                alpha_star = - b / (2*a)

                # Compute new alpha
                alpha = np.amax([ np.median([ 0.01*alpha, alpha_star, 0.5*alpha ]), beta1 ])

                # Calculations for checking sufficient decrease condition
                proj_step = project_onto_feasible( x + alpha*descent_direction, self.lower_bounds, self.upper_bounds)
                phi_alpha = self.qfunc(proj_step)
            

            # Take step
            x = x + alpha*descent_direction

        return x


    def conjugate_gradient_subproblem_solve(self, A, b, search_direction_maxits=50, x0=None, eta1=1e-8):
        """Solves the subproblem in the conjugate gradient substep."""

        # Initialization
        if x0 is None:
            x = np.ones(n)
        else:
            x = x0

        # Keep track of qs
        qs = np.asarray([])
        
        # Initial
        r = b - A.matvec(x)
        d = r.copy()
        qs = np.append(qs, 0.5*( x.T @ (A @ x) ) - (b.T @ x)    )

        # Iterate conjugate gradient
        satisfied_criterion = False
        for j in range(search_direction_maxits):

            # Compute next step
            alpha = (r.T @ r)/(d.T @ A.matvec(d) )
            x = x + alpha*d
            rnew = r - alpha * A.matvec( d )
            beta = (rnew.T @ rnew)/(r.T @ r)
            d = rnew + beta*d
            r = rnew

            # Evaluate q
            qs = np.append(qs, 0.5*( x.T @ (A @ x) ) - (b.T @ x)    )

            # Check termination criterion
            try:
                qdiff = qs[-2] - qs[-1]
                other_qdiffs = qs[:-2] - qs[1:-1]
                if qdiff < eta1*np.amax(other_qdiffs):
                    satisfied_criterion = True
                    break
            except:
                pass

        return x


    def gradient_projection_substep(self, y0=None, maxits=100, bound_tol=1e-8, eta2=1e-8, mu=0.1):
        """Performs the gradient projection iterates. Here $l$ and $u$ are the vectors of lower and upper bounds."""

        if y0 is None:
            y = np.ones(self.n)
        else:
            y = y0

        # Setup
        _, prev_free_vars = determine_bound_and_free(y, self.lower_bounds, self.upper_bounds, tol=bound_tol)
        qs = np.asarray([])

        # Do iteration
        for j in range(maxits):

            ### Compute direction (NOTE: \nabla q(y) = A y - b)
            current_residual = self.A.matvec(y) - self.b
            direction = -current_residual

            ### Compute stepsize alpha

            # Initial calculations for stepsize
            alpha =  (current_residual.T @ current_residual) / ( current_residual.T @ (self.A @ current_residual) ) # initial alpha
            phi_zero = self.qfunc( project_onto_feasible(y, self.lower_bounds, self.upper_bounds) ) # \phi( P(y) )
            proj_step = project_onto_feasible(y + alpha*direction, self.lower_bounds, self.upper_bounds)
            phi_alpha = self.qfunc( proj_step  )

            # Compute new alpha until sufficient decrease condition satisfied
            while phi_alpha > phi_zero + mu*( ( self.grad_qfunc(y)  ).T @ ( proj_step - y )  ):

                # Just do simple halving search
                alpha = 0.5*alpha

                # Calculations for checking sufficient decrease condition
                proj_step = project_onto_feasible( y + alpha*direction , self.lower_bounds, self.upper_bounds)
                phi_alpha = self.qfunc(proj_step)
            
            
            ### Take step and project onto feasible
            y_uncon = y + alpha*direction
            y = project_onto_feasible(y_uncon, self.lower_bounds, self.upper_bounds)
            
            ### Check convergence criterion
            
            # Check if free variables are the same
            _, new_free_vars  = determine_bound_and_free(y, self.lower_bounds, self.upper_bounds, tol=bound_tol)
            if np.array_equal(new_free_vars, prev_free_vars): 
                break
            else:
                prev_free_vars = new_free_vars

            # Check qdiffs
            qs = np.append(qs, 0.5*( y.T @ (self.A @ y) ) - (self.b.T @ y)    )
            
            try:
                qdiff = qs[-2] - qs[-1]
                other_qdiffs = qs[:-2] - qs[1:-1]
                if qdiff < eta2*np.amax(other_qdiffs):
                    break
            except:
                pass

        return y


    def eval_projected_gradient(self, x, tol=1e-8):
        """Evaluates the projected gradient."""
        grad = self.grad_qfunc(x)
        proj_grad = []
        for j in range(self.n):
            if np.abs(x[j] - self.lower_bounds[j]) < tol:
                proj_grad.append( np.amin([grad[j], 0.0]) )
            elif np.abs(x[j] - self.upper_bounds[j]) < tol:
                proj_grad.append( np.amax([grad[j], 0.0]) )
            else:
                proj_grad.append(grad[j])

        return np.asarray(proj_grad)



