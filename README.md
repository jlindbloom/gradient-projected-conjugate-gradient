# gradient-projected-conjugate-gradient

This repository provides a Python implementation of the gradient projected conjugate gradient algorithm (GPCG) presented in [[1]](#1) for solving bound-constrained quadratic programs of the form
```math
\text{argmin}_{ x_i \in [l_i, u_i] \text{ for } i = 1, \ldots, n } \,\, \frac{1}{2} x^T A x - b^T x
```
where $b \in \mathbb{R}^n$ and $A \in \mathbb{R}^{n \times n}$ is a SPD matrix. Here the $l_i$ and/or $u_i$ may be infinite, e.g., we can solve quadratic programs with nonnegativity constraints.

Install with ``pip install gpcg``.

## References
<a id="1">[1]</a> 
Moré, J., & Toraldo, G. (1991). On the Solution of Large Quadratic Programming Problems with Bound Constraints. SIAM Journal on Optimization, 1(1), 93-113.



