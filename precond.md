# Overview 

The aim of preconditioner is to accelerate GREMS convergence
GREMS algorithm struggles when eigenvalues are both pos/eng
This implies oscilltion and requires more complex polynomial solutions
The exact details of GREMS IDK 
What is known is that given HelmHoltz Operator
\left(-\nebla^2 - k^2(x) \right) u(x) = f(x)
shifting the k^2(x) into the complex plane (1 + i \beta) k^2(x) 
dampens the oscillatory nature of the solution, u(x), by e^{-k_{imag} x} 
 
 
# Questions: 
- How does this have the effect of aligning eigenvalue signs? 

# Challenges: 
- Implement laplacian_shift with PML absorption (no extra reflection) intact
- Implement multigrid v-cycle for efficient calculation of P^{-1} = \nebla^2 - (1 + i \beta) k^2(x), \quad where \beta = 0.1

# Solved 
- Passing preconditioner into JAX grems, grems already handles precond

Notes
- B must be chosen carefully arXiv:1507.02097

J-wave implementation of PML
- s_{j} = 1 + i \alpha_{j}, \quad where \alpha_{j} = \alpha_{max} \dot \left(\frac{dist_{j}}{\text{PML_thickness}})^{p}, \quad where p=?
