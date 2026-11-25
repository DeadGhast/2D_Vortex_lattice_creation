# 2D Quantum Vortex Simulation

A PyTorch-based numerical solver for simulating dynamics of lattice of quantum vortices.

## Governing Equation

Equation used for simulating vortices[1]:

$$i \frac{\partial \psi}{\partial t} = \left[ -\nabla^2 + r^2 + g|\psi|^2 + i\left(\alpha(r) - \sigma|\psi|^2\right) \right]\psi$$

where:
- $g$ is the nonlinear interaction strength,
- $\alpha(r) = \alpha \Theta(R_p - r)$ is the effective pumping rate
- $R_p$ is the cutoff radius
- $\sigma$ is the effective rate of saturation loss.

## References
[1] J. Keeling and N. G. Berloff, “Spontaneous rotating vortex lattices in a pumped decaying condensate,” Phys. Rev. Lett. 100, 250401 (2008):
