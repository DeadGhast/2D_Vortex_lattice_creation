Lx = 25 * 2
Nx = 2**9 + 1
dx = Lx / Nx
# Nx = round(Lx/dx)
# print(Nx)
tmax = 50
dt = 0.001

"Parameters for the equation: alpha, sigma and R_p"
R_p = 4.5
"""
R_p = 2.75 -> 1 vortex
R_p = 4.5 -> 3 vortices
R_P = 5 -> creates from 5 to 7 vorticies
R_p = 6 -> 17 vortices
R_p = 7 -> 28 vortices
""" 
alpha_0 = 4.4
g = 1
sigma = 0.3
