import numpy as np

import torch
import torch.fft as fft
from scipy.integrate import simpson


def integration_over_matrix(matrix, x):
    """
    Function performs numerical integration of 'matrix' over the domain (x by x) - Nx = len(x)

    Parameters:
    matrix: (ndarray [Nx x Nx]) Square matrix
    x: (ndarray, [1 x Nx]) Axis along which integration is performed

    Returns:
    value: (float) Integration result
    """
    value = simpson([simpson(strip, x) for strip in matrix], x)
    return value

def solve_Diff_2D(Parameters, initial_condition, function, solver_type):
    """
    Function calculates the time derivative 'dt' of 'initial_condition' using the 4th-order RK method

    Parameters:
    Parameters [Lx, Ly, Nx, Ny, D, tmax, Nt]: set of equation parameters
    initial_condition (ndarray (Nx, Ny)): Value of function U resulting from the diffusion equation at a given time point
    solver_type (string): "CPU" or "GPU"
    """
    if solver_type == "CPU":
        Lx, Ly, Nx, Ny, tmax, dt, X, Y = Parameters

        U = initial_condition

        # 1
        U_1 = function(U)

        v_1 = U + 0.5 * U_1 * dt
        # 2
        U_2 = function(v_1)
        v_1 = U + 0.5 * U_2 * dt
        # 3
        U_3 = function(v_1)
        v_1 = U + U_3 * dt
        # 4
        U_4 = function(v_1)

        U = U + (U_1 + 2 * U_2 + 2 * U_3 + U_4) * dt / 6
        # modulus_squared = np.real(np.conj(U) * U)
        return U

    if solver_type == "GPU":
        Lx, Ly, Nx, Ny, tmax, dt, X, Y = Parameters

        U = initial_condition

        # 1
        U_1 = function(U)

        v_1 = U + 0.5 * U_1 * dt
        # 2
        U_2 = function(v_1)
        v_1 = U + 0.5 * U_2 * dt
        # 3
        U_3 = function(v_1)
        v_1 = U + U_3 * dt
        # 4
        U_4 = function(v_1)

        U = U + (U_1 + 2 * U_2 + 2 * U_3 + U_4) * dt / 6
        # modulus_squared = np.real(np.conj(U) * U)
        return U


def wave_vectors_for_SFSM_method(Parameters, device='cpu'):
    Lx, Ly, Nx, Ny, tmax, dt, X, Y = Parameters

    kappa_x = (2 * torch.pi / Lx) * torch.arange(-Nx / 2, Nx / 2, device=device)
    kappa_y = (2 * torch.pi / Ly) * torch.arange(-Ny / 2, Ny / 2, device=device)
    kappa_x = fft.fftshift(kappa_x)
    kappa_y = fft.fftshift(kappa_y)

    K_x, K_y = torch.meshgrid(kappa_x**2, kappa_y**2, indexing="ij")

    return K_x, K_y


def solver_SFSM(Parameters, initial_condition, special_function, *, wave_vectors_included=True):
    "Function REQUIRES a precomputed meshgrid of wave vectors"
    if wave_vectors_included:
        X, Y, dt, K_x, K_y = Parameters
    else:
        Lx, Ly, Nx, Ny, tmax, dt, X, Y = Parameters
        K_x, K_y = wave_vectors_for_SFSM_method(
            Parameters, device=initial_condition.device
        )

    """
    We assume that special_function has the structure:
    special_function[0] - term multiplying the kinetic energy
    special_function[1] - remaining terms
    """

    # Step 1: Half-step evolution for potential, interaction, pumping, and dissipation
    kinetic_coeff, non_kinetic = special_function(initial_condition)
    time_evolution_1 = torch.exp((-1j) * (dt / 2) * non_kinetic) * initial_condition

    # Step 2: Full-step kinetic energy evolution in Fourier space
    fhat = fft.fft2(time_evolution_1)
    fhat_with_kinetic = torch.exp(kinetic_coeff * (1j) * dt * (K_x + K_y)) * fhat
    inverse_transform = fft.ifft2(fhat_with_kinetic)

    # Step 3: Second half-step non-kinetic evolution
    _, non_kinetic_2 = special_function(inverse_transform)
    time_evolution_2 = torch.exp((-1j) * (dt / 2) * non_kinetic_2) * inverse_transform

    return time_evolution_2


def read_file(file_path):
    """
    Function for reading data from an '.npy' file.
    Results are stored in the 'data' and 'time_steps' variables

    data       [list] - list of wavefunction values for specific times
    time_steps [list] - list of times corresponding to the wavefunctions
    :param file_path:
    :return:
    """
    data = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                data.append(np.load(f))
            except:
                break
    time_steps = data[-1]
    data = data[0:-1]
    return data, time_steps