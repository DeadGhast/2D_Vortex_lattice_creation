import numpy as np
import torch
import torch.fft as fft


def wave_vectors_for_SFSM_method(Parameters, device='cpu'):
    """
    Function calculates squared wave vectors needed for TSSP method
    """
    
    Lx, Ly, Nx, Ny = Parameters

    kappa_x = (2 * torch.pi / Lx) * torch.arange(-Nx / 2, Nx / 2, device=device)
    kappa_y = (2 * torch.pi / Ly) * torch.arange(-Ny / 2, Ny / 2, device=device)
    kappa_x = fft.fftshift(kappa_x)
    kappa_y = fft.fftshift(kappa_y)

    K_x, K_y = torch.meshgrid(kappa_x**2, kappa_y**2, indexing="ij")

    return K_x, K_y


def solver_SFSM(Parameters, initial_condition, special_function):
    """
    Function solves diffrential equation one time step forward usign TSP
    """

    
    "Function REQUIRES a precomputed meshgrid of wave vectors"
    K_x, K_y, dt = Parameters
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
    Function ile.
    Results are stored in the 'data' and 'time_steps' varfor reading data from an '.npy' fiables

    data       [list] - list of wavefunction values for specific times
    time_steps [list] - list of times at which the wavefunction distribution was saved to the file
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