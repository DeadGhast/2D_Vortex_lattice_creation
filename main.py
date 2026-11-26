import numpy as np
import torch
import matplotlib.pyplot as plt
from rk4 import solver_SFSM, wave_vectors_for_SFSM_method, read_file
from parameters import *
import os.path
from os import getcwd
from tqdm import tqdm



def eq_with_SFSM_parameters(r, g, alpha, sigma):
    "Diffrentiential equation for creating lattice of quantum vortices"
    def vortex_equation(psi):
        "Kinetic part"
        kinetic_part = (-1)

        "Remaining part"
        modulus_squared = torch.conj(psi) * psi
        potential = r
        interaction = g * modulus_squared
        pumping_and_dissipation = (1j) * ((alpha) - (sigma * modulus_squared))

        non_kinetic_part = potential + interaction + pumping_and_dissipation
        return kinetic_part, non_kinetic_part
    return vortex_equation

if __name__ == '__main__':

    "Selecting 'cuda' accelerator - if unavailable, falls back to 'cpu'"
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {my_device}")
    if my_device.type == "cuda":
        print(torch.cuda.get_device_name(0))
    #=================================================================================#
    "Creating spatial grid"
    dx = Lx / Nx
    print(f'dx = {dx}')
    x = torch.linspace(-Lx / 2, Lx / 2, Nx, device=my_device)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    r = torch.sqrt(X ** 2 + Y ** 2)
    T = np.arange(0, tmax, dt)
    #=================================================================================#
    "Defining alpha using a step function"
    step_function = torch.ones_like(X, device=my_device)
    step_function[r > R_p] = 0
    alpha = alpha_0 * step_function
    #=================================================================================#
    "Defininf initial value of wave function"
    SEED = None# 64713 
    if isinstance(SEED, int):
        rng = np.random.default_rng(SEED)
    else:
        rng = np.random.default_rng()
    mu, sig = 0, 0.05  # mean and standard deviation
    psi_noise = (rng.normal(mu, sig, size=(Nx, Nx)))

    psi_noise = torch.asarray(psi_noise)
    psi_noise = psi_noise.to(my_device)
    psi = psi_noise
    #=================================================================================#
    period = 10_000  # How often we get a snapshot of the simulation
    compute_SFSM = True
    #=================================================================================#
    "Creating path for saving results"
    path = getcwd()
    file_name = 'results_R_' + str(R_p) + "_a_" + str(alpha_0) + "_SFSM"
    file_name = file_name.replace(".", "_", 10)
    file_name = file_name + ".npy"
    file_path = os.path.join(path, file_name)
    #=================================================================================#
    def computing_SFSM(psi, parameters, equation, time, filepath,):
        """
        Function solves diffrential equation to create a lattice of quantum vortices

        We create a loop iterating up to a specific time 'tmax'
        - Every 'period' steps, function save the wavefunction distribution to a file.
        At the end, save the list of times at which the wavefunction distribution was saved to the file

        We will obtain a file with 'tmax'/'dt'/'period' + 1 entries (tmax/dt/200 - wavefunctions, 1 - times)
        """
        times = [0]
        new_psi = psi
        f = open(filepath, 'wb')
        wavefunction = psi.to('cpu')
        print(f'Saved for 0')
        np.save(f, wavefunction)
        # -------------------------------------------------------------#
        n = 1
        for t in tqdm(time[1:]):
            new_psi = solver_SFSM(Parameters=parameters, initial_condition=new_psi, special_function=equation)

            if n % period == 0:
                wavefunction = new_psi.to('cpu').numpy()
                np.save(f, wavefunction)
                times.append(t)
            n += 1

        # Saving the last iteration
        wavefunction = new_psi.to('cpu').numpy()
        np.save(f, wavefunction)
        times.append(t)
        #=================#
        # Saving the list of times at which the wavefunction distribution was saved to the file
        np.save(f, times)
        f.close()

    #=================================================================================#
    "Starting simulation"
    parameters = [Lx, Lx, Nx, Nx]
    k_x, k_y = wave_vectors_for_SFSM_method(parameters, device=my_device)
    new_parameters = [k_x, k_y, dt]

    "Creating equation"
    special_function = eq_with_SFSM_parameters(r=r**2, g=g, alpha=alpha, sigma=sigma)

    "Creating simulation"
    if compute_SFSM:
        computing_SFSM(psi=psi, parameters=new_parameters, equation=special_function, time=T, filepath=file_path)

    "Obtaining results from the simulation"
    data, time_steps = read_file(file_path=file_path)
    print(len(time_steps))
    print(tmax/dt/period + 1)
    
    fig, ax = plt.subplots()
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    X, Y = np.meshgrid(x, x)
    ax.pcolormesh(X, Y, np.abs(data[-1])**2)
    plt.show()
    