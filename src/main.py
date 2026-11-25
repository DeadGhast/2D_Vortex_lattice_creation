import numpy as np
import torch
import matplotlib.pyplot as plt
from rk4 import solver_SFSM, wave_vectors_for_SFSM_method, read_file
from parameters import *
import time
import os.path
from os import getcwd
from tqdm import tqdm



def eq_with_SFSM_parameters(r, g, alpha, sigma, R_p, dx):
    "Equation with simulation parameters used ONLY for the SFSM (TSPS) method"
    def vortex_equation(psi):
        "Kinetic part"
        kinetic_part = (-1)

        "Remaining operators"
        modulus_squared = torch.conj(psi) * psi
        potential = r
        interaction = g * modulus_squared
        pumping_and_dissipation = (1j) * ((alpha) - (sigma * modulus_squared))

        non_kinetic_part = potential + interaction + pumping_and_dissipation
        return kinetic_part, non_kinetic_part
    return vortex_equation


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t_start = time.time()
    "Selecting 'cuda' accelerator - if unavailable, falls back to 'cpu'"
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {my_device}")
    if my_device.type == "cuda":
        print(torch.cuda.get_device_name(0))
    # -------------------------------------------------------------#
    "Creating spatial grid"
    dx = Lx / Nx
    print(f'dx = {dx}')
    x = torch.linspace(-Lx / 2, Lx / 2, Nx, device=my_device)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    r = torch.sqrt(X ** 2 + Y ** 2)
    T = np.arange(0, tmax, dt)
    # -------------------------------------------------------------#
    "Defining alpha using a step function"
    step_function = torch.ones_like(X, device=my_device)
    step_function[r > R_p] = 0
    alpha = alpha_0 * step_function
    # -------------------------------------------------------------#
    mu, sig = 0, 0.05  # mean and standard deviation
    psi_noise = (np.random.normal(mu, sig, size=(Nx, Nx)))

    psi_noise = torch.asarray(psi_noise)
    psi_noise = psi_noise.to(my_device)
    psi = psi_noise
    # -------------------------------------------------------------#
    period = 10_000  # How often we get a snapshot of the simulation
    compute_SFSM = True
    # -------------------------------------------------------------#
    path = getcwd()
    file_name = 'results_R_' + str(R_p) + "_a_" + str(alpha_0) + "_SFSM"
    file_name = file_name.replace(".", "_", 10)
    file_name = file_name + ".npy"
    file_path = os.path.join(path, file_name)

    if compute_SFSM:
        """
        We create a loop iterating up to a specific time 'tmax'
        Options:
        a) Every 200 steps, save the wavefunction distribution to a file.
        b) Save for specific times - stored in the 'save_times' variable
        At the end, save the list of times for which the wavefunction distribution was saved to the file

        We will obtain a file with tmax/dt/200 + 1 entries (tmax/dt/200 - wavefunctions, 1 - times)
        """
        times = [0]
        parameters = [Lx, Lx, Nx, Nx, tmax, dt, X, Y]
        k_x, k_y = wave_vectors_for_SFSM_method(parameters, device=my_device)
        new_parameters = [X, Y, dt, k_x, k_y]

        special_function = eq_with_SFSM_parameters(r=r**2, g=g, alpha=alpha, sigma=sigma, R_p=R_p, dx=dx)

        f = open(file_path, 'wb')
        wavefunction = psi.to('cpu')
        print(f'Saved for 0')
        np.save(f, wavefunction)
        # -------------------------------------------------------------#
        n = 1
        t1 = time.time()
        for t in tqdm(T[1:]):
            psi = solver_SFSM(Parameters=new_parameters, initial_condition=psi, special_function=special_function)

            if n % period == 0:
                # print(f'Done for {n*dt}')
                wavefunction = psi.to('cpu').numpy()
                np.save(f, wavefunction)
                times.append(t)
            n += 1

        np.save(f, times)
        f.close()

        t2 = time.time()
        print(f"Program execution time is {t2 - t1} s = {(t2 - t1) / 60} min = {(t2 - t1) / 3600} h")
        print(f'Execution time of a single step is {(t2 - t1) / (round(tmax / dt))} s')
        t_end = time.time()
        print(f"Total program runtime {t_end - t_start}")

    data, time_steps = read_file(file_path=file_path)
    fig, ax = plt.subplots()
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    X, Y = np.meshgrid(x, x)
    ax.pcolormesh(X, Y, np.abs(data[-1])**2)
    plt.show()