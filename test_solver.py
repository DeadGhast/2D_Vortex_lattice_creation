import pytest
import numpy as np
import torch
from rk4 import solver_SFSM, wave_vectors_for_SFSM_method
from main import eq_with_SFSM_parameters

@pytest.mark.parametrize(
    "device_str",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_solver_runs_on_available_devices(device_str):
    device = torch.device(device_str)
    Lx, Nx, dt = 10.0, 32, 0.005
    dx = Lx / Nx

    x = torch.linspace(-Lx / 2, Lx / 2, Nx, device=device)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    r2 = X**2 + Y**2

    k_x, k_y = wave_vectors_for_SFSM_method([Lx, Lx, Nx, Nx], device=device)
    params = [k_x, k_y, dt]
    eq = eq_with_SFSM_parameters(r=r2, g=1.0, alpha=0.5, sigma=0.3)

    psi = torch.exp(-r2 / 4.0).to(torch.complex128)
    psi_next = solver_SFSM(Parameters=params, initial_condition=psi, special_function=eq)

    assert psi_next is not None
    assert psi_next.device.type == device.type
    assert not torch.isnan(psi_next).any()

def test_norm_conservation_conservative_limit():
    """With zero pumping (alpha=0) and dissipation (sigma=0), total norm must be conserved."""
    Lx, Nx, dt = 20.0, 64, 0.005
    dx = Lx / Nx
    x = torch.linspace(-Lx / 2, Lx / 2, Nx)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    r2 = X**2 + Y**2

    k_x, k_y = wave_vectors_for_SFSM_method([Lx, Lx, Nx, Nx], device="cpu")
    params = [k_x, k_y, dt]
    eq = eq_with_SFSM_parameters(r=0.5 * r2, g=1.0, alpha=0.0, sigma=0.0)

    psi = torch.exp(-(r2) / 4.0).to(torch.complex128)
    initial_norm = torch.sum(torch.abs(psi) ** 2).item() * dx**2

    for _ in range(100):
        psi = solver_SFSM(Parameters=params, initial_condition=psi, special_function=eq)

    final_norm = torch.sum(torch.abs(psi) ** 2).item() * dx**2
    assert np.isclose(initial_norm, final_norm, rtol=1e-5)


def test_harmonic_oscillator_ground_state():
    """For V = 0.5*r^2 and g=alpha=sigma=0, psi = exp(-r^2/2) is stationary in density."""
    Lx, Nx, dt = 16.0, 64, 0.001
    dx = Lx / Nx
    x = torch.linspace(-Lx / 2, Lx / 2, Nx)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    r2 = X**2 + Y**2

    k_x, k_y = wave_vectors_for_SFSM_method([Lx, Lx, Nx, Nx], device="cpu")
    params = [k_x, k_y, dt]
    # Note: kinetic term in eq_with_SFSM_parameters has factor -1 (corresponding to -nabla^2),
    # so V = r^2 matches the ground state exp(-r^2 / 2).
    eq = eq_with_SFSM_parameters(r=r2, g=0.0, alpha=0.0, sigma=0.0)

    psi_0 = torch.exp(-r2 / 2.0).to(torch.complex128)
    psi = psi_0.clone()

    for _ in range(200):
        psi = solver_SFSM(Parameters=params, initial_condition=psi, special_function=eq)

    density_diff = torch.max(torch.abs(torch.abs(psi) ** 2 - torch.abs(psi_0) ** 2)).item()
    assert density_diff < 5*1e-3