# core_solvers.py
"""
The engine of the simulation.
Contains the core numerical methods for solving the convection equations.
All functions are Numba-jitted for high performance.
"""

import numpy as np
from numba import njit, prange
import os 

@njit
def tridiagonal_solver_numba(sub, dia, sup, rhs):
    """A fast, Numba-optimized tridiagonal solver using the Thomas Algorithm."""
    n = len(rhs)
    x = np.zeros_like(rhs)
    d = dia.copy()
    r = rhs.copy()

    # Forward elimination phase
    for i in range(1, n):
        w = sub[i] / d[i - 1]
        d[i] = d[i] - w * sup[i - 1]
        r[i] = r[i] - w * r[i - 1]

    # Backward substitution phase
    x[n - 1] = r[n - 1] / d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (r[i] - sup[i] * x[i + 1]) / d[i]

    return x

@njit
def compute_psi(omg, psi, sub, sup, c, nn, nz, oodz2):
    """Computes the stream function (psi) from vorticity (omg) by solving a Poisson equation."""
    for n in range(1, nn + 1):
        c3 = float(n * n) * c * c
        dia_temp = np.zeros(nz)
        dia_temp[0] = 1.0
        for k in range(1, nz - 1):
            dia_temp[k] = 2.0 * oodz2 + c3
        dia_temp[nz - 1] = 1.0

        # This is where the magic happens - solving the system for each mode
        psi[:, n] = tridiagonal_solver_numba(sub, dia_temp, sup, omg[:, n])
    return psi

@njit(parallel=True)
def compute_linear_terms(Nz, Nn, oodz2, c, Ra, Pr, tem, omg, dtemdt, domgdt):
    """Calculates the linear terms of the equations (diffusion and buoyancy)."""
    for k in prange(1, Nz - 1):
        for n in range(0, Nn + 1):
            # Temperature diffusion
            dtemdt[k, n, 1] = oodz2 * (tem[k + 1, n] - 2 * tem[k, n] + tem[k - 1, n]) - (n * c)**2 * tem[k, n]

            # Vorticity diffusion and buoyancy
            if n > 0:
                domgdt[k, n, 1] = Ra * Pr * (n * c) * tem[k, n] + Pr * (oodz2 * (omg[k + 1, n] - 2 * omg[k, n] + omg[k - 1, n]) - (n * c)**2 * omg[k, n])
            else:
                domgdt[k, 0, 1] = 0

    return dtemdt, domgdt

@njit(parallel=True)
def compute_nonlinear_terms(Nz, Nn, dz, tem, psi, omg, c1, c2, dtemdt, domgdt):
    """Calculates the non-linear advection terms, which make the problem interesting."""
    # This is a complex part of the code, handling interactions between different modes.
    # The original implementation is kept here.
    for k in prange(1, Nz-1):
        # n = 0 case
        tem_sum = 0.0
        for n1 in range(1, Nn+1):
            dpsi_dz = (psi[k+1, n1] - psi[k-1, n1]) / (2 * dz)
            dtem_dt = (tem[k+1, n1] - tem[k-1, n1]) / (2 * dz)
            tem_sum += n1 * (dpsi_dz * tem[k, n1] + psi[k, n1] * dtem_dt)
        dtemdt[k, 0, 1] += (-c2) * tem_sum

        # n > 0 case
        for n in range(1, Nn+1):
            for n1 in range(Nn+1):
                # n2 + n1 = n
                n2 = n - n1
                if 1 <= n2 <= Nn:
                    dtemdt[k,n,1] -= c1 * (-n1 * (psi[k+1, n2] - psi[k-1, n2]) * tem[k, n1] + n2 * psi[k, n2] * (tem[k+1, n1] - tem[k-1, n1]))
                    domgdt[k,n,1] -= c1 * (-n1 * (psi[k+1, n2] - psi[k-1, n2]) * omg[k, n1] + n2 * psi[k, n2] * (omg[k+1, n1] - omg[k-1, n1]))

                # n2 - n1 = n
                n2 = n + n1
                if 1 <= n2 <= Nn:
                    dtemdt[k,n,1] -= c1 * (n1 * (psi[k+1, n2] - psi[k-1, n2]) * tem[k, n1] + n2 * psi[k, n2] * (tem[k+1, n1] - tem[k-1, n1]))
                    if n != 0:
                        domgdt[k,n,1] += c1 * (n1 * (psi[k+1, n2] - psi[k-1, n2]) * omg[k, n1] + n2 * psi[k, n2] * (omg[k+1, n1] - omg[k-1, n1]))

                # n1 - n2 = n
                n2 = n1 - n
                if n != 0 and 1 <= n2 <= Nn:
                    dtemdt[k,n,1] -= c1 * (n1 * (psi[k+1, n2] - psi[k-1, n2]) * tem[k, n1] + n2 * psi[k, n2] * (tem[k+1, n1] - tem[k-1, n1]))
                    domgdt[k,n,1] -= c1 * (n1 * (psi[k+1, n2] - psi[k-1, n2]) * omg[k, n1] + n2 * psi[k, n2] * (omg[k+1, n1] - omg[k-1, n1]))

    return dtemdt, domgdt

@njit(parallel=True)
def update_variables(Nz, Nn, dt, tem, omg, dtemdt, domgdt):
    """Updates the temperature and vorticity fields using the 2nd-order Adams-Bashforth time-stepping scheme."""
    for k in prange(1, Nz - 1):
        for n in range(0, Nn + 1):
            tem[k, n] += dt / 2 * (3 * dtemdt[k, n, 1] - dtemdt[k, n, 0])
            omg[k, n] += dt / 2 * (3 * domgdt[k, n, 1] - domgdt[k, n, 0])
    return tem, omg
