"""
Percus-Yevick Analytical Solutions for Hard Spheres
====================================================

Exact analytical results from the Percus-Yevick (PY) closure of the
Ornstein-Zernike equation for the hard-sphere fluid.  These serve as
gold-standard reference solutions for validating the neural functional.

Direct Correlation Function
---------------------------
The PY c(r) for hard spheres was solved analytically by Wertheim (1963)
and Thiele (1963):

    c(r) = -lambda_1 - 6*eta*lambda_2*(r/sigma) - eta*lambda_1/2*(r/sigma)^3
           for r < sigma, and 0 otherwise

where
    lambda_1 = (1 + 2*eta)^2 / (1 - eta)^4
    lambda_2 = -(1 + eta/2)^2 / (1 - eta)^4

Structure Factor
----------------
    S(k) = 1 / (1 - rho * c_hat(k))

where c_hat(k) is the 3D Fourier transform of c(r) and rho = 6*eta/(pi*sigma^3).

Compressibility
---------------
PY compressibility route:
    chi_T = (1 - eta)^4 / (1 + 2*eta)^2

Contact Value
-------------
PY contact value:
    g(sigma+) = (1 + eta/2) / (1 - eta)^2

References
----------
M. S. Wertheim, Phys. Rev. Lett. 10, 321 (1963).
E. Thiele, J. Chem. Phys. 39, 474 (1963).
J.-P. Hansen and I. R. McDonald, Theory of Simple Liquids, 4th ed. (2013).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array

jax.config.update("jax_enable_x64", True)


# ══════════════════════════════════════════════════════════════
# DIRECT CORRELATION FUNCTION
# ══════════════════════════════════════════════════════════════


def py_direct_correlation(r: Array, eta: float, sigma: float = 1.0) -> Array:
    """
    Analytical Percus-Yevick direct correlation function c(r).

    The PY c(r) for hard spheres (Wertheim 1963):

        c(r) = -lambda_1 - 6*eta*lambda_2*(r/sigma)
               - eta*lambda_1/2*(r/sigma)^3     for r < sigma

        c(r) = 0                                  for r >= sigma

    where
        lambda_1 = (1 + 2*eta)^2 / (1 - eta)^4
        lambda_2 = -(1 + eta/2)^2 / (1 - eta)^4

    Parameters
    ----------
    r : Array
        Separation distances. Can be scalar or array.
    eta : float
        Packing fraction, eta = pi*rho*sigma^3/6.
    sigma : float, optional
        Hard-sphere diameter. Default 1.0.

    Returns
    -------
    Array
        Direct correlation function c(r) evaluated at each r.
    """
    r = jnp.asarray(r, dtype=jnp.float64)
    eta = jnp.float64(eta)

    lambda_1 = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4
    lambda_2 = -(1.0 + eta / 2.0) ** 2 / (1.0 - eta) ** 4

    x = r / sigma  # reduced distance

    c_inside = (
        -lambda_1
        - 6.0 * eta * lambda_2 * x
        - eta * lambda_1 / 2.0 * x ** 3
    )

    return jnp.where(r < sigma, c_inside, 0.0)


# ══════════════════════════════════════════════════════════════
# FOURIER TRANSFORM OF DIRECT CORRELATION FUNCTION
# ══════════════════════════════════════════════════════════════


def py_direct_correlation_fourier(k: Array, eta: float, sigma: float = 1.0) -> Array:
    """
    Analytical 3D Fourier transform of the PY direct correlation function.

    The isotropic 3D FT of the radially symmetric c(r) is:

        c_hat(k) = (4*pi/k) * integral_0^sigma r*sin(kr)*c(r) dr

    With c(r) polynomial inside the core, this yields closed-form
    integrals J_n(q) = integral_0^1 t^n * sin(qt) dt (t = r/sigma, q = k*sigma).

    The result is expressed via the Ashcroft-Lekner form:

        -rho*c_hat(k) = 24*eta * [a*f1(q) + 6*eta*b*f2(q) + (a*eta/2)*f3(q)]

    where a = lambda_1, b = lambda_2 (PY coefficients), and:

        f1(q) = (sin q - q cos q) / q^3                             [limit 1/3]
        f2(q) = (-q^2 cos q + 2q sin q + 2 cos q - 2) / q^4        [limit 1/4]
        f3(q) = (-q^4 cos q + 4q^3 sin q + 12q^2 cos q
                 - 24q sin q - 24 cos q + 24) / q^6                 [limit 1/6]

    Taylor expansions are used for |q| < 0.1 to avoid catastrophic
    cancellation in the numerators.

    Parameters
    ----------
    k : Array
        Wavevector magnitudes. Can be scalar or array.
    eta : float
        Packing fraction.
    sigma : float, optional
        Hard-sphere diameter. Default 1.0.

    Returns
    -------
    Array
        Fourier transform c_hat(k) evaluated at each k.

    References
    ----------
    N. W. Ashcroft and J. Lekner, Phys. Rev. 145, 83 (1966).
    J.-P. Hansen and I. R. McDonald, Theory of Simple Liquids, 4th ed.,
    Eq. 4.5.17 (2013).
    """
    k = jnp.asarray(k, dtype=jnp.float64)
    eta = jnp.float64(eta)

    rho = 6.0 * eta / (jnp.pi * sigma ** 3)

    # PY coefficients (lambda_1 and lambda_2)
    a = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4
    b = -(1.0 + eta / 2.0) ** 2 / (1.0 - eta) ** 4

    q = k * sigma
    sinq = jnp.sin(q)
    cosq = jnp.cos(q)

    # Switch to Taylor series for small q to avoid division by near-zero
    is_small = jnp.abs(q) < 0.1

    # f1(q) = (sin q - q cos q) / q^3
    # Taylor: 1/3 - q^2/30 + q^4/840
    f1_exact = (sinq - q * cosq) / jnp.where(is_small, 1.0, q ** 3)
    f1_taylor = 1.0 / 3.0 - q ** 2 / 30.0 + q ** 4 / 840.0
    f1_val = jnp.where(is_small, f1_taylor, f1_exact)

    # f2(q) = (-q^2 cos q + 2q sin q + 2 cos q - 2) / q^4
    # Taylor: 1/4 - q^2/36 + q^4/960
    f2_num = -q ** 2 * cosq + 2.0 * q * sinq + 2.0 * cosq - 2.0
    f2_exact = f2_num / jnp.where(is_small, 1.0, q ** 4)
    f2_taylor = 1.0 / 4.0 - q ** 2 / 36.0 + q ** 4 / 960.0
    f2_val = jnp.where(is_small, f2_taylor, f2_exact)

    # f3(q) = (-q^4 cos q + 4q^3 sin q + 12q^2 cos q
    #          - 24q sin q - 24 cos q + 24) / q^6
    # Taylor: 1/6 - q^2/48 + q^4/1080
    f3_num = (-q ** 4 * cosq + 4.0 * q ** 3 * sinq
              + 12.0 * q ** 2 * cosq - 24.0 * q * sinq
              - 24.0 * cosq + 24.0)
    f3_exact = f3_num / jnp.where(is_small, 1.0, q ** 6)
    f3_taylor = 1.0 / 6.0 - q ** 2 / 48.0 + q ** 4 / 1080.0
    f3_val = jnp.where(is_small, f3_taylor, f3_exact)

    # Combine: -rho * c_hat = 24*eta * [a*f1 + 6*eta*b*f2 + (a*eta/2)*f3]
    neg_rho_chat = 24.0 * eta * (a * f1_val + 6.0 * eta * b * f2_val
                                  + a * eta / 2.0 * f3_val)

    c_hat = -neg_rho_chat / rho

    return c_hat


# ══════════════════════════════════════════════════════════════
# STRUCTURE FACTOR
# ══════════════════════════════════════════════════════════════


def py_structure_factor(k: Array, eta: float, sigma: float = 1.0) -> Array:
    """
    Percus-Yevick structure factor S(k) for hard spheres.

        S(k) = 1 / (1 - rho * c_hat(k))

    where rho = 6*eta/(pi*sigma^3) and c_hat(k) is the PY direct
    correlation function in Fourier space.

    Parameters
    ----------
    k : Array
        Wavevector magnitudes.
    eta : float
        Packing fraction.
    sigma : float, optional
        Hard-sphere diameter. Default 1.0.

    Returns
    -------
    Array
        Structure factor S(k).

    Notes
    -----
    At k=0, S(0) = rho*kT*chi_T where chi_T is the isothermal
    compressibility from the compressibility route:

        S(0) = (1-eta)^4 / (1+2*eta)^2

    This provides a consistency check.
    """
    k = jnp.asarray(k, dtype=jnp.float64)
    rho = 6.0 * eta / (jnp.pi * sigma ** 3)

    c_hat = py_direct_correlation_fourier(k, eta, sigma)

    return 1.0 / (1.0 - rho * c_hat)


# ══════════════════════════════════════════════════════════════
# COMPRESSIBILITY
# ══════════════════════════════════════════════════════════════


def py_compressibility(eta: float) -> Array:
    """
    PY isothermal compressibility from the compressibility route.

        chi_T = (1 - eta)^4 / (1 + 2*eta)^2

    This equals S(k=0) and provides a consistency check for the
    structure factor computation.

    Parameters
    ----------
    eta : float
        Packing fraction.

    Returns
    -------
    Array
        Dimensionless compressibility rho*kT*chi_T = S(0).

    Notes
    -----
    The PY compressibility route and virial route give DIFFERENT
    pressures (thermodynamic inconsistency of PY closure).
    The Carnahan-Starling EOS interpolates between the two.
    """
    eta = jnp.float64(eta)
    return (1.0 - eta) ** 4 / (1.0 + 2.0 * eta) ** 2


# ══════════════════════════════════════════════════════════════
# CONTACT VALUE
# ══════════════════════════════════════════════════════════════


def py_contact_value(eta: float, sigma: float = 1.0) -> Array:
    """
    PY contact value of the radial distribution function.

        g(sigma+) = (1 + eta/2) / (1 - eta)^2

    Parameters
    ----------
    eta : float
        Packing fraction.
    sigma : float, optional
        Hard-sphere diameter. Default 1.0 (unused, included for API
        consistency).

    Returns
    -------
    Array
        Contact value g(sigma+).

    Notes
    -----
    The exact (Carnahan-Starling) contact value is:

        g_CS(sigma+) = (1 - eta/2) / (1 - eta)^3

    The PY result underestimates the contact value at high eta.
    """
    eta = jnp.float64(eta)
    return (1.0 + eta / 2.0) / (1.0 - eta) ** 2


# ══════════════════════════════════════════════════════════════
# NUMERICAL g(r) FROM INVERSE FOURIER TRANSFORM
# ══════════════════════════════════════════════════════════════


def compute_py_g_numerical(
    r_values: Array,
    eta: float,
    sigma: float = 1.0,
    k_max: float = 100.0,
    n_k: int = 10000,
) -> Array:
    """
    Numerical pair distribution function g(r) from inverse Fourier
    transform of the PY structure factor.

    Uses the isotropic inverse FT:

        g(r) = 1 + 1/(2*pi^2*rho) * integral_0^inf k^2 [S(k)-1]
                                       * sin(kr)/(kr) dk

    The integral is evaluated by trapezoidal quadrature on a uniform
    k-grid from 0 to k_max.

    Parameters
    ----------
    r_values : Array
        Separation distances at which to evaluate g(r). Should satisfy
        r > 0.
    eta : float
        Packing fraction.
    sigma : float, optional
        Hard-sphere diameter. Default 1.0.
    k_max : float, optional
        Upper cutoff for k integration. Default 100.0. Should be large
        enough that S(k) ~ 1 at k_max.
    n_k : int, optional
        Number of quadrature points. Default 10000.

    Returns
    -------
    Array
        Pair distribution function g(r) at each r value.

    Notes
    -----
    For r < sigma, the exact result is g(r) = 0. Numerical errors from
    the truncated integral may give small nonzero values inside the core.

    The accuracy depends on k_max and n_k. For eta > 0.3, use
    k_max >= 150 and n_k >= 20000 for well-converged results.
    """
    r_values = jnp.asarray(r_values, dtype=jnp.float64)
    rho = 6.0 * eta / (jnp.pi * sigma ** 3)

    # Uniform k grid (exclude k=0 to avoid 0/0 in sin(kr)/(kr))
    dk = k_max / n_k
    k_grid = jnp.linspace(dk, k_max, n_k)

    # S(k) - 1 on the grid (this is the total correlation in Fourier space)
    s_minus_1 = py_structure_factor(k_grid, eta, sigma) - 1.0

    # Integrand for each r: k^2 * [S(k)-1] * sin(kr)/(kr)
    # Shape: (n_r,) from broadcasting (n_k,) with (n_r, 1)
    # We vectorize over r using vmap-like broadcasting.

    def _g_at_r(r):
        """Compute g(r) for a single r value."""
        kr = k_grid * r
        # sin(kr)/(kr) with safe limit at kr->0
        sinc_kr = jnp.where(
            jnp.abs(kr) > 1e-10,
            jnp.sin(kr) / kr,
            1.0 - kr ** 2 / 6.0,
        )
        integrand = k_grid ** 2 * s_minus_1 * sinc_kr
        integral = jnp.trapezoid(integrand, dx=dk)
        return 1.0 + integral / (2.0 * jnp.pi ** 2 * rho)

    # Vectorize over all r values
    g_values = jax.vmap(_g_at_r)(r_values)

    return g_values
