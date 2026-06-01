"""
Computation of the spectrum of a transmon qubit in the charge basis.

Theory
------
Superconducting qubit described by the Cooper Pair Box (CPB)
Hamiltonian:

    H = 4 E_C (n_hat - n_g)^2 - E_J cos(phi_hat) 

where:
    E_C  : charging energy  [GHz]
    E_J  : Josephson energy [GHz]
    n_g  : dimensionless gate charge (0 to 0.5 by symmetry)
    n_hat: Cooper pair number operator (integer eigenvalues)
    phi_hat: superconducting phase (conjugate to n_hat)

In the charge basis {|n>}, n in [-N_max, N_max]:
    <n | n_hat     | n'> = n  * delta(n, n') --> diagonal term 4 E_C (n - n_g)^2
    <n | cos(phi)  | n'> = 0.5 * delta(|n-n'|, 1)  --> off-diagonal term -E_J/2 (|n><n+1| + |n><n-1|)

So H is a real symmetric tridiagonal matrix that we diagonalize numerically.   --> scipy.linalg.eigh_tridiagonal for efficiency.

Transmon regime: E_J / E_C >> 1 (typically 20-100).
This exponentially suppresses charge noise sensitivity at the cost of reduced anharmonicity: alpha ~ -E_C * (8 E_J/E_C)^{1/4} power-law decay.

References
----------
Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)  
Koch et al. PRA 76, 042319 (2007)
Alan Salari, Microwave Techniques in Superconducting Quantum Computers

"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from dataclasses import dataclass


# ── Define TransmonHamiltonian ──────────────────────────────────────────────────

def build_hamiltonian(E_C: float, E_J: float, n_g: float, N_max: int = 20) -> np.ndarray:
    """
    Build the CPB/transmon Hamiltonian in the charge basis.

    Parameters
    ----------
    E_C   : Charging energy in GHz (= e^2 / 2C_Sigma)
    E_J   : Josephson energy in GHz
    n_g   : Gate charge (dimensionless, 0 to 0.5 by symmetry)
    N_max : Charge basis truncation: n in [-N_max, N_max]

    Returns
    -------
    H : (2*N_max+1, 2*N_max+1) real symmetric numpy array
    H ~ 4 E_C (n_hat - n_g)^2 - (E_J/2) (|n><n+1| + |n><n-1|) in charge basis 
      ~ tridiagonal matrix with diagonal 4 E_C (n - n_g)^2 and off-diagonal -E_J/2
    """
    dim = 2 * N_max + 1
    ns = np.arange(-N_max, N_max + 1, dtype=float)

    diagonal = 4 * E_C * (ns - n_g) ** 2
    off_diagonal = np.full(dim - 1, -E_J / 2)

    H = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    return H


def diagonalize(E_C: float, E_J: float, n_g: float, N_max: int = 20, n_levels: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize the transmon Hamiltonian and return the lowest energy levels
    and eigenvectors.

    Uses scipy's eigh_tridiagonal for efficiency (tridiagonal real symmetric). 

    Tridiagonal matrix: square matrix that contains nonzero entries only on three specific diagonals: 
                        - the main diagonal, 
                        - the diagonal immediately above it, 
                        - and the diagonal immediately below it.
    Parameters
    ----------
    E_C      : Charging energy [GHz]
    E_J      : Josephson energy [GHz]
    n_g      : Gate charge
    N_max    : Charge basis truncation
    n_levels : Number of levels to return

    Returns
    -------
    energies   : (n_levels,) array of eigenvalues in GHz
    eigenvectors: (dim, n_levels) array of eigenvectors in charge basis
    """
    dim = 2 * N_max + 1
    ns = np.arange(-N_max, N_max + 1, dtype=float)

    diag = 4 * E_C * (ns - n_g) ** 2
    off = np.full(dim - 1, -E_J / 2)

    energies, vectors = eigh_tridiagonal(diag, off, eigvals_only=False, select='i', select_range=(0, n_levels - 1))    # Eigenvalues with indices min = 0 <= i <= max = n_levels - 1
    
    return energies, vectors


# ── Build energy sprectrum of transmon qubit from Hamiltonian and diagonalization ───────────────────────────────────────────────────

@dataclass
class TransmonSpectrum:
    """
    Container for the computed spectrum of a transmon qubit.

    All frequencies in GHz unless stated otherwise.
    """
    E_C: float
    E_J: float
    n_g: float
    energies: np.ndarray     # lowest eigenvalues [GHz]
    eigenvectors: np.ndarray # corresponding eigenvectors in charge basis

    @property
    def ratio(self) -> float:
        """E_J / E_C ratio (must be >> 1 for transmon regime)."""
        return self.E_J / self.E_C

    @property
    def omega_01(self) -> float:
        """Qubit transition frequency E1 - E0 [GHz]."""
        return float(self.energies[1] - self.energies[0])

    @property
    def omega_12(self) -> float:
        """Second transition E2 - E1 [GHz]."""
        return float(self.energies[2] - self.energies[1])

    @property
    def anharmonicity(self) -> float:
        """
        Anharmonicity alpha = omega_12 - omega_01 [GHz].
        Negative for a transmon (subharmonic ladder).
        """
        return self.omega_12 - self.omega_01

    @property
    def anharmonicity_MHz(self) -> float:
        """Anharmonicity in MHz."""
        return self.anharmonicity * 1000

    @property
    def analytic_anharmonicity(self) -> float:
        """
        Analytic approximation valid for E_J/E_C >> 1:  alpha ≈ -E_C [GHz].
        """
        return -self.E_C

    def charge_matrix_element(self, m: int, n: int) -> float:
        """
        Matrix element <m|n_hat|n> of the charge operator.
        """
        ns = np.arange(-20, 21, dtype=float)
        return float(np.dot(self.eigenvectors[:, m], ns * self.eigenvectors[:, n]))

    def __repr__(self):
        return (
            f"TransmonSpectrum("
            f"E_C={self.E_C:.3f} GHz, E_J={self.E_J:.3f} GHz, "
            f"EJ/EC={self.ratio:.1f}, ng={self.n_g:.2f})\n"
            f"  fundamental frequency = {self.omega_01:.4f} GHz\n"
            f"  anharmonicity = {self.anharmonicity_MHz:.1f} MHz\n"
            f"  ground_state_energy  = {self.energies[0]:.4f} GHz, "
            f"  first_excited_state  = {self.energies[1]:.4f} GHz, "
            f"  second_excited_state  = {self.energies[2]:.4f} GHz"
        )


def compute_spectrum(E_C: float, E_J: float, n_g: float = 0.0, N_max: int = 20) -> TransmonSpectrum:
    """
    Compute the transmon spectrum for given circuit parameters.

    Parameters
    ----------
    E_C  : Charging energy [GHz]
    E_J  : Josephson energy [GHz]
    n_g  : Gate charge (default 0, sweet spot)
    N_max: Charge basis truncation (20 is accurate for E_J/E_C up to ~100)

    Returns
    -------
    TransmonSpectrum dataclass with energies and derived quantities.
    """
    energies, vectors = diagonalize(E_C, E_J, n_g, N_max)
    return TransmonSpectrum(E_C=E_C, E_J=E_J, n_g=n_g, energies=energies, eigenvectors=vectors)


# ── Charge dispersion ─────────────────────────────────────────────────────────

def charge_dispersion(E_C: float, E_J: float, n_g_points: int = 101, N_max: int = 20) -> dict:
    """
    The charge dispersion quantifies the sensitivity of the energy level to charge noise.
        epsilon_m = max_{ng} E_m(ng) - min_{ng} E_m(ng)
    
    Strategy: Compute energy levels as a function of gate charge n_g over [0, 0.5].

    In the transmon regime it decays exponentially with sqrt(E_J/E_C).

    Parameters
    ----------
    E_C        : Charging energy [GHz]
    E_J        : Josephson energy [GHz]
    n_g_points : Number of gate charge points
    N_max      : Charge basis truncation

    Returns
    -------
    dict with keys:
        'ng'     : array of gate charge values
        'levels' : (n_g_points, 3) array of E0, E1, E2 vs ng
        'epsilon': (3,) charge dispersion for each level [GHz]
        'qubit_dispersion': dispersion of omega_01 = epsilon_1 - epsilon_0 [MHz]
    """
    ng_arr = np.linspace(0, 0.5, n_g_points)
    levels = np.zeros((n_g_points, 3))

    for i, ng in enumerate(ng_arr):
        e, _ = diagonalize(E_C, E_J, ng, N_max, n_levels=3)
        levels[i] = e[:3]

    epsilon = levels.max(axis=0) - levels.min(axis=0)
    qubit_disp = abs(epsilon[1] - epsilon[0]) * 1000  # MHz

    return {
        'ng': ng_arr,
        'levels': levels,
        'epsilon': epsilon,
        'qubit_dispersion_MHz': qubit_disp,
    }


# ── Parameter sweeps ──────────────────────────────────────────────────────────

def sweep_ratio(E_C: float, ratio_min: float = 1.0, ratio_max: float = 100.0, n_points: int = 200, N_max: int = 20) -> dict:
    """
    Sweep E_J/E_C ratio and compute anharmonicity and qubit frequency.
    Useful for visualizing the CPB-to-transmon crossover.

    Parameters
    ----------
    E_C       : Fixed charging energy [GHz]
    ratio_min : Minimum E_J/E_C
    ratio_max : Maximum E_J/E_C
    n_points  : Number of points in the sweep
    N_max     : Charge basis truncation

    Returns
    -------
    dict with keys:
        'ratio'        : E_J/E_C array
        'omega_01'     : qubit frequency [GHz]
        'anharmonicity': anharmonicity [MHz]
        'analytic_anh' : analytic approximation alpha ≈ -E_C [MHz]
    """
    ratios = np.linspace(ratio_min, ratio_max, n_points)
    omega_01 = np.zeros(n_points)
    anharmonicity = np.zeros(n_points)

    for i, r in enumerate(ratios):
        E_J = r * E_C
        e, _ = diagonalize(E_C, E_J, n_g=0.0, N_max=N_max, n_levels=3)
        omega_01[i] = e[1] - e[0]
        anharmonicity[i] = (e[2] - e[1] - (e[1] - e[0])) * 1000

    return {
        'ratio': ratios,
        'omega_01': omega_01,
        'anharmonicity_MHz': anharmonicity,
        'analytic_anharmonicity_MHz': np.full(n_points, -E_C * 1000),
    }


# ── Potential well ────────────────────────────────────────────────────────────

def cosine_potential(E_J: float, phi_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the cosine Josephson potential V(phi) = -E_J * cos(phi).

    Parameters
    ----------
    E_J        : Josephson energy [GHz]
    phi_points : Number of phase points

    Returns
    -------
    phi : phase array from -2pi to 2pi
    V   : potential values [GHz]
    """
    phi = np.linspace(-2 * np.pi, 2 * np.pi, phi_points)
    V = -E_J * np.cos(phi)
    return phi, V