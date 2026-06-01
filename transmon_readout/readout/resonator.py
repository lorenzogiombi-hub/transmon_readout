"""
Readout resonator modelled as a two-port network.

Physics overview
----------------
A λ/4 coplanar waveguide (CPW) resonator coupled to a transmission line is well described by its scattering parameter S_{21}(omega).
Near the resonance frequency omega_r:

    S_{21}(omega) = 1 - (K_ext / 2) / (i delta + K/2)

where
    delta      = omega - omega_r   : detuning from resonance
    K      = K_int + K_ext : total linewidth (loss + coupling)
    K_ext  : external (coupling) loss rate
    K_int  : internal (intrinsic) loss rate

Jaynes–Cummings model of a qubit coupled to a resonator:
    H_JC = omega_r (a†a + 1/2) + (omega_01/2) sigma_z + g(a σ₊ + a† σ₋)

When a transmon qubit is dispersively coupled to the resonator:

    H_disp = (omega_r + chi sigma_z) (a†a + 1/2) + (omega_01/2) sigma_z

the resonator frequency shifts by ±chi depending on the qubit state:

    omega_r -> omega_r + chi   (qubit in |e>)
    omega_r -> omega_r - chi   (qubit in |g>)  [sign convention varies]

This is the dispersive readout shift. chi can be computed from the Jaynes–Cummings model in the dispersive limit:

    chi = g^2 / delta_qr   (single-mode, ignoring higher levels)

or more accurately for a transmon (including the |f> level):

    chi = -g^2 alpha / (delta_qr (delta_qr + alpha))

where
    g     : qubit-resonator coupling strength [GHz]
    delta_qr  = omega_01 - omega_r : qubit-resonator detuning
    alpha     : transmon anharmonicity (negative)

Increasing E_J/E_C reduces |alpha| and thus reduces |chi|, making the resonator shift smaller and readout more challenging. 
The strong-dispersive limit (2|chi| > K) is desirable for high-fidelity single-shot readout.

Reference: Alan Salari, Microwave Techniques in Superconducting Quantum Computers
Reference: Koch et al. PRA 76, 042319 (2007)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from ..core.transmon import TransmonSpectrum


# ── Build a resonator class ────────────────────────────────────────────────────────

@dataclass
class ReadoutResonator:
    """
    Hanger-mode λ/4 resonator as a two-port network.

    Parameters
    ----------
    omega_r   : Bare resonator frequency [GHz] (2π already absorbed — angular)
    kappa_ext : External coupling rate [GHz]
    kappa_int : Internal loss rate [GHz]
    """
    omega_r:   float        # [GHz]
    kappa_ext: float        # [GHz]
    kappa_int: float = 0.0  # [GHz]

    @property
    def kappa(self) -> float:
        """Total linewidth κ = κ_ext + κ_int [GHz]."""
        return self.kappa_ext + self.kappa_int

    @property
    def Q_ext(self) -> float:
        """External quality factor Q_ext = ω_r / κ_ext."""
        return self.omega_r / self.kappa_ext

    @property
    def Q_int(self) -> float:
        """Internal quality factor Q_int = ω_r / κ_int."""
        if self.kappa_int == 0:
            return np.inf
        return self.omega_r / self.kappa_int

    @property
    def Q_total(self) -> float:
        """Total quality factor Q = ω_r / κ."""
        return self.omega_r / self.kappa

    def S21(self, freqs: np.ndarray, shift: float = 0.0) -> np.ndarray:
        """
        Transmission S_{21}(omega) of the hanger resonator.

        S_{21} = 1 - (K_ext/2) / (i*delta + K/2)

        Parameters
        ----------
        freqs : frequency array [GHz]
        shift : dispersive shift to apply to ω_r [GHz]

        Returns
        -------
        S21 : complex array, shape = freqs.shape
        """
        omega_res = self.omega_r + shift
        delta = freqs - omega_res
        S21 = 1.0 - (self.kappa_ext / 2.0) / (1j * delta + self.kappa / 2.0)
        return S21

    def S21_dB(self, freqs: np.ndarray, shift: float = 0.0) -> np.ndarray:
        """Return |S_{21}|^2 in dB."""
        return 20 * np.log10(np.abs(self.S21(freqs, shift)))

    def phase(self, freqs: np.ndarray, shift: float = 0.0) -> np.ndarray:
        """Return arg(S_{21}) in radians."""
        return np.angle(self.S21(freqs, shift))

    def group_delay(self, freqs: np.ndarray, shift: float = 0.0) -> np.ndarray:
        """
        Group delay τ = -d(arg S_{21})/d_omega [ns].
        Computed by finite difference.
        """
        df = freqs[1] - freqs[0]
        ph = self.phase(freqs, shift)
        tau = -np.gradient(ph, df)  # [GHz⁻¹] = ns / (2π)
        return tau / (2 * np.pi)   # convert to ns

    def __repr__(self):
        return (
            f"ReadoutResonator(omega_r={self.omega_r:.4f} GHz, "
            f"kappa_ext={self.kappa_ext*1e3:.2f} MHz, "
            f"kappa_int={self.kappa_int*1e3:.2f} MHz, "
            f"Q_total={self.Q_total:.0f})"
        )


# ── Compute dispersive coupling from resonator and transmon qubit ───────────────────────────────────────────────────────

@dataclass
class DispersiveCoupler:
    """
    Compute the dispersive shift chi from qubit and resonator parameters.

    Uses the transmon-accurate formula (including the |f> level):

        chi = -g² alpha / (delta_qr (Delta_qr + alpha))

    where Delta_qr = omega_01 - omega_r (positive when qubit is above resonator).

    Parameters
    ----------
    qubit   : TransmonSpectrum object
    resonator: ReadoutResonator object
    g       : Qubit-resonator coupling strength [GHz]
    """
    qubit:     TransmonSpectrum
    resonator: ReadoutResonator
    g:         float   # [GHz]

    @property
    def delta_qr(self) -> float:
        """Qubit-resonator detuning delta_qr = ω_01 - ω_r [GHz]."""
        return self.qubit.omega_01 - self.resonator.omega_r

    @property
    def chi_bare(self) -> float:
        """
        Single-mode dispersive shift chi = g²/delta_qr [GHz].
        Valid when |delta_qr| >> g (strong-dispersive limit).
        """
        return self.g ** 2 / self.delta_qr

    @property
    def chi(self) -> float:
        """
        Transmon-accurate dispersive shift including anharmonicity:

            chi = -g² alpha / (delta_qr (delta_qr + alpha))

        Returns [GHz]; negative for typical transmon parameters.
        """
        alpha = self.qubit.anharmonicity  # GHz, negative
        dqr = self.delta_qr
        return -(self.g ** 2 * alpha) / (dqr * (dqr + alpha))

    @property
    def chi_MHz(self) -> float:
        return self.chi * 1e3

    @property
    def shift_g(self) -> float:
        """Resonator shift when qubit in |g>: -chi [GHz]."""
        return -self.chi

    @property
    def shift_e(self) -> float:
        """Resonator shift when qubit in |e>: +chi [GHz]."""
        return +self.chi

    @property
    def separation(self) -> float:
        """Frequency separation of the two dressed resonances: 2|chi| [GHz]."""
        return 2 * abs(self.chi)

    @property
    def separation_MHz(self) -> float:
        return self.separation * 1e3

    @property
    def is_resolved(self) -> bool:
        """
        True if the two resonances are frequency-resolved, i.e. 2|chi| > kappa ~ width of the resonator.
        The resolved-sideband (strong dispersive) limit enables single-shot QND readout.
        """
        return self.separation > self.resonator.kappa

    def __repr__(self):
        resolved = "RESOLVED ✓" if self.is_resolved else "UNRESOLVED ✗"
        return (
            f"DispersiveCoupler(\n"
            f"  g       = {self.g*1e3:.1f} MHz\n"
            f"  delta_qr    = {self.delta_qr*1e3:.1f} MHz\n"
            f"  chi (bare)= {self.chi_bare*1e3:.2f} MHz\n"
            f"  chi (full)= {self.chi_MHz:.2f} MHz\n"
            f"  2|chi|    = {self.separation_MHz:.2f} MHz\n"
            f"  kappa_total = {self.resonator.kappa*1e3:.2f} MHz\n"
            f"  Status  : {resolved}\n"
            f")"
        )


def dressed_resonator(coupler: DispersiveCoupler, qubit_state: Literal['g', 'e'], ) -> tuple[ReadoutResonator, float]:
    """
    Return a ReadoutResonator with the dispersive shift applied and the shift value.
    The resonator parameters (kappa, Q) are unchanged; only omega_r shifts.

    Parameters
    ----------
    coupler     : DispersiveCoupler object
    qubit_state : 'g' (ground) or 'e' (excited)

    Returns
    -------
    (resonator_shifted, shift_GHz)
    """
    shift = coupler.shift_g if qubit_state == 'g' else coupler.shift_e
    r = coupler.resonator
    shifted = ReadoutResonator(
        omega_r   = r.omega_r + shift,
        kappa_ext = r.kappa_ext,
        kappa_int = r.kappa_int,
    )
    return shifted, shift
