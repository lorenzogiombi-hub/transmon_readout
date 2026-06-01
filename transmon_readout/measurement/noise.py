"""
Amplifier noise and measurement integration time — multi-amplifier chain model.

Chain topology (physical order from coldest to warmest)
---------------------------------------------------------
                 Readout resonator
                        │
                    [JPA]  ← 10 mK stage  (optional first amp)
                        │
                   [HEMT]  ← 4 K stage
                        │
                   [LNA]   ← 300 K room temperature
                        │
                 IQ demodulator

Each stage adds noise referred back to the input via the Friis formula:

    N_sys = N_1 + N_2/G_1 + N_3/(G_1 G_2) + …

where N_i and G_i are the added noise (photons) and power gain of stage i.
Because G_JPA ~ 20 dB and G_HEMT ~ 40 dB, the room-temperature LNA
contributes negligibly once a cold first stage is present.

Amplifier models
----------------
AmplifierType  Location    T_N (typ.)   N_add @ 6 GHz   Notes
─────────────  ─────────   ──────────   ─────────────   ──────────────────────
LNA            300 K       ~50 K        ~1800            Baseline, no cryogenics
HEMT           4 K         2-10 K       70-350           Standard first stage
JPA            10 mK       ~0.05 K      ~2 (near SQL)    Parametric, quantum-limited

The standard quantum limit (SQL) for a phase-insensitive amplifier is N_add ≥ 1/2.
A JPA operating as a phase-sensitive amplifier can beat this on one quadrature
(squeezing), but for IQ readout we assume the phase-insensitive limit N_add ≈ 1/2.

References
----------
Krantz et al., Appl. Phys. Rev. 6, 021318 (2019) — Secs. V, VIII
Alan Salari, Microwave Techniques in Superconducting Quantum Computers
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from ..readout.resonator import ReadoutResonator, DispersiveCoupler

# Physical constants
_HBAR_KB = 7.638e-3   # ℏ / k_B  [K·ns]  (= K / GHz in natural units here)


# ── Individual amplifier stage ────────────────────────────────────────────────

@dataclass
class AmplifierStage:
    """
    Single amplifier stage characterised by its noise temperature and gain.

    Parameters
    ----------
    name       : Human-readable label (e.g. 'HEMT', 'JPA', 'LNA')
    T_N        : Noise temperature [K]
    gain_dB    : Available power gain [dB]
    omega_r    : Resonator frequency [GHz]  — used to convert T_N → N_add
    location   : Physical temperature stage (informational)
    kind       : 'hemt' | 'jpa' | 'lna'  (controls colour and SQL check)
    """
    name:      str
    T_N:       float    # [K]
    gain_dB:   float    # [dB]
    omega_r:   float    # [GHz]
    location:  str = ''
    kind: Literal['hemt', 'jpa', 'lna', 'generic'] = 'generic'

    @property
    def gain_linear(self) -> float:
        return 10 ** (self.gain_dB / 10)

    @property
    def N_add(self) -> float:
        """Added noise in photon units: N_add = k_B T_N / (ℏ ω_r)."""
        return self.T_N / (_HBAR_KB * self.omega_r)

    @property
    def noise_quanta(self) -> float:
        """N_add + 1/2 (includes vacuum fluctuations)."""
        return self.N_add + 0.5

    @property
    def is_quantum_limited(self) -> bool:
        """True if operating within 2x the standard quantum limit."""
        return self.N_add <= 1.0

    @property
    def noise_figure_dB(self) -> float:
        """
        Classical noise figure NF = 10 log10(1 + T_N / T_0)
        referenced to T_0 = 290 K (IEEE standard).
        """
        return 10 * np.log10(1 + self.T_N / 290.0)

    def __repr__(self):
        sql = '★ near-SQL' if self.is_quantum_limited else ''
        return (
            f"  [{self.name:6s}] T_N={self.T_N:.3g} K  "
            f"N_add={self.N_add:.2f}  G={self.gain_dB:.0f} dB  "
            f"NF={self.noise_figure_dB:.1f} dB  {self.location}  {sql}"
        )


# ── Preset factory functions ──────────────────────────────────────────────────

def make_lna(omega_r: float, T_N: float = 50.0, gain_dB: float = 40.0) -> AmplifierStage:
    """
    Room-temperature Low-Noise Amplifier (LNA).
    Typical commercial microwave LNA: T_N ~ 30-100 K.
    """
    return AmplifierStage(
        name='LNA',
        T_N=T_N,
        gain_dB=gain_dB,
        omega_r=omega_r,
        location='300 K (room temperature)',
        kind='lna',
    )


def make_hemt(omega_r: float, T_N: float = 4.0, gain_dB: float = 40.0) -> AmplifierStage:
    """
    Cryogenic High Electron Mobility Transistor (HEMT) at 4 K.
    T_N ~ 2-10 K. Gain ~ 35-45 dB.
    """
    return AmplifierStage(
        name='HEMT',
        T_N=T_N,
        gain_dB=gain_dB,
        omega_r=omega_r,
        location='4 K (LHe stage)',
        kind='hemt',
    )


def make_jpa(omega_r: float, T_N: float = 0.05, gain_dB: float = 20.0, squeezing: bool = False) -> AmplifierStage:
    """
    Josephson Parametric Amplifier (JPA) at the mixing-chamber (10 mK) stage.

    JPA is a quantum-limited amplifier. 
    In the phase-insensitive mode N_add ~ 1/2 (SQL). 
    In the phase-sensitive (squeezing) mode it can go below 1/2 on one quadrature at the cost of the other.

    Parameters
    ----------
    T_N       : Effective noise temperature [K].
                A near-SQL JPA has T_N ~ h omega/2k_B ≈ 0.14 K @ 6 GHz.
                We use 0.05 K as an optimistic but achievable value.
    gain_dB   : Typical JPA gain 15-25 dB.
    squeezing : If True, halve the effective noise (single-quadrature limit).
    """
    effective_T_N = T_N / 2.0 if squeezing else T_N
    return AmplifierStage(
        name='JPA' + (' (sq)' if squeezing else ''),
        T_N=effective_T_N,
        gain_dB=gain_dB,
        omega_r=omega_r,
        location='10 mK (mixing chamber)',
        kind='jpa',
    )


# ── Amplifier chain ───────────────────────────────────────────────────────────

@dataclass
class AmplifierChain:
    """
    Cascaded amplifier chain using the Friis noise formula.

    Stages are ordered from input (closest to qubit) to output.
    The system noise referred to the input is:

        N_sys = N_1 + N_2/G_1 + N_3/(G_1 G_2) + …

    Parameters
    ----------
    stages : list of AmplifierStage, ordered from first (coldest) to last
    """
    stages: list[AmplifierStage]

    @property
    def N_sys(self) -> float:
        """
        System noise in photon units (Friis formula, referred to chain input).
        """
        n_sys = 0.0
        cumulative_gain = 1.0
        for i, stage in enumerate(self.stages):
            n_sys += stage.N_add / cumulative_gain
            cumulative_gain *= stage.gain_linear
        return n_sys

    @property
    def noise_quanta(self) -> float:
        """System noise quanta = N_sys + 1/2."""
        return self.N_sys + 0.5

    @property
    def total_gain_dB(self) -> float:
        return sum(s.gain_dB for s in self.stages)

    @property
    def first_stage(self) -> AmplifierStage:
        return self.stages[0]

    def friis_breakdown(self) -> list[dict]:
        """
        Return the per-stage Friis contribution to system noise.
        """
        rows = []
        cumulative_gain = 1.0
        for stage in self.stages:
            contrib = stage.N_add / cumulative_gain
            rows.append({
                'name': stage.name,
                'N_add': stage.N_add,
                'gain_dB': stage.gain_dB,
                'N_contrib': contrib,
                'fraction': 0.0,   # filled below
            })
            cumulative_gain *= stage.gain_linear

        total = sum(r['N_contrib'] for r in rows)
        for r in rows:
            r['fraction'] = r['N_contrib'] / total if total > 0 else 0.0
        return rows

    def __repr__(self):
        lines = [f"AmplifierChain  N_sys={self.N_sys:.2f}  "
                 f"G_total={self.total_gain_dB:.0f} dB"]
        for s in self.stages:
            lines.append(repr(s))
        return '\n'.join(lines)


# ── Pre-built chain presets ───────────────────────────────────────────────────

def chain_lna_only(omega_r: float) -> AmplifierChain:
    """Baseline: single room-temperature LNA, no cryogenics."""
    return AmplifierChain(stages=[make_lna(omega_r)])


def chain_hemt_lna(omega_r: float) -> AmplifierChain:
    """Standard: cryogenic HEMT at 4 K followed by room-temperature LNA."""
    return AmplifierChain(stages=[
        make_hemt(omega_r, gain_dB=40),
        make_lna(omega_r, gain_dB=30),
    ])


def chain_jpa_hemt_lna(omega_r: float, squeezing: bool = False) -> AmplifierChain:
    """
    State-of-the-art: JPA at 10 mK → HEMT at 4 K → LNA at 300 K.
    With a 20 dB JPA the HEMT and LNA contributions are suppressed by 1/100.
    """
    return AmplifierChain(stages=[
        make_jpa(omega_r, gain_dB=20, squeezing=squeezing),
        make_hemt(omega_r, gain_dB=40),
        make_lna(omega_r, gain_dB=30),
    ])


# ── Measurement setup ─────────────────────────────────────────────────────────

@dataclass
class MeasurementSetup:
    """
    Full measurement chain: qubit + resonator + coupler + amplifier chain.

    Parameters
    ----------
    coupler : DispersiveCoupler with qubit and resonator
    chain   : AmplifierChain (single stage or multi-stage Friis)
    n_bar   : Mean intra-cavity photon number (drive strength)
    """
    coupler: DispersiveCoupler
    chain:   AmplifierChain
    n_bar:   float = 1.0

    # Legacy compatibility: accept a single AmplifierStage wrapped automatically
    def __post_init__(self):
        if isinstance(self.chain, AmplifierStage):
            self.chain = AmplifierChain(stages=[self.chain])

    @property
    def resonator(self) -> ReadoutResonator:
        return self.coupler.resonator

    @property
    def amplifier(self) -> AmplifierChain:
        """Alias so old code using .amplifier still works."""
        return self.chain

    # ── IQ signal ─────────────────────────────────────────────────────────────

    def IQ_signal(self, probe_freq: float | None = None) -> dict:
        """
        IQ voltage for |g⟩ and |e⟩ at a given probe frequency.
        Returns dict with 'V_g', 'V_e', 'contrast', 'angle_deg'.
        """
        if probe_freq is None:
            probe_freq = self.resonator.omega_r

        f = np.atleast_1d(probe_freq)
        r = self.resonator

        V_g = r.S21(f, shift=self.coupler.shift_g)[0]
        V_e = r.S21(f, shift=self.coupler.shift_e)[0]

        contrast = abs(V_g - V_e)
        angle = np.degrees(np.angle(V_e - V_g))

        return {
            'probe_freq': float(probe_freq),
            'V_g': V_g,
            'V_e': V_e,
            'contrast': float(contrast),
            'angle_deg': float(angle),
        }

    def optimal_probe_freq(self, n_points: int = 2000) -> float:
        """Frequency maximising IQ contrast |V_g - V_e| within ±5kappa."""
        r = self.resonator
        freqs = np.linspace(
            r.omega_r - 5 * r.kappa,
            r.omega_r + 5 * r.kappa,
            n_points,
        )
        V_g = r.S21(freqs, shift=self.coupler.shift_g)
        V_e = r.S21(freqs, shift=self.coupler.shift_e)
        return float(freqs[np.argmax(np.abs(V_g - V_e))])

    # ── SNR ───────────────────────────────────────────────────────────────────

    def snr_rate(self, probe_freq: float | None = None) -> float:
        """
        SNR accumulation rate Γ_SNR [ns⁻¹].

            Γ_SNR = |V_g - V_e|² * kappa_ext * n̄ / (2 N_sys_quanta)

        The system noise N_sys is from the Friis chain.
        """
        if probe_freq is None:
            probe_freq = self.optimal_probe_freq()

        iq = self.IQ_signal(probe_freq)
        r = self.resonator

        gamma = (iq['contrast'] ** 2 * r.kappa_ext * self.n_bar / (2.0 * self.chain.noise_quanta))
        return float(gamma)

    def integration_time(self, snr_target: float = 4.0, probe_freq: float | None = None,) -> dict:
        """
        Integration time to reach snr_target.
            t_int = snr_target / Γ_SNR   [ns]
            F     = 1 - erfc(√SNR / √2) / 2
        """
        from scipy.special import erfc

        if probe_freq is None:
            probe_freq = self.optimal_probe_freq()

        gamma = self.snr_rate(probe_freq)
        t_int = snr_target / gamma if gamma > 0 else np.inf
        fidelity = 1.0 - 0.5 * erfc(np.sqrt(snr_target) / np.sqrt(2))
        iq = self.IQ_signal(probe_freq)

        return {
            'probe_freq_GHz':  float(probe_freq),
            'snr_rate_per_ns': float(gamma),
            'snr_target':      snr_target,
            't_int_ns':        float(t_int),
            't_int_us':        float(t_int / 1e3),
            'fidelity':        float(fidelity),
            'fidelity_pct':    float(fidelity * 100),
            'IQ_contrast':     float(iq['contrast']),
            'n_bar':           self.n_bar,
            'N_sys':           float(self.chain.N_sys),
            'chi_MHz':         self.coupler.chi_MHz,
            'kappa_MHz':       self.resonator.kappa * 1e3,
        }

    def snr_vs_time(self, t_max_ns: float = 5000.0, n_points: int = 500, probe_freq: float | None = None,) -> dict:
        """SNR(t) = Γ_SNR * t and corresponding fidelity array."""
        from scipy.special import erfc

        if probe_freq is None:
            probe_freq = self.optimal_probe_freq()

        gamma = self.snr_rate(probe_freq)
        t = np.linspace(0, t_max_ns, n_points)
        snr = gamma * t
        fidelity = np.where(snr > 0, 1.0 - 0.5 * erfc(np.sqrt(snr) / np.sqrt(2)),0.5,)
        return {'t_ns': t, 'snr': snr, 'fidelity': fidelity}

    def noise_budget(self) -> dict:
        """Per-stage Friis breakdown plus quantum floor."""
        rows = self.chain.friis_breakdown()
        N_quantum = 0.5
        N_total = self.chain.N_sys + N_quantum
        return {
            'stages': rows,
            'N_sys':     self.chain.N_sys,
            'N_quantum': N_quantum,
            'N_total':   N_total,
        }

    def __repr__(self):
        t = self.integration_time()
        return (
            f"MeasurementSetup(\n"
            f"  n̄_r      = {self.n_bar:.1f} photons\n"
            f"  N_sys    = {self.chain.N_sys:.2f}  (Friis chain)\n"
            f"  Γ_SNR    = {t['snr_rate_per_ns']*1e3:.4f} μs⁻¹\n"
            f"  t_int    = {t['t_int_us']:.3f} μs  (SNR={t['snr_target']})\n"
            f"  Fidelity = {t['fidelity_pct']:.3f} %\n"
            f"{self.chain}\n"
            f")"
        )


# # ── Legacy compatibility shim ─────────────────────────────────────────────────

class Amplifier(AmplifierStage):
    """
    Backward-compatible alias for a single-stage AmplifierStage.
    Old code using Amplifier(T_N=..., gain_dB=..., omega_r=...) still works.
    """
    def __init__(self, T_N: float, gain_dB: float, omega_r: float):
        super().__init__(
            name='Amp',
            T_N=T_N,
            gain_dB=gain_dB,
            omega_r=omega_r,
            kind='generic',
        )
