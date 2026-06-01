"""
transmon_readout
================
Numerical simulation of dispersive qubit readout via a coplanar-waveguide resonator.

Modules
-------
core        : Transmon qubit Hamiltonian (from transmon.py)
readout     : Readout resonator (two-port S-parameters) + dispersive shift
measurement : Amplifier noise model and integration time estimation
plots       : Publication-quality figures

Quick start
-----------
>>> from transmon_readout import quick_start
>>> quick_start()
"""

from .core import compute_spectrum, TransmonSpectrum
from .readout import ReadoutResonator, DispersiveCoupler
from .measurement import Amplifier, MeasurementSetup
from .plots import plot_all, plot_parameter_sweeps


def quick_start():
    """Run the default example and print a summary."""
    print("=" * 62)
    print("  Transmon Dispersive Readout — Quick Start")
    print("=" * 62)

    # 1. Transmon qubit
    spec = compute_spectrum(E_C=0.25, E_J=15.0, n_g=0.0)
    print(spec)

    # 2. Readout resonator
    resonator = ReadoutResonator(omega_r=6.0, kappa_ext=2e-3, kappa_int=0.4e-3)
    print("\n", resonator)

    # 3. Dispersive coupling
    coupler = DispersiveCoupler(qubit=spec, resonator=resonator, g=0.08)
    print("\n", coupler)

    # 4. Measurement setup — HEMT at 4 K
    amp = Amplifier(T_N=4.0, gain_dB=40, omega_r=resonator.omega_r)
    setup = MeasurementSetup(coupler=coupler, amplifier=amp, n_bar=2.0)
    print("\n", setup)

    # 5. Integration time table
    print("\n─── Integration Time Budget ───")
    for snr in [1, 4, 9]:
        t = setup.integration_time(snr_target=snr)
        print(f"  SNR={snr:2d}  ->  t_int = {t['t_int_us']:7.2f} us   "
              f"(F = {t['fidelity_pct']:.3f} %)")

    # 6. Plots
    print("\nGenerating figures ...")
    fig = plot_all(setup, save_path="/tmp/readout_sim.png")
    import matplotlib.pyplot as plt
    plt.show()
