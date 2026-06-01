#!/usr/bin/env python3
"""
run_simulation.py — entry-point script for the transmon readout simulation.

Usage
-----
    python run_simulation.py                # default parameters
    python run_simulation.py --EC 0.3 --EJ 18 --g 100 --nbar 3
"""

import argparse
import sys
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from transmon_readout.core import compute_spectrum
from transmon_readout.readout import ReadoutResonator, DispersiveCoupler
from transmon_readout.measurement import (
    AmplifierChain, MeasurementSetup,
    chain_lna_only, chain_hemt_lna, chain_jpa_hemt_lna,
)

from transmon_readout.plots import (
    plot_all, plot_parameter_sweeps, plot_IQ_comparison,
    plot_S21_shift, plot_charge_dispersion,
    plot_snr_comparison, plot_noise_budget, plot_tint_summary,
    plot_signal_accumulation,
)


def parse_args():
    p = argparse.ArgumentParser(description='Transmon dispersive readout simulation')
    p.add_argument('--EC',   type=float, default = 0.25,  help='Charging energy [GHz]')
    p.add_argument('--EJ',   type=float, default = 28.0,  help='Josephson energy [GHz]')
    p.add_argument('--ng',   type=float, default = 0.0,   help='Gate charge')
    p.add_argument('--wr',   type=float, default = 6.0,   help='Resonator freq [GHz]')
    p.add_argument('--kext', type=float, default = 2.0,   help='kappa_ext [MHz]')
    p.add_argument('--kint', type=float, default = 0.4,   help='kappa_int [MHz]')
    p.add_argument('--g',    type=float, default = 80.0,  help='Coupling strength [MHz]')
    p.add_argument('--nbar', type=float, default = 2.0,   help='Intra-cavity photons')
    p.add_argument('--out',  type=str,   default = 'readout_sim.png',
                   help='Output figure path')
    return p.parse_args()


def print_banner(title, width=66):
    print('\n' + '=' * width)
    pad = (width - len(title) - 2) // 2
    print(' ' * pad + title)
    print('=' * width)


def main():
    args = parse_args()

    print_banner('Transmon Qubit — Dispersive Readout Simulation')

    # ── 1. Qubit ──────────────────────────────────────────────────────────────
    print('\n[1/4] Computing transmon spectrum ...')
    spec = compute_spectrum(E_C=args.EC, E_J=args.EJ, n_g=args.ng)
    print(spec)

    # ── 2. Resonator ─────────────────────────────────────────────────────────
    print('\n[2/4] Building readout resonator ...')
    resonator = ReadoutResonator(
        omega_r=args.wr,
        kappa_ext=args.kext * 1e-3,
        kappa_int=args.kint * 1e-3,
    )
    print(resonator)

    # ── 3. Dispersive coupling ────────────────────────────────────────────────
    print('\n[3/4] Computing dispersive coupling ...')
    coupler = DispersiveCoupler(qubit=spec, resonator=resonator, g=args.g * 1e-3)
    print(coupler)

    # ── 4. Three-chain comparison ─────────────────────────────────────────────
    print('\n[4/4] Comparing amplifier chains ...')

    omega_r = resonator.omega_r
    chains = {
        'LNA only         (300 K)': chain_lna_only(omega_r),
        'HEMT + LNA   (4 K + 300 K)': chain_hemt_lna(omega_r),
        'JPA + HEMT + LNA (10 mK + 4 K + 300 K)': chain_jpa_hemt_lna(omega_r),
    }

    # Friis chain summary
    print()
    for chain_name, chain in chains.items():
        print(f'  Chain: {chain_name}')
        rows = chain.friis_breakdown()
        for row in rows:
            print(f'    {row["name"]:8s}  N_add={row["N_add"]:8.2f}  '
                  f'contrib={row["N_contrib"]:8.3f} ph  ({row["fraction"]*100:.1f}%)')
        print(f'    {"TOTAL":8s}  N_sys={chain.N_sys:8.2f} ph\n')

    # Integration time table
    snr_targets = [1.0, 4.0, 9.0, 16.0]
    fidelities  = [1.0 - 0.5 * __import__('scipy').special.erfc(
                       np.sqrt(s) / np.sqrt(2)) for s in snr_targets]

    col_w = 16
    print('  Integration Time (μs) to reach SNR target')
    header = f'  {"Chain":<42}' + ''.join(
        f'{"SNR="+str(int(s))+" (F="+f"{f*100:.1f}"+"%)":<{col_w}}'
        for s, f in zip(snr_targets, fidelities))
    print(header)
    print('  ' + '-' * (42 + col_w * len(snr_targets)))

    best_chain = None
    for chain_name, chain in chains.items():
        ms = MeasurementSetup(coupler=coupler, chain=chain, n_bar=args.nbar)
        row = f'  {chain_name:<42}'
        for snr in snr_targets:
            t = ms.integration_time(snr_target=snr)['t_int_us']
            row += f'{t:<{col_w}.3f}'
        print(row)
        if best_chain is None:
            best_chain = (chain_name, chain)

    # Speedup factors relative to LNA-only
    print()
    ms_lna = MeasurementSetup(coupler=coupler,
                               chain=chain_lna_only(omega_r), n_bar=args.nbar)
    t_lna = ms_lna.integration_time(snr_target=4.0)['t_int_us']
    print('  Speedup vs LNA-only at SNR=4:')
    for chain_name, chain in chains.items():
        ms = MeasurementSetup(coupler=coupler, chain=chain, n_bar=args.nbar)
        t = ms.integration_time(snr_target=4.0)['t_int_us']
        speedup = t_lna / t
        print(f'    {chain_name:<48} ×{speedup:.1f}')

    # ── 5. Optimal probe frequency ────────────────────────────────────────────
    ms_ref = MeasurementSetup(coupler=coupler,
                               chain=chain_jpa_hemt_lna(omega_r), n_bar=args.nbar)
    opt_f = ms_ref.optimal_probe_freq()
    iq    = ms_ref.IQ_signal(opt_f)
    print(f'\n  Optimal probe frequency: {opt_f:.5f} GHz  '
          f'(delta = {(opt_f-resonator.omega_r)*1e3:+.2f} MHz from ω_r)')
    print(f'  IQ contrast |ΔV| = {iq["contrast"]:.4f}')

    # ── 6. Figures ────────────────────────────────────────────────────────────
    ms_jpa = MeasurementSetup(coupler=coupler, chain=chain_jpa_hemt_lna(omega_r), n_bar=args.nbar)

    def save(fig, path):
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved → {path}')

    print('\nGenerating figures ...')

    # ── Fig A: resonator transmission + charge dispersion ─────────────────────
    fig_a, axes_a = plt.subplots(
        3, 1, figsize=(10, 12),
        gridspec_kw={'height_ratios': [1.2, 1.2, 1.0]},
    )
    plot_S21_shift(ms_jpa, ax_mag=axes_a[0], ax_phase=axes_a[1], show=False)
    plot_charge_dispersion(coupler.qubit.E_C, coupler.qubit.E_J,
                           ax=axes_a[2], show=False)
    fig_a.suptitle(
        'Transmon Dispersive Readout — Resonator & Qubit\n'
         r'$E_C= %.2f \text{GHz}, \,\, E_J= %.2f \text{GHz}, \,\,\omega_r= %.1f \text{GHz}, \,\, g= %.0f \text{MHz},  \,\, \chi= %.2f \text{MHz}$' % (
            coupler.qubit.E_C, coupler.qubit.E_J, resonator.omega_r, coupler.g*1e3, coupler.chi_MHz
        ),
        fontsize=20, fontweight='bold',
    )
    fig_a.tight_layout(rect=[0, 0, 1, 0.96])
    path_a = args.out.replace('.png', '_resonator_qubit.png')
    save(fig_a, path_a)

    # ── Fig B: SNR accumulation + noise budget + integration time ──────────────
    import matplotlib.gridspec as gridspec
    fig_b = plt.figure(figsize=(18, 12))
    gs_b  = gridspec.GridSpec(2, 2, figure=fig_b, hspace=0.42, wspace=0.35)

    ax_snr  = fig_b.add_subplot(gs_b[0, 0:2])
    ax_bud  = fig_b.add_subplot(gs_b[1, 0])
    ax_tint = fig_b.add_subplot(gs_b[1, 1])

    plot_snr_comparison(coupler, n_bar=args.nbar, ax=ax_snr,  show=False)
    plot_noise_budget  (coupler, n_bar=args.nbar, ax=ax_bud,  show=False)
    plot_tint_summary  (coupler, n_bar=args.nbar, ax=ax_tint, show=False)


    fig_b.suptitle(
        'Transmon Dispersive Readout — Noise and SNR analysis\n'
        r'$E_C= %.2f \text{GHz}, \,\, E_J= %.2f \text{GHz}, \,\,\omega_r= %.1f \text{GHz}, \,\, g= %.0f \text{MHz},  \,\, \chi= %.2f \text{MHz}, \,\, \bar{{n}} = %.1f$' % (
            coupler.qubit.E_C, coupler.qubit.E_J, resonator.omega_r, coupler.g*1e3, coupler.chi_MHz, args.nbar
        ),
        fontsize=20, fontweight='bold',
    )
    path_b = args.out.replace('.png', '_measurement.png')
    save(fig_b, path_b)




    # ── Fig C: signal accumulation (one panel per chain) ──────────────────────
    chains_for_accum = [
        ('JPA + HEMT + LNA', chain_jpa_hemt_lna(omega_r)),
        ('HEMT + LNA',       chain_hemt_lna(omega_r)),
        ('LNA only',         chain_lna_only(omega_r)),
    ]
    show_ylabel = [True, False, False]

    fig_c, axes_c = plt.subplots(1, 3, figsize=(18, 8))
    for ax_c, (chain_label, chain), ylabel in zip(axes_c, chains_for_accum, show_ylabel):
        ms_c = MeasurementSetup(coupler=coupler, chain=chain, n_bar=args.nbar)
        plot_signal_accumulation(ms_c, ax=ax_c, show=False)
        # Prepend the chain name to the existing subplot title
        ax_c.set_title(f'{chain_label}\n' + ax_c.get_title(), fontsize=16)
        if ylabel:
            ax_c.set_ylabel(r'Accumulated signal  $Y(T) = \sum_i y_i$  (a.u.)', fontsize=16)
    # fig_c.suptitle(
    #     'Qubit State Separation with time sampling — Amplifier Chain Comparison\n',
    #     # r'Means $\propto T$,  noise bands $\propto \sqrt{T}$',
    #     fontsize=20, fontweight='bold',
    # )
    fig_c.tight_layout(rect=[0, 0, 1, 0.93])
    path_c = args.out.replace('.png', '_signal_accumulation.png')
    save(fig_c, path_c)

    # ── Fig D: IQ plane comparison ─────────────────────────────────────────────
    fig_d = plot_IQ_comparison(coupler, n_bar=args.nbar, show=False)
    fig_d.suptitle('IQ Plane — Noise Ellipses at SNR = 4\n'
                 r'$\text{ellipse radius} = 1 \sigma$', fontsize=20, fontweight='bold'
    )
    fig_d.tight_layout()
    path_d = args.out.replace('.png', '_IQ_comparison.png')
    save(fig_d, path_d)

    # ── Fig E: parameter sweeps ────────────────────────────────────────────────
    fig_e = plot_parameter_sweeps(ms_jpa, show=False)
    path_e = args.out.replace('.png', '_sweeps.png')
    save(fig_e, path_e)

    print('\nDone.')



if __name__ == '__main__':
    main()