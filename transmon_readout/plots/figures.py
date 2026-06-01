"""
Publication-quality plots for the transmon readout simulation.

Figures
-------
1. S21 transmission: |g> and |e> resonances (magnitude + phase)
2. IQ plane: resonance circles + signal points + noise ellipses
3. SNR accumulation vs time — three amplifier chains overlaid
4. IQ plane comparison: three chains side by side at equal t_int
5. Noise budget bar chart: Friis breakdown per chain
6. Parameter sweeps: t_int vs E_J/E_C and vs n̄ for all three chains
7. Charge dispersion of the qubit
"""

from __future__ import annotations

from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy.special import erfc

from ..core.transmon import compute_spectrum, charge_dispersion
from ..readout.resonator import ReadoutResonator, DispersiveCoupler
from ..measurement.noise import (AmplifierChain, MeasurementSetup, chain_lna_only, chain_hemt_lna, chain_jpa_hemt_lna,)


# ── Shared palette ────────────────────────────────────────────────────────────

CHAIN_COLORS = {
    'LNA only':         '#DC2626',   # red
    'HEMT + LNA':       "#D9A806",   # amber
    'JPA + HEMT + LNA': "#240596", # blue-violet
}
CHAIN_LABELS = list(CHAIN_COLORS.keys())

COLORS = {
    'ground':  '#2563EB',
    'excited': "#DC2926",
    'accent':  '#059669',
    'chi':     '#7C3AED',
}

font_size = 20

def _style():
    plt.rcParams.update({
        'font.family':        'serif',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.labelsize':     font_size + 1,
        'axes.titlesize':     font_size + 2,
        'legend.fontsize':    font_size - 2,
        'xtick.labelsize':    font_size,
        'ytick.labelsize':    font_size,
        'figure.dpi':         150,
    })


def _make_three_setups(coupler: DispersiveCoupler, n_bar: float) -> dict[str, MeasurementSetup]:
    """Return {label: MeasurementSetup} for all three amplifier chains."""
    omega_r = coupler.resonator.omega_r
    return {
        'LNA only':            MeasurementSetup(coupler, chain_lna_only(omega_r),    n_bar),
        'HEMT + LNA':          MeasurementSetup(coupler, chain_hemt_lna(omega_r),    n_bar),
        'JPA + HEMT + LNA':    MeasurementSetup(coupler, chain_jpa_hemt_lna(omega_r), n_bar),
    }


# ── Figure 1: S21 shift ───────────────────────────────────────────────────────

def plot_S21_shift(setup: MeasurementSetup, ax_mag=None, ax_phase=None, n_points: int = 2000, show: bool = True) -> plt.Figure:
    """
    |S_{21}|(omega) and phase for |g⟩ and |e⟩, with optimal probe marked.
    When a microwave probe signal passes through a readout line coupled to these resonators, the resonators "suck" energy out of the line at their resonant frequencies. 
    This energy absorption results in the dips seen in the transmission spectrum
    """
    _style()
    r = setup.resonator
    coupler = setup.coupler

    hw = max(5 * r.kappa, 3 * abs(coupler.chi))
    freqs = np.linspace(r.omega_r - hw, r.omega_r + hw, n_points)
    freqs_MHz = (freqs - r.omega_r) * 1e3

    S_g = r.S21(freqs, shift=coupler.shift_g)
    S_e = r.S21(freqs, shift=coupler.shift_e)
    opt_f = setup.optimal_probe_freq()
    opt_MHz = (opt_f - r.omega_r) * 1e3

    if ax_mag is None or ax_phase is None:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    else:
        fig = ax_mag.get_figure()

    ax_mag.plot(freqs_MHz, 20 * np.log10(np.abs(S_g)), color=COLORS['ground'],  lw=2, label='|g⟩')
    ax_mag.plot(freqs_MHz, 20 * np.log10(np.abs(S_e)), color=COLORS['excited'], lw=2, label='|e⟩', ls='--')
    ax_mag.axvline(opt_MHz, color=COLORS['accent'], lw=1.5, ls=':', label='Opt. probe')
    ax_mag.axvspan(coupler.shift_g * 1e3, coupler.shift_e * 1e3, alpha=0.07, color=COLORS['chi'], label=f'2|χ| = {coupler.separation_MHz:.1f} MHz')
    ax_mag.set_ylabel(r'$|S_{21}|$ (dB)', fontsize=font_size)
    ax_mag.set_title('Dispersive Readout: Resonator Transmission')
    ax_mag.legend(loc='lower right', framealpha=0.9)
    ax_mag.grid(True, alpha=0.3)

    ax_phase.plot(freqs_MHz, np.degrees(np.angle(S_g)), color=COLORS['ground'],  lw=2)
    ax_phase.plot(freqs_MHz, np.degrees(np.angle(S_e)), color=COLORS['excited'], lw=2, ls='--')
    ax_phase.axvline(opt_MHz, color=COLORS['accent'], lw=1.2, ls=':')
    ax_phase.set_xlabel(r'Detuning from $\omega_r$ (MHz)', fontsize=font_size)
    ax_phase.set_ylabel(r'$\angle S_{21}$ (deg)', fontsize=font_size)
    ax_phase.grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── Figure 2: SNR accumulation — three chains ─────────────────────────────────

def plot_snr_comparison(coupler: DispersiveCoupler, n_bar: float = 2.0, snr_target: float = 4.0, ax=None, show: bool = True) -> plt.Figure:
    """
    SNR(t) curves for LNA-only, HEMT+LNA, JPA+HEMT+LNA on one axis.
    Right y-axis shows fidelity.
    Plot shows how quickly the SNR accumulates for each chain, and the corresponding readout fidelity as a function of integration time.
    Takeout: parametric amplifiers can achieve the same SNR (and thus fidelity) in a fraction of the time compared to traditional HEMT amplifiers, which is crucial for fast qubit readout and high-throughput quantum computing applications.
    Vertical dashed lines mark the integration time to reach the SNR target.
    """
    _style()
    setups = _make_three_setups(coupler, n_bar)

    # Determine a common time axis: long enough for the slowest (LNA only)
    t_lna = setups['LNA only'].integration_time(snr_target)['t_int_ns']
    t_max = max(2.0 * t_lna, 1e3)

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(9, 5))
    else:
        ax1 = ax
        fig = ax.get_figure()
    ax2 = ax1.twinx()

    for label, col in CHAIN_COLORS.items():
        ms = setups[label]
        data = ms.snr_vs_time(t_max_ns=t_max, n_points=1200)
        t_us = data['t_ns'] / 1e3
        snr  = data['snr']
        fid  = data['fidelity'] * 100

        ax1.plot(t_us, snr, color=col, lw=2.2, label=label)
        ax2.plot(t_us, fid, color=col, lw=2., ls=':', alpha=0.8)

        # Mark target integration time
        t_mark = ms.integration_time(snr_target)['t_int_us']
        ax1.axvline(t_mark, color=col, lw=1.8, ls='--', alpha=0.85)
        ax1.text(t_mark * 1.01, snr_target + 500.15, f'{t_mark:.1f} μs', fontsize=10.5, color=col)

    # ax1.axhline(snr_target, color='grey', lw=1, ls='-', alpha=0.5, label=f'SNR = {snr_target:.0f} target')
    
    ax1.set_xlabel('Integration time (μs)')
    ax1.set_ylabel('SNR')
    # ax1.tick_params(axis='y', labelcolor='#1e3a8a')
    ax2.set_ylabel('Fidelity (%)')
    # ax2.tick_params(axis='y', labelcolor='#374151')
    ax2.set_ylim(50, 100)
    ax1.set_ylim(0)
    ax1.set_title(r'$\text{SNR Accumulation}: \bar{n} = %d \, \text{photons}, \, \chi = %.2f \, \text{MHz}, \, \text{SNR target} = %d$' % (n_bar, coupler.chi_MHz, snr_target))
    ax1.legend(loc='center right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── Figure 3: IQ plane comparison — three chains ──────────────────────────────

def plot_IQ_comparison(coupler: DispersiveCoupler, n_bar: float = 2.0, snr_target: float = 4.0, show: bool = True) -> plt.Figure:
    """
    Three IQ panels side by side (one per chain) at the same integration time.
    The noise ellipses are drawn to the same scale, making the benefit of acolder amplifier immediately visible.

    S_{21} defines a circe in the IQ plane; the |g⟩ and |e⟩ states correspond to two points on this circle, separated by a distance proportional to the dispersive shift χ. 
    The centers for |g⟩ and |e⟩ are plotted as points in the IQ plane, and the optimal probe frequency is chosen to maximize the distance between these two points, thus maximizing the readout contrast.

    The noise in the measurement can be represented as ellipses around these points, where the size of the ellipse indicates the uncertainty in distinguishing between the two states. 
    By plotting these ellipses for different amplifier chains, we can visually compare how much each chain reduces the time to achieve the same sigma.
    """
    _style()
    setups = _make_three_setups(coupler, n_bar)
    t_int_dict = {lbl: ms.integration_time(snr_target)['t_int_ns'] for lbl, ms in setups.items()}

    r = coupler.resonator
    hw = max(5 * r.kappa, 3 * abs(coupler.chi))
    freqs = np.linspace(r.omega_r - hw, r.omega_r + hw, 1500)

    V_g_trace = r.S21(freqs, shift=coupler.shift_g)
    V_e_trace = r.S21(freqs, shift=coupler.shift_e)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for ax, (label, col) in zip(axes, CHAIN_COLORS.items()):
        ms = setups[label]  # Get the MeasurementSetup for this chain
        t_ns = t_int_dict[label] # integration time to reach the SNR target for this chain
        opt_f = ms.optimal_probe_freq() # optimal probe frequency for this chain
        iq = ms.IQ_signal(opt_f) # Get the IQ signal (V_g, V_e) for this chain at the optimal probe frequency

        V_g = iq['V_g']  # IQ point for |g⟩ state
        V_e = iq['V_e']  # IQ point for |e⟩ state

        # Noise radius: sigma \propto sqrt(N_sys_quanta / (κ_ext n̄ t))
        sigma = np.sqrt(ms.chain.noise_quanta / (r.kappa_ext * n_bar * t_ns + 1e-20))

        print(f'{label}: N_sys={ms.chain.N_sys:.2f} quanta, t_int={t_ns/1e3:.2f} μs, sigma={sigma:.3f}')
        ax.plot(V_g_trace.real, V_g_trace.imag, color=COLORS['ground'], alpha=0.25, lw=1.2)
        ax.plot(V_e_trace.real, V_e_trace.imag, color=COLORS['excited'], alpha=0.25, lw=1.2)

        for center, state_col in [(V_g, COLORS['ground']), (V_e, COLORS['excited'])]:
            ax.scatter([center.real], [center.imag], s=90, color=state_col, zorder=5)
            print(f'{label} - {state_col}: center at ({center.real:.3f}, {center.imag:.3f}), sigma={sigma:.3f}')
            ellipse = Ellipse(
                (center.real, center.imag),
                width=2*sigma, height=2*sigma,
                edgecolor=state_col, facecolor=state_col,
                alpha=0.18, lw=1.5,
            )
            ax.add_patch(ellipse)

        # Separation arrow
        ax.annotate('', xy=(V_e.real, V_e.imag), xytext=(V_g.real, V_g.imag),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
        mid = (V_g + V_e) / 2
        ax.text(mid.real + 0.02, mid.imag + 0.02,
                r'$|\Delta V|= %.3f$' % iq["contrast"], fontsize=10)

        ax.set_title(f'{label}\nN_sys={ms.chain.N_sys:.1f}  '
                     f't_int={t_ns/1e3:.2f} μs', fontsize=14, color=col)
        ax.set_xlabel(r'$I (\text{Re}[S_{21}])$')
        ax.set_ylabel(r'$Q (\text{Im}[S_{21}])$')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.4, alpha=0.3)
        ax.axvline(0, color='k', lw=0.4, alpha=0.3)

    # fig.suptitle('IQ Plane — Noise Ellipses at SNR = 4\n'
    #              r'(ellipse radius = 1 $\sigma$)', fontsize=18, fontweight='bold')
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── Figure 4: Noise budget bar chart ─────────────────────────────────────────

def plot_noise_budget(coupler: DispersiveCoupler,
                       n_bar: float = 2.0,
                       ax=None, show: bool = True) -> plt.Figure:
    """
    Stacked bar chart of the Friis noise contributions for each chain.
    Stages are stacked; the vacuum floor (0.5) is shown as a baseline.
    """
    _style()
    setups = _make_three_setups(coupler, n_bar)

    stage_colors = {
        'LNA':          '#DC2626',
        'HEMT':         '#D97706',
        'JPA':          '#059669',
        'JPA (sq)':     '#34d399',
        'Quantum (SQL)':'#6B7280',
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    x_positions = np.arange(len(setups))
    bar_width = 0.55
    chain_labels_short = ['LNA only\n(300 K)', 'HEMT+LNA\n(4 K + 300 K)', 'JPA+HEMT+LNA\n(10 mK + 4 K + 300 K)']

    for xi, (label, ms) in enumerate(setups.items()):
        budget = ms.noise_budget()  # Get the noise budget breakdown for this chain
        bottom = 0.0
        for stage_info in budget['stages']:
            sname = stage_info['name'] # e.g. 'LNA', 'HEMT', 'JPA'
            nc = stage_info['N_contrib'] # noise contribution in quanta
            col = stage_colors.get(sname, '#94a3b8') # default to gray if stage name not in colors
            ax.bar(xi, nc, bottom=bottom, width=bar_width, 
                   color=col, label=sname if xi == 0 else '_nolegend_',
                   edgecolor='white', linewidth=0.8) # Stack the bars for each stage
            if nc > 0.5: # Only label contributions that are significant enough to read
                ax.text(xi, bottom + nc/2, f'{nc:.1f}',
                        ha='center', va='center', fontsize=10.5, color='white',
                        fontweight='bold') 
            bottom += nc

        # Quantum floor bar
        ax.bar(xi, 0.5, bottom=0, width=bar_width,
               color=stage_colors['Quantum (SQL)'], alpha=0.4,
               label='Quantum floor (0.5)' if xi == 0 else '_nolegend_',
               hatch='///', edgecolor='white')

        # Total label
        ax.text(xi, bottom + 0.5, f'N_sys={ms.chain.N_sys:.1f}',
                ha='center', va='bottom', fontsize=12)

    # SQL line
    ax.axhline(0.5, color='grey', ls=':', lw=1.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(chain_labels_short, fontsize=14)
    ax.set_ylabel('Noise contribution (photons)')
    ax.set_title('Friis Noise Budget')
    ax.set_yscale('log')
    ax.set_ylim(0.1, None)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels)
              if not (l in seen or seen.add(l))]
    ax.legend(*zip(*unique), loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── Figure 5: t_int table summary ────────────────────────────────────────────

def plot_tint_summary(coupler: DispersiveCoupler, n_bar: float = 2.0, ax=None, show: bool = True) -> plt.Figure:
    """
    Grouped bar chart of t_int for three SNR targets across the three chains.
    """
    _style()
    setups = _make_three_setups(coupler, n_bar)
    snr_targets = [1.0, 4.0, 9.0]
    snr_labels  = ['SNR=1\n(F≈84%)', 'SNR=4\n(F≈97.7%)', 'SNR=9\n(F≈99.9%)']

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    n_snr   = len(snr_targets)
    n_chain = len(setups)
    width   = 0.22
    x       = np.arange(n_snr)

    for ci, (label, col) in enumerate(CHAIN_COLORS.items()):
        ms = setups[label]
        t_vals = [ms.integration_time(s)['t_int_us'] for s in snr_targets]
        offset = (ci - 1) * width
        bars = ax.bar(x + offset, t_vals, width=width, color=col,
                      label=label, edgecolor='white', linewidth=0.6)
        for bar, tv in zip(bars, t_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.05,
                    f'{tv:.2f}' if tv < 100 else f'{tv:.0f}',
                    ha='center', va='bottom', fontsize=10.5, color=col,
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set_ylabel('Integration time (μs)')
    ax.set_title(r'$\text{Integration Time}, \,\bar{{n}} = %d \, \text{photons}$' % n_bar)
    ax.set_yscale('log')
    # ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── Figure 6: Parameter sweeps ────────────────────────────────────────────────

def plot_parameter_sweeps(base_setup: MeasurementSetup, show: bool = True) -> plt.Figure:
    """
    Left:  t_int vs E_J/E_C for all three chains
    Right: t_int vs n̄ for all three chains

    Increasing E_J/E_C reduces |alpha| and thus reduces |chi|, making the resonator shift smaller and readout more challenging. 
    The strong-dispersive limit (2|chi| > K) is desirable for high-fidelity single-shot readout.
    """
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    coupler = base_setup.coupler
    r = coupler.resonator
    E_C = coupler.qubit.E_C

    ratios  = np.linspace(10, 100, 50)
    n_bars  = np.logspace(-1, 2, 70)

    for label, col in CHAIN_COLORS.items():
        t_ratio = []
        for ratio in ratios:
            E_J = ratio * E_C
            spec = compute_spectrum(E_C=E_C, E_J=E_J, n_g=0.0)
            c = DispersiveCoupler(qubit=spec, resonator=r, g=coupler.g)
            # print chi value
            print(f'{label}: E_J/E_C={ratio:.1f}, chi={c.chi_MHz:.2f} MHz')
            chain = (chain_lna_only(r.omega_r)    if 'JPA' not in label and 'HEMT' not in label
                     else chain_hemt_lna(r.omega_r) if 'JPA' not in label
                     else chain_jpa_hemt_lna(r.omega_r))
            ms = MeasurementSetup(coupler=c, chain=chain, n_bar=base_setup.n_bar)
            t_ratio.append(ms.integration_time(snr_target=4.0)['t_int_us'])
        ax1.plot(ratios, t_ratio, color=col, lw=2, label=label)

        t_nbar = []
        for nb in n_bars:
            chain = (chain_lna_only(r.omega_r)    if 'JPA' not in label and 'HEMT' not in label
                     else chain_hemt_lna(r.omega_r) if 'JPA' not in label
                     else chain_jpa_hemt_lna(r.omega_r))
            ms = MeasurementSetup(coupler=coupler, chain=chain, n_bar=nb)
            t_nbar.append(ms.integration_time(snr_target=4.0)['t_int_us'])
        ax2.loglog(n_bars, t_nbar, color=col, lw=2, label=label)

    ax1.set_xlabel(r'$E_J / E_C$')
    ax1.set_ylabel(r'$t_{\text{int}} (\mu s)$  [SNR=4]')
    ax1.set_title(r'Integration Time vs $E_J/E_C$')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(5, 500)

    ax2.set_xlabel('Mean photon number n̄')
    ax2.set_ylabel('t_int (μs)  [SNR=4]')
    ax2.set_title('Integration Time vs Drive Strength')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')


    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── Figure 7: Charge dispersion ───────────────────────────────────────────────

def plot_charge_dispersion(E_C: float, E_J: float,
                            ax=None, show: bool = True) -> plt.Figure:
    """Energy levels vs gate charge n_g."""
    _style()
    data = charge_dispersion(E_C, E_J, n_g_points=200)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    for i, (lbl, col) in enumerate(zip(['E₀', 'E₁', 'E₂'], ['#2563EB', '#DC2626', "#AFC437"])):
        e = data['levels'][:, i] - data['levels'][:, 0].min()
        ax.plot(data['ng'], e, color=col, lw=2.2, label=lbl)

    ax.set_xlabel('Gate charge n_g (e)', fontsize=font_size)
    ax.set_ylabel('Energy (GHz, offset)', fontsize=font_size)
    ax.set_title(
        r'$\text{Charge Dispersion}  E_J/E_C= %.0f, \delta\omega_{01} = %.4f$' % (E_J/E_C, data["qubit_dispersion_MHz"])
    )
    ax.legend(fontsize=font_size)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── Master figure ─────────────────────────────────────────────────────────────

def plot_all(setup: MeasurementSetup, save_path: str | None = None) -> plt.Figure:
    """
    Generate a single comprehensive figure with all panels.
    Uses the setup's coupler; compares all three amplifier chains internally.
    """
    _style()
    coupler = setup.coupler
    n_bar   = setup.n_bar

    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

    ax_mag    = fig.add_subplot(gs[0, 0:2])
    ax_phase  = fig.add_subplot(gs[1, 0:2])
    ax_disp   = fig.add_subplot(gs[0:2, 2])
    ax_snr    = fig.add_subplot(gs[2, 0:3])
    ax_budget = fig.add_subplot(gs[3, 0:2])
    ax_tint   = fig.add_subplot(gs[3, 2])

    plot_S21_shift(setup, ax_mag=ax_mag, ax_phase=ax_phase, show=False)
    plot_charge_dispersion(coupler.qubit.E_C, coupler.qubit.E_J, ax=ax_disp, show=False)
    plot_snr_comparison(coupler, n_bar=n_bar, ax=ax_snr, show=False)
    plot_noise_budget(coupler, n_bar=n_bar, ax=ax_budget, show=False)
    plot_tint_summary(coupler, n_bar=n_bar, ax=ax_tint, show=False)

    fig.suptitle(
        'Transmon Dispersive Readout — Amplifier Chain Comparison\n'
        f'E_C={coupler.qubit.E_C:.3f} GHz,  E_J={coupler.qubit.E_J:.2f} GHz,  '
        f'ω_r={coupler.resonator.omega_r:.3f} GHz,  '
        f'g={coupler.g*1e3:.0f} MHz,  χ={coupler.chi_MHz:.2f} MHz,  '
        f'n̄={n_bar} photons',
        fontsize=11, fontweight='bold', y=0.995,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig



# ── Fig 8: Signal accumulation — separating Gaussian clouds ──────────────────

def plot_signal_accumulation(setup, n_points=500, ax=None, show=True):
    """
    Show how the accumulated IQ signal separates the two qubit states over time.

    Physical picture
    ----------------
    At each time step the detector adds one projected sample y_i to the running
    sum Y(T) = sum y_i.  Because every sample has mean mu_{g,e} = -/+ delta/2
    (where delta = |V_g - V_e| is the IQ separation), the running sum has means
    that grow linearly in T:

        mu_e(T) = +(delta/2) * kappa * T      (kappa*T = number of samples M)
        mu_g(T) = -(delta/2) * kappa * T

    The variance of Y(T) is the sum of M independent per-sample variances:

        Var[Y(T)] = M * sigma^2_1 = (kappa*T) * (N_sys/2)

    so the 1-sigma band grows only as sqrt(T):

        sigma_Y(T) = sqrt(kappa * N_sys / 2) * sqrt(T)

    The two Gaussian clouds therefore separate linearly while their widths grow
    as sqrt(T).  The clouds become distinguishable once the separation exceeds
    ~2*sigma_Y, i.e. once SNR_power >= 1.
    """
    _style()

    r       = setup.resonator
    kappa   = r.kappa                        # total linewidth kappa [GHz = ns^-1]
    N_sys   = setup.chain.noise_quanta       # N_sys + 1/2

    opt_f   = setup.optimal_probe_freq()
    iq      = setup.IQ_signal(opt_f)
    delta   = iq['contrast']                 # delta = |V_g - V_e|

    # Time axis: run to 4x the SNR_power=1 crossing so both the noisy and the well-separated regimes are visible.
    # SNR_power = 1  =>  T_cross = N_sys / (delta^2 * kappa)
    T_cross = N_sys / (delta**2 * kappa) if (delta > 0 and kappa > 0) else 1e3
    T_max   = 4.0 * T_cross
    T       = np.linspace(0, T_max, n_points)

    # Mean accumulated signals: mu(T) = +/- (delta/2) * kappa * T
    half_sep = 0.5 * delta * kappa           # slope [ns^-1]
    mu_e     =  half_sep * T
    mu_g     = -half_sep * T

    # 1-sigma band: sigma_Y(T) = sqrt(kappa * N_sys/2 * T)
    sigma_Y  = np.sqrt(kappa * N_sys / 2.0 * T)

    # SNR_power crossing times for annotation markers
    snr_crossings = {}
    for snr_val in [1, 4, 9]:
        tc = snr_val * N_sys / (delta**2 * kappa)
        if tc < T_max:
            snr_crossings[snr_val] = tc

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    col_e = COLORS['excited']
    col_g = COLORS['ground']

    # Mean trajectories
    ax.plot(T, mu_e, color=col_e, lw=2.2, label=r'$+(\delta/2)\,\kappa T$   [|e⟩]')
    ax.plot(T, mu_g, color=col_g, lw=2.2, label=r'$-(\delta/2)\,\kappa T$   [|g⟩]')

    # 1-sigma shaded bands
    ax.fill_between(T, mu_e - sigma_Y, mu_e + sigma_Y, color=col_e, alpha=0.22, label=r'$\pm 1\sigma_Y = \sqrt{\kappa N_{sys} T / 2}$')
    ax.fill_between(T, mu_g - sigma_Y, mu_g + sigma_Y, color=col_g, alpha=0.22)

    # Reference sqrt(T) envelope (dashed grey), anchored at T_max/4
    # sigma_y = np.sqrt(kappa * N_sys / 2.0 * T)
    # ax.plot(T,  sigma_y, color='grey', lw=1.2, ls=':', alpha=0.65, label=r'$\sigma_Y$ envelope (reference)') 
    # ax.plot(T, -sigma_y, color='grey', lw=1.2, ls=':', alpha=0.65)

    # Vertical dashed lines at SNR crossing times
    y_lo, y_hi = mu_g[-1] - sigma_Y[-1] * 1.1, mu_e[-1] + sigma_Y[-1] * 1.1
    for snr_val, tc in snr_crossings.items():
        ax.axvline(tc, color='black', lw=0.9, ls='--', alpha=0.35)
        ax.text(tc * 1.05, y_lo + 0.06 * (y_hi - y_lo),
                f'SNR={snr_val}',
                fontsize=10, color='#374151', alpha=0.8,
                rotation=90, va='bottom')

    ax.set_xlim(0, T_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('Sampling time  $T$  ($\mu$s)', fontsize=16)
    # ax.set_ylabel(r'Accumulated signal  $Y(T) = \sum_i y_i$  (a.u.)', fontsize=16)
    ax.set_title(
        r'$N_{{sys}} = %.2f \text{ph}$ ' % (N_sys)
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:g}'))
    ax.axhline(0, color='k', lw=0.5, alpha=0.25)
    ax.legend(loc='upper left', framealpha=0.92, fontsize=12)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        'Qubit State Separation vs sampling time — Amplifier Chain Comparison\n\n'
        r'$\delta = %.2f,  \kappa = %.2f \text{MHz}$' % (delta, kappa*1e3),
        fontsize=20, fontweight='bold',
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig
