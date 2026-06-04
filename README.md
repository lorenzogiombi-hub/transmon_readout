# Transmon readout chain simulator

A Python package for simulating dispersive readout of superconducting transmon qubits. The simulation covers the full chain from qubit Hamiltonian to measurement fidelity: charge-basis diagonalisation of the transmon, two-port S-parameter modelling of the readout resonator, dispersive coupling between qubit and resonator, and a Friis cascaded noise model for three amplifier configurations (room-temperature LNA, cryogenic HEMT, and Josephson parametric amplifier).

---

## Physics overview

A transmon qubit is coupled dispersively to a coplanar waveguide resonator. In the dispersive limit the qubit-resonator interaction shifts the resonator frequency by ±χ depending on the qubit state, producing two distinguishable transmission responses S₂₁(ω). A microwave probe tone is sent through the feedline, the output is amplified and IQ-demodulated, and the integrated signal is compared against a threshold.

The signal-to-noise ratio accumulates as:

```
SNR_power(T) = δ² · κ · T / N_sys
```

where δ = |V_g − V_e| is the IQ separation between the two qubit states, κ is the total resonator linewidth (integration bandwidth), T is the integration time, and N_sys is the system noise in photon units from the Friis amplifier chain. The readout fidelity is:

```
F = 1 − (1/2) erfc( sqrt(SNR_power / 2) )
```

---

## Installation

```bash
# Clone or unzip the project, then install dependencies
pip install numpy scipy matplotlib
```

No package installation is required — run scripts from the project root directory where `transmon_readout/` is visible.

---

## Quick start

```python
from transmon_readout.core      import compute_spectrum
from transmon_readout.readout   import ReadoutResonator, DispersiveCoupler
from transmon_readout.measurement import MeasurementSetup, chain_jpa_hemt_lna
from transmon_readout.plots     import plot_all

# 1. Transmon qubit
spec = compute_spectrum(E_C=0.25, E_J=15.0, n_g=0.0)
print(spec)

# 2. Readout resonator
resonator = ReadoutResonator(omega_r=6.0, kappa_ext=2e-3, kappa_int=0.4e-3)

# 3. Dispersive coupling
coupler = DispersiveCoupler(qubit=spec, resonator=resonator, g=0.08)
print(coupler)

# 4. Measurement setup with JPA chain
chain = chain_jpa_hemt_lna(omega_r=6.0)
setup = MeasurementSetup(coupler=coupler, chain=chain, n_bar=2.0)
print(setup)

# 5. Integration time to reach SNR = 4
result = setup.integration_time(snr_target=4.0)
print(f"t_int = {result['t_int_us']:.3f} µs   F = {result['fidelity_pct']:.2f}%")

# 6. All figures
fig = plot_all(setup)
```

Or run the command-line entry point:

```bash
python run_simulation.py
python run_simulation.py --EC 0.3 --EJ 18 --g 100 --nbar 3
```

---

## Project structure

```
transmon_readout/
├── core/
│   └── transmon.py        Transmon Hamiltonian and spectrum
├── readout/
│   └── resonator.py       Two-port resonator and dispersive coupling
├── measurement/
│   └── noise.py           Amplifier chain and integration time
├── plots/
│   └── figures.py         All publication-quality figures
└── __init__.py
run_simulation.py           CLI entry point
```

All frequencies are in **GHz** and all times are in **ns** throughout, unless explicitly stated otherwise.

---

## Module reference

### `transmon_readout.core`

#### `TransmonSpectrum`

A dataclass holding the numerically computed spectrum of a transmon qubit. Returned by `compute_spectrum()` — you do not instantiate it directly.

```python
spec = compute_spectrum(E_C=0.25, E_J=15.0)
```

| Attribute / property | Type | Description |
|---|---|---|
| `E_C` | `float` | Charging energy [GHz] |
| `E_J` | `float` | Josephson energy [GHz] |
| `n_g` | `float` | Gate charge (dimensionless) |
| `energies` | `ndarray` | Lowest eigenvalues [GHz] |
| `eigenvectors` | `ndarray` | Eigenvectors in the charge basis |
| `ratio` | `float` | E_J / E_C — the key transmon parameter |
| `omega_01` | `float` | Qubit transition frequency E₁ − E₀ [GHz] |
| `omega_12` | `float` | Second transition E₂ − E₁ [GHz] |
| `anharmonicity` | `float` | α = ω₁₂ − ω₀₁ [GHz], negative for transmon |
| `anharmonicity_MHz` | `float` | Same in MHz |
| `analytic_anharmonicity` | `float` | Approximation α ≈ −E_C [GHz] |
| `charge_matrix_element(m, n)` | `float` | ⟨m\|n̂\|n⟩, relevant for coupling strength |

**Physics note.** The Hamiltonian is

```
H = 4 E_C (n̂ − n_g)² − E_J cos(φ̂)
```

diagonalised in the charge basis {|n⟩, n ∈ [−N_max, N_max]} using `scipy.linalg.eigh_tridiagonal`. The transmon regime E_J/E_C ≫ 1 exponentially suppresses charge noise sensitivity while the anharmonicity decreases only as a weak power law (α ≈ −E_C for large ratios).

---

#### Free functions — `core`

| Function | Returns | Description |
|---|---|---|
| `compute_spectrum(E_C, E_J, n_g=0, N_max=20)` | `TransmonSpectrum` | Main entry point: diagonalise and return spectrum |
| `build_hamiltonian(E_C, E_J, n_g, N_max=20)` | `ndarray` | Build the (2N+1)×(2N+1) tridiagonal Hamiltonian matrix |
| `diagonalize(E_C, E_J, n_g, N_max=20, n_levels=6)` | `(energies, vectors)` | Diagonalise and return the lowest `n_levels` eigenpairs |
| `charge_dispersion(E_C, E_J, n_g_points=101, N_max=20)` | `dict` | Energy levels vs gate charge n_g; includes dispersion ε_m |
| `sweep_ratio(E_C, ratio_min, ratio_max, n_points, N_max)` | `dict` | Sweep E_J/E_C; returns ω₀₁ and α vs ratio |
| `cosine_potential(E_J, phi_points=300)` | `(phi, V)` | Josephson potential V(φ) = −E_J cos(φ) |

---

### `transmon_readout.readout`

#### `ReadoutResonator`

Models a hanger-mode (side-coupled) λ/4 coplanar waveguide resonator as a two-port network. The transmission is

```
S₂₁(ω) = 1 − (κ_ext/2) / (iΔ + κ/2)
```

where Δ = ω − ω_r and κ = κ_ext + κ_int.

```python
resonator = ReadoutResonator(omega_r=6.0, kappa_ext=2e-3, kappa_int=0.4e-3)
```

| Attribute / property | Type | Description |
|---|---|---|
| `omega_r` | `float` | Bare resonator frequency [GHz] |
| `kappa_ext` | `float` | External (coupling) loss rate κ_ext [GHz] |
| `kappa_int` | `float` | Internal (intrinsic) loss rate κ_int [GHz] |
| `kappa` | `float` | Total linewidth κ = κ_ext + κ_int [GHz] |
| `Q_ext` | `float` | External quality factor ω_r / κ_ext |
| `Q_int` | `float` | Internal quality factor ω_r / κ_int |
| `Q_total` | `float` | Loaded quality factor ω_r / κ |
| `S21(freqs, shift=0)` | `ndarray` | Complex S₂₁ at given frequencies with optional dispersive shift |
| `S21_dB(freqs, shift=0)` | `ndarray` | \|S₂₁\| in dB |
| `phase(freqs, shift=0)` | `ndarray` | arg(S₂₁) in radians |
| `group_delay(freqs, shift=0)` | `ndarray` | −d(arg S₂₁)/dω [ns] |

**Physics note.** The hanger geometry means the resonator hangs off the side of a through-feedline rather than terminating it. When the probe is far from ω_r, nearly all power passes straight through (\|S₂₁\| ≈ 1). Near resonance, power is absorbed into the resonator, creating a notch in transmission. This geometry is standard in circuit QED because multiple resonators at different frequencies can share one feedline.

---

#### `DispersiveCoupler`

Computes the dispersive shift χ between a transmon qubit and a readout resonator, using the transmon-accurate formula that accounts for the second excited state |f⟩:

```
χ = −g² α / (Δ_qr (Δ_qr + α))
```

where g is the coupling strength, Δ_qr = ω₀₁ − ω_r is the qubit-resonator detuning, and α is the transmon anharmonicity. This differs from the simpler two-level result χ = g²/Δ_qr by the factor α/(Δ_qr + α).

```python
coupler = DispersiveCoupler(qubit=spec, resonator=resonator, g=0.08)
```

| Attribute / property | Type | Description |
|---|---|---|
| `qubit` | `TransmonSpectrum` | Transmon qubit |
| `resonator` | `ReadoutResonator` | Readout resonator |
| `g` | `float` | Qubit-resonator coupling strength [GHz] |
| `delta_qr` | `float` | Detuning Δ_qr = ω₀₁ − ω_r [GHz] |
| `chi_bare` | `float` | Two-level approximation χ = g²/Δ_qr [GHz] |
| `chi` | `float` | Transmon-accurate dispersive shift [GHz] |
| `chi_MHz` | `float` | Same in MHz |
| `shift_g` | `float` | Resonator shift for qubit in \|g⟩: −χ [GHz] |
| `shift_e` | `float` | Resonator shift for qubit in \|e⟩: +χ [GHz] |
| `separation` | `float` | 2\|χ\| — frequency gap between the two dressed resonances [GHz] |
| `separation_MHz` | `float` | Same in MHz |
| `is_resolved` | `bool` | True if 2\|χ\| > κ (strong-dispersive / single-shot regime) |

**Physics note.** `is_resolved = True` indicates the two resonances are spectrally distinguishable — the resolved-sideband limit where single-shot QND readout is possible without post-selection. The condition 2|χ| > κ is the standard design target.

---

#### Free functions — `readout`

| Function | Returns | Description |
|---|---|---|
| `dressed_resonator(coupler, qubit_state)` | `(ReadoutResonator, float)` | Returns a copy of the resonator with the dispersive shift applied for state `'g'` or `'e'` |

---

### `transmon_readout.measurement`

#### `AmplifierStage`

A single amplifier stage characterised by its noise temperature T_N and power gain. The added noise in photon units is:

```
N_add = k_B T_N / (ħ ω_r)
```

The standard quantum limit for a phase-insensitive amplifier is N_add ≥ 1/2.

```python
stage = AmplifierStage(name='HEMT', T_N=4.0, gain_dB=40, omega_r=6.0,
                       location='4 K', kind='hemt')
```

| Attribute / property | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable label |
| `T_N` | `float` | Noise temperature [K] |
| `gain_dB` | `float` | Available power gain [dB] |
| `omega_r` | `float` | Resonator frequency used for N_add conversion [GHz] |
| `location` | `str` | Physical stage (informational) |
| `kind` | `str` | `'hemt'`, `'jpa'`, `'lna'`, or `'generic'` |
| `gain_linear` | `float` | Power gain as a linear factor |
| `N_add` | `float` | Added noise quanta k_B T_N / (ħ ω_r) |
| `noise_quanta` | `float` | N_add + 1/2 (includes vacuum floor) |
| `is_quantum_limited` | `bool` | True if N_add ≤ 1 (within 2× of the SQL) |
| `noise_figure_dB` | `float` | Classical noise figure referenced to 290 K [dB] |

Three factory functions build physically motivated presets:

| Function | Stage | Typical T_N | Location |
|---|---|---|---|
| `make_lna(omega_r)` | Room-temperature LNA | 50 K | 300 K |
| `make_hemt(omega_r)` | Cryogenic HEMT | 4 K | 4 K stage |
| `make_jpa(omega_r)` | Josephson parametric amplifier | 0.05 K | 10 mK (mixing chamber) |

---

#### `AmplifierChain`

A cascade of `AmplifierStage` objects. The system noise referred to the input is computed with the Friis formula:

```
N_sys = N_1 + N_2/G_1 + N_3/(G_1 G_2) + ...
```

Because the gain of the first stage suppresses the noise contributions of all subsequent stages, the first amplifier dominates. This is why placing a near-quantum-limited JPA at the mixing chamber reduces N_sys from ~90 photons (HEMT alone) to ~2 photons.

```python
chain = AmplifierChain(stages=[make_jpa(6.0), make_hemt(6.0), make_lna(6.0)])
```

| Attribute / property | Type | Description |
|---|---|---|
| `stages` | `list[AmplifierStage]` | Ordered from first (coldest, closest to qubit) to last |
| `N_sys` | `float` | Friis input-referred system noise [photons] |
| `noise_quanta` | `float` | N_sys + 1/2 |
| `total_gain_dB` | `float` | Sum of stage gains [dB] |
| `friis_breakdown()` | `list[dict]` | Per-stage noise contribution and fraction of total |

Three pre-built chains are provided as convenience functions:

| Function | Stages | Typical N_sys @ 6 GHz |
|---|---|---|
| `chain_lna_only(omega_r)` | LNA at 300 K | ~1092 |
| `chain_hemt_lna(omega_r)` | HEMT at 4 K → LNA at 300 K | ~88 |
| `chain_jpa_hemt_lna(omega_r)` | JPA at 10 mK → HEMT at 4 K → LNA at 300 K | ~2 |

---

#### `MeasurementSetup`

The top-level class combining qubit, resonator, coupler, and amplifier chain into a complete readout model. All SNR quantities use the **power convention**:

```
SNR_power(T) = δ² · κ · T / N_sys
F(T)         = 1 − (1/2) erfc( sqrt(SNR_power / 2) )
T_int        = SNR_target · N_sys / (δ² · κ)
```

where δ = |V_g − V_e| is the IQ contrast from S₂₁ at the optimal probe frequency, κ = κ_ext + κ_int is the total resonator linewidth, and N_sys = N_add + 1/2 includes the vacuum floor.

```python
setup = MeasurementSetup(coupler=coupler, chain=chain, n_bar=2.0)
```

| Attribute / property | Type | Description |
|---|---|---|
| `coupler` | `DispersiveCoupler` | Qubit-resonator system |
| `chain` | `AmplifierChain` | Amplifier noise chain |
| `n_bar` | `float` | Mean intra-cavity photon number (informational) |
| `resonator` | `ReadoutResonator` | Shortcut to `coupler.resonator` |
| `amplifier` | `AmplifierChain` | Alias for `chain` |

| Method | Returns | Description |
|---|---|---|
| `optimal_probe_freq()` | `float` | Probe frequency [GHz] that maximises \|V_g − V_e\| |
| `IQ_signal(probe_freq)` | `dict` | Complex phasors V_g, V_e, contrast δ, and angle at the given probe frequency |
| `snr_rate(probe_freq)` | `float` | SNR_power accumulation rate Γ = δ² κ / N_sys [ns⁻¹] |
| `snr_power(t_ns, probe_freq)` | `float` | SNR_power at integration time t [ns] |
| `integration_time(snr_target, probe_freq)` | `dict` | Integration time, fidelity, and diagnostics for a given SNR_power target |
| `snr_vs_time(t_max_ns, n_points, probe_freq)` | `dict` | Arrays of t, SNR_power(t), and F(t) |
| `noise_budget()` | `dict` | Friis breakdown with per-stage noise contributions |

The `integration_time()` return dict contains:

| Key | Description |
|---|---|
| `t_int_ns` / `t_int_us` | Integration time in ns and µs |
| `fidelity` / `fidelity_pct` | Readout fidelity as fraction and percentage |
| `snr_rate_per_ns` | Γ_SNR [ns⁻¹] |
| `IQ_contrast` | δ = \|V_g − V_e\| |
| `kappa_MHz` | Total linewidth [MHz] |
| `N_sys` | System noise quanta |
| `chi_MHz` | Dispersive shift [MHz] |

**SNR target guide:**

| `snr_target` | Fidelity |
|---|---|
| 1 | ~76 % |
| 4 | ~92 % |
| 9 | ~98 % |
| 16 | ~99.6 % |

---

### `transmon_readout.plots`

All functions accept an optional `ax` argument for embedding into existing figures, and a `show=True` flag. They return the `matplotlib.figure.Figure` object.

| Function | Description |
|---|---|
| `plot_S21_shift(setup)` | \|S₂₁\|(ω) and ∠S₂₁(ω) for \|g⟩ and \|e⟩, with optimal probe frequency marked |
| `plot_snr_comparison(coupler, n_bar)` | SNR_power(t) for all three amplifier chains on one axis, with fidelity on the right axis |
| `plot_IQ_comparison(coupler, n_bar)` | Three IQ-plane panels (one per chain) with resonance circle traces and 1σ noise ellipses |
| `plot_noise_budget(coupler, n_bar)` | Stacked bar chart of Friis noise contributions per stage per chain, log scale |
| `plot_tint_summary(coupler, n_bar)` | Grouped bar chart of t_int for multiple SNR targets across all three chains |
| `plot_signal_accumulation(setup)` | Accumulated IQ signal μ_e(T) = +(δ/2)κT and μ_g(T) = −(δ/2)κT with ±1σ = √(κ N_sys T / 2) shaded bands; illustrates how the Gaussian clouds separate as √T grows slower than T |
| `plot_parameter_sweeps(setup)` | t_int vs E_J/E_C (left) and vs n̄ (right) for all three chains |
| `plot_charge_dispersion(E_C, E_J)` | Energy levels E₀, E₁, E₂ vs gate charge n_g |
| `plot_all(setup, save_path)` | Master figure combining S₂₁, charge dispersion, SNR accumulation, noise budget, and t_int summary |

---

## Output files from `run_simulation.py`

| File | Contents |
|---|---|
| `*_resonator_qubit.png` | \|S₂₁\| magnitude, \|S₂₁\| phase, charge dispersion |
| `*_measurement.png` | SNR accumulation, Friis noise budget, integration time summary |
| `*_signal_accumulation.png` | Separating Gaussian clouds for all three amplifier chains |
| `*_IQ_comparison.png` | IQ plane with noise ellipses at the target integration time |
| `*_sweeps.png` | t_int vs E_J/E_C and vs n̄ |

---

## Default parameters

| Parameter | Default | Description |
|---|---|---|
| `E_C` | 0.25 GHz | Charging energy |
| `E_J` | 15.0 GHz | Josephson energy (E_J/E_C = 60) |
| `ω_r` | 6.0 GHz | Resonator frequency |
| `κ_ext` | 2.0 MHz | External coupling rate |
| `κ_int` | 0.4 MHz | Internal loss rate |
| `g` | 80 MHz | Qubit-resonator coupling |
| `n̄` | 2.0 | Mean intra-cavity photons |
| JPA T_N | 0.05 K | N_add ≈ 1.1 at 6 GHz |
| HEMT T_N | 4.0 K | N_add ≈ 87 at 6 GHz |
| LNA T_N | 50 K | N_add ≈ 1092 at 6 GHz |

---

## Key references

- Koch et al., *Phys. Rev. A* **76**, 042319 (2007) — transmon qubit; dispersive shift formula
- Blais et al., *Phys. Rev. A* **69**, 062320 (2004) — circuit QED; dispersive Hamiltonian
- Krantz et al., *Appl. Phys. Rev.* **6**, 021318 (2019) — readout SNR model; amplifier chain; IQ detection
- Khalil et al., *J. Appl. Phys.* **111**, 054510 (2012) — hanger resonator S₂₁ formula
- Clerk et al., *Rev. Mod. Phys.* **82**, 1155 (2010) — quantum noise theory; Friis formula
- Caves, *Phys. Rev. D* **26**, 1817 (1982) — standard quantum limit for linear amplifiers
