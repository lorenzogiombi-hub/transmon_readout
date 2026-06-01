import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Costomization of the plot ---
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

font_size = 20
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams.update({'lines.linewidth': 1.5})
mpl.rcParams.update({'axes.linewidth': 1.})
mpl.rcParams.update({'axes.labelsize': font_size+1})
mpl.rcParams.update({'xtick.labelsize': font_size})
mpl.rcParams.update({'ytick.labelsize': font_size})
# but make legend smaller
mpl.rcParams.update({'legend.fontsize': 16})

def chi(g, alpha, omega_r, omega_q):
    """
    Compute the dispersive shift chi using the transmon-accurate formula:
chi = - (g^2 * alpha) / (delta_qr * (delta_qr + alpha))
where g is the coupling strength (assumed constant for this plot), alpha is the anharmonicity, and delta_qr is the qubit-resonator detuning.    

return chi
    """
    delta_qr = omega_q - omega_r
    return - (g**2 * alpha) / (delta_qr * (delta_qr + alpha))  


if __name__ == "__main__":
    # Example parameters
    g = 0.8 # coupling strength in GHz
    omega_r = 6.0  # resonator frequency in GHz
    E_C = 0.3  # charging energy in GHz
    ratios = np.linspace(10, 100, 100)  # E_J/E_C ratios to sweep

    chi_values = []
    for r in ratios:
        E_J = r * E_C
        alpha = -E_C * (1 - (E_C / (E_J + E_C)))  # approximate anharmonicity
        omega_q = np.sqrt(8 * E_J * E_C) - E_C  # approximate qubit frequency
        chi_val = chi(g, alpha, omega_r, omega_q)
        chi_values.append(chi_val)
        print(f"E_J/E_C: {r:.1f}, alpha: {alpha:.3f} GHz, omega_q: {omega_q:.3f} GHz, chi: {chi_val*1e3:.2f} MHz")

    fig, ax = plt.subplots()
    ax.plot(ratios, np.array(chi_values)*1e3, marker='o')
    ax.set_xlabel(r'$E_J / E_C$')
    ax.set_ylabel(r'Dispersive Shift $\chi$ (MHz)')
    ax.set_title(r'Dispersive Shift $\chi$ vs $E_J/E_C$')
    ax.grid(True)
    fig.tight_layout()
    plt.show()