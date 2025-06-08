import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import shutil
from scipy.linalg import eigh
from tqdm import tqdm  # Pour afficher une barre de progression (pip install tqdm)

# === 1. Choix des paramètres ===
def get_parameters():
    """Demande à l'utilisateur les paramètres de simulation, avec ou sans valeurs par défaut."""
    print("Voulez-vous utiliser des paramètres prédéfinis ?")
    use_defaults = input("Tapez 'o' pour oui, 'n' pour saisir manuellement : ").lower()

    if use_defaults == 'o':
        return {
            "v0": -4000,
            "e": 5,
            "xc": 0.6,
            "sigma": 0.05,
            "nt": 90000,
            "bar_x0": 0.8,
            "bar_width": 0.1,
            "dt": 1E-7,
            "dx": 0.001,
            "L": 2.0,
        }
    else:
        return {
            "v0": float(input("Profondeur du potentiel V₀ (ex: -4000): ") or -4000),
            "e": float(input("Rapport E/V₀ (ex: 5): ") or 5),
            "xc": float(input("Position initiale du paquet x_c (ex: 0.6): ") or 0.6),
            "sigma": float(input("Largeur du paquet σ (ex: 0.05): ") or 0.05),
            "nt": int(input("Nombre de pas de temps nt (ex: 90000): ") or 90000),
            "bar_x0": float(input("Position de début de la barrière (ex: 0.8): ") or 0.8),
            "bar_width": float(input("Largeur de la barrière (ex: 0.1): ") or 0.1),
            "dt": float(input("Pas de temps dt (ex: 1e-7): ") or 1E-7),
            "dx": float(input("Pas d'espace dx (ex: 0.001): ") or 0.001),
            "L": float(input("Longueur totale L (ex: 2.0): ") or 2.0),
        }

def run_simulation(params):
    """Lance la simulation de la propagation du paquet d'onde."""

    # === 2. Initialisation des paramètres ===
    v0, e, xc, sigma = params['v0'], params['e'], params['xc'], params['sigma']
    nt, bar_x0, bar_width = params['nt'], params['bar_x0'], params['bar_width']
    dt, dx, L = params['dt'], params['dx'], params['L']

    nx = int(L / dx)
    x = np.linspace(0, (nx - 1) * dx, nx)
    s = dt / dx**2
    E = e * v0  # énergie totale
    k = math.sqrt(2 * abs(E))
    A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))  # Normalisation gaussienne

    # === 3. Potentiel et état initial ===
    V = np.zeros(nx)
    V[(x >= bar_x0) & (x <= bar_x0 + bar_width)] = v0
    psi = A * np.exp(1j * k * x - ((x - xc) ** 2) / (2 * sigma ** 2))
    re, im = np.real(psi), np.imag(psi)
    b = np.zeros(nx)

    nd = nt // 1000 + 1
    final_densite = np.zeros((nd, nx))
    final_densite[0, :] = np.abs(psi) ** 2
    it = 0

    # === 4. Boucle temporelle ===
    for i in tqdm(range(1, nt), desc="Simulation temps réel"):
        if i % 2 != 0:
            b[1:-1] = im[1:-1]
            im[1:-1] += s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        else:
            re[1:-1] -= s * (im[2:] + im[:-2]) - 2 * im[1:-1] * (s + V[1:-1] * dt)

        if i % 1000 == 0 and it + 1 < nd:
            it += 1
            final_densite[it, :] = re**2 + im**2

    return x, V, final_densite, nd

def create_animation(x, V, final_densite, nd, params):
    """Crée une animation de la densité de probabilité."""
    fig = plt.figure()
    line, = plt.plot([], [], label="Densité de probabilité")

    plt.xlim(0, params['L'])
    plt.ylim(0, np.max(final_densite) * 1.1)
    plt.plot(x, V * np.max(final_densite) / abs(params['v0']), label="Potentiel (échelle adaptée)")
    plt.vlines([params['bar_x0'], params['bar_x0'] + params['bar_width']], 0, np.max(final_densite), colors='red', linestyles='dashed')
    plt.xlabel("Position x")
    plt.ylabel("Densité |ψ(x)|²")
    plt.title(f"Propagation du paquet d'onde (E/V₀ = {params['e']})")
    plt.legend()

    def init():
        line.set_data([], [])
        return line,

    def animate(j):
        line.set_data(x, final_densite[j])
        return line,

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, interval=100, blit=True)
    plt.show()

    filename = f"propagation_paquet_onde.gif"
    ani.save(filename, writer="pillow")
    print(f"Animation sauvegardée sous '{filename}'.")

def solve_stationary_states(params, x):
    """Calcule et affiche les états stationnaires dans le potentiel."""
    switch = input("\nAfficher les états stationnaires ? (o/n) : ").lower()
    if switch != 'o':
        return

    N = len(x)
    dx = x[1] - x[0]
    x0 = params['bar_x0'] + params['bar_width'] / 2

    V_stat = np.zeros(N)
    V_stat[np.abs(x - x0) <= params['bar_width'] / 2] = params['v0']

    laplacian = (
        np.diag(np.full(N, -2.0)) +
        np.diag(np.ones(N - 1), 1) +
        np.diag(np.ones(N - 1), -1)
    ) / dx**2

    H = -laplacian + np.diag(V_stat)
    eigvals, eigvecs = eigh(H)

    n_states = int(input("Nombre d'états à afficher (ex: 5) : ")or 5)
    plt.figure(figsize=(10, 6))
    for n in range(n_states):
        psi_n = eigvecs[:, n]
        psi_n /= np.sqrt(np.trapezoid(np.abs(psi_n)**2, x))
        plt.plot(x, np.abs(psi_n)**2 + eigvals[n], label=f"État {n}, E={eigvals[n]:.2f}")

    plt.plot(x, V_stat, 'k--', label="Potentiel V(x)")
    plt.vlines([x0 - params['bar_width']/2, x0 + params['bar_width']/2], 0, np.max(eigvals[:n_states])*1.2, colors='red', linestyles='dashed')
    plt.xlabel("Position x")
    plt.ylabel("ψ²(x) + E")
    plt.title("États stationnaires dans un puits de potentiel fini")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"etats_stationnaires.png")
    plt.show()

# === Exécution principale ===
if __name__ == "__main__":
    params = get_parameters()
    print("\nParamètres utilisés :")
    for key, value in params.items():
        print(f"{key} = {value}")

    x, V, final_densite, nd = run_simulation(params)
    create_animation(x, V, final_densite, nd, params)
    solve_stationary_states(params, x)