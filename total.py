import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from scipy.linalg import eigh

# === Choix des paramètres prédéfinis ou personnalisés ===
print("Voulez-vous utiliser des paramètres prédéfinis ?")
use_defaults = input("Tapez 'o' pour oui, 'n' pour saisir manuellement : ").lower()

if use_defaults == 'o':
    v0 = -4000
    e = 5
    xc = 0.6
    sigma = 0.05
    nt = 90000
    bar_x0 = 0.8
    bar_width = 0.1
    print("Paramètres utilisés :")
    print(f"V₀={v0}, E/V₀={e}, x_c={xc}, σ={sigma}, nt={nt}, bar_x0={bar_x0}, bar_width={bar_width}")
else:
    v0 = float(input("Profondeur du potentiel V₀ (en unités d’énergie, ex: -4000): "))
    e = float(input("Rapport E/V₀ (sans unité, ex: 5): "))
    xc = float(input("Position initiale du paquet x_c (en unités de longueur, ex: 0.6): "))
    sigma = float(input("Largeur (écart-type) du paquet σ (en unités de longueur, ex: 0.05): "))
    nt = int(input("Durée de simulation (nombre de pas de temps, ex: 90000): "))
    bar_x0 = float(input("Position de début de la barrière (en unités de longueur, ex: 0.8): "))
    bar_width = float(input("Largeur de la barrière (en unités de longueur, ex: 0.1): "))

# === Simulation du paquet d’onde ===
dt = 1E-7
dx = 0.001
nx = int(1 / dx) * 2
nd = int(nt / 1000) + 1
s = dt / dx**2
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
E = e * v0
k = math.sqrt(2 * abs(E))

o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= bar_x0) & (o <= bar_x0 + bar_width)] = v0

cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * sigma**2))
densite = np.zeros((nt, nx))
densite[0, :] = np.abs(cpt) ** 2
final_densite = np.zeros((nd, nx))

re = np.real(cpt)
im = np.imag(cpt)
b = np.zeros(nx)

it = 0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]
        im[1:-1] += s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i, 1:-1] = re[1:-1]**2 + b[1:-1] * im[1:-1]
    else:
        re[1:-1] -= s * (im[2:] + im[:-2]) - 2 * im[1:-1] * (s + V[1:-1] * dt)

    if (i - 1) % 1000 == 0:
        final_densite[it, :] = densite[i, :]
        it += 1

# === Animation matplotlib ===
def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j, :])
    return line,

plot_title = f"Propagation du paquet d'onde (E/V₀ = {e})"
fig = plt.figure()
line, = plt.plot([], [])
plt.ylim(0, np.max(final_densite) * 1.1)
plt.xlim(0, max(o))
plt.plot(o, (V * np.max(final_densite) / abs(v0)), label="Potentiel (échelle adaptée)")
plt.title(plot_title)
plt.xlabel("Position x (unité arbitraire)")
plt.ylabel("Densité de probabilité |ψ(x)|²")
plt.legend()

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, blit=False, interval=100, repeat=False)
plt.show()

# === Demander à l'utilisateur s'il souhaite voir les états stationnaires ===
switch = input("\nVoulez-vous afficher les états stationnaires du puits de potentiel ? (o/n) : ").lower()
if switch == 'o':

    # === Paramètres adaptés pour le second code ===
    L = 1.0
    N = 1000
    dx = L / (N - 1)
    x = np.linspace(0, L, N)

    V0 = v0
    a = bar_width
    x0 = bar_x0 + bar_width / 2
    n_states = int(input("Nombre d'états stationnaires à afficher (ex: 5): "))

    V_stationnaire = np.zeros(N)
    V_stationnaire[np.abs(x - x0) <= a / 2] = V0

    hbar2_2m = 1
    diag = np.full(N, -2.0)
    offdiag = np.full(N - 1, 1.0)
    laplacian = (np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)) / dx**2
    H = -hbar2_2m * laplacian + np.diag(V_stationnaire)

    eigvals, eigvecs = eigh(H)

    # === Affichage des états stationnaires ===
    plt.figure(figsize=(10, 6))
    for n in range(n_states):
        psi_n = eigvecs[:, n]
        psi_n /= np.sqrt(np.trapz(psi_n**2, x))
        plt.plot(x, psi_n**2 + eigvals[n], label=f"État {n}, E = {eigvals[n]:.2f}")

    plt.plot(x, V_stationnaire, color='black', linestyle='--', label="Potentiel V(x)")

    # Calcul des bornes dynamiques pour l'axe x afin d'inclure toute la barrière
    bar_left = x0 - a / 2
    bar_right = x0 + a / 2
    marge = 0.1 * L  # 10% de marge de chaque côté

    x_min = max(0, bar_left - marge)
    x_max = min(L, bar_right + marge)
    # Si la barrière est trop proche du bord, on garde au moins [0, L]
    if x_min >= x_max or (bar_left < 0 and bar_right > L):
        x_min, x_max = 0, L

    plt.xlim(x_min, x_max)

    plt.xlabel("Position x (unité arbitraire)")
    plt.ylabel("ψ²(x) + E (unité d’énergie)")
    plt.title("États stationnaires dans un puits de potentiel fini")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()