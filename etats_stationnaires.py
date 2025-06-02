import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # Pour diagonaliser matrice symétrique

# === Paramètres physiques et numériques ===
L = 1.0                # Longueur de la boîte (en unités arbitraires)
N = 1000               # Nombre de points d'espace
dx = L / (N - 1)       # Pas d'espace
x = np.linspace(0, L, N)

# === Entrée utilisateur avec unités ===
V0 = float(input("Profondeur du puits V0 (en unités d'énergie, ex: -4000): "))
a = float(input("Largeur du puits a (en unités de longueur, ex: 0.2): "))
x0 = float(input("Position du centre du puits x0 (en unités de longueur, ex: 0.5): "))
n_states = int(input("Nombre d'états stationnaires à afficher (ex: 5): "))

# === Potentiel : puits carré ===
V = np.zeros(N)
V[np.abs(x - x0) <= a/2] = V0  # Potentiel dans le puits

# === Construction du Hamiltonien H = T + V ===
hbar2_2m = 1  # Unités réduites : ℏ² / 2m = 1
diag = np.full(N, -2.0)
offdiag = np.full(N - 1, 1.0)
laplacian = (np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)) / dx**2
H = -hbar2_2m * laplacian + np.diag(V)

# === Diagonalisation ===
eigvals, eigvecs = eigh(H)

# === Affichage des états stationnaires ===
plt.figure(figsize=(10, 6))
for n in range(n_states):
    psi_n = eigvecs[:, n]
    psi_n /= np.sqrt(np.trapz(psi_n**2, x))  # Normalisation
    plt.plot(x, psi_n**2 + eigvals[n], label=f"État {n}, E = {eigvals[n]:.2f} unités")

plt.plot(x, V, color='black', linestyle='--', label="Potentiel V(x)")
plt.xlabel("Position x (unité arbitraire)")
plt.ylabel("ψ²(x) + E (unité d’énergie)")
plt.title("États stationnaires dans un puits de potentiel fini")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
