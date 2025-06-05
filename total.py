import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from scipy.linalg import eigh

# === Choix des paramètres prédéfinis ou personnalisés ===
v0 = -200              # Profondeur du potentiel de la barrière
e = 3                  # Rapport E/V₀ (énergie/hauteur du potentiel)
xc = 0.6               # Position initiale du centre du paquet d'onde
sigma = 0.05           # Largeur (écart-type) du paquet d'onde initial
nt = 100000            # Nombre total de pas de temps (durée de la simulation)
bar_x0 = 0.8           # Position de début de la barrière de potentiel
bar_width = 0.1        # Largeur de la barrière de potentiel

print("Voici les paramètres par défaut :")
print(f"V₀={v0}, E/V₀={e}, x_c={xc}, σ={sigma}, nt={nt}, bar_x0={bar_x0}, bar_width={bar_width}")
print("Voulez-vous utiliser des paramètres prédéfinis ?")
use_defaults = input("Tapez 'o' pour oui, 'n' pour saisir manuellement : ").lower()

if use_defaults == 'o':
    print("Paramètres utilisés :")
    print(f"V₀={v0}, E/V₀={e}, x_c={xc}, σ={sigma}, nt={nt}, bar_x0={bar_x0}, bar_width={bar_width}")
else:
    v0 = float(input("Profondeur du potentiel V₀ (en unités d’énergie, ex: -4000): "))   # Profondeur du potentiel
    e = float(input("Rapport E/V₀ (sans unité, ex: 5): "))                               # Rapport énergie/potentiel
    xc = float(input("Position initiale du paquet x_c (en unités de longueur, ex: 0.6): ")) # Position initiale
    sigma = float(input("Largeur (écart-type) du paquet σ (en unités de longueur, ex: 0.05): ")) # Largeur du paquet
    nt = int(input("Durée de simulation (nombre de pas de temps, ex: 90000): "))         # Nombre de pas de temps
    bar_x0 = float(input("Position de début de la barrière (en unités de longueur, ex: 0.8): ")) # Début barrière
    bar_width = float(input("Largeur de la barrière (en unités de longueur, ex: 0.1): "))        # Largeur barrière
    print("Paramètres utilisés :")
    print(f"V₀={v0}, E/V₀={e}, x_c={xc}, σ={sigma}, nt={nt}, bar_x0={bar_x0}, bar_width={bar_width}")

# === Simulation du paquet d’onde ===
dt = 1E-7              # Pas de temps
dx = 0.001             # Pas d'espace (maillage spatial)
nx = int(1 / dx) * 2   # Nombre de points de discrétisation spatiale
nd = int(nt / 1000) + 1 # Nombre d'images à sauvegarder pour l'animation
s = dt / dx**2         # Constante de stabilité du schéma numérique
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi))) # Facteur de normalisation du paquet d'onde
E = e * v0             # Énergie du paquet d'onde
k = math.sqrt(2 * abs(E)) # Nombre d’onde du paquet (lié à l'énergie)

o = np.linspace(0, (nx - 1) * dx, nx) # Tableau des positions spatiales
V = np.zeros(nx)                      # Tableau du potentiel (0 partout sauf barrière)
V[(o >= bar_x0) & (o <= bar_x0 + bar_width)] = v0 # Potentiel de la barrière

cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * sigma**2)) # Paquet d’onde initial (complexe)
densite = np.zeros((nt, nx))        # Tableau pour stocker la densité de probabilité à chaque pas de temps
densite[0, :] = np.abs(cpt) ** 2    # Densité initiale
final_densite = np.zeros((nd, nx))  # Tableau pour stocker la densité pour l’animation (échantillonnée)

re = np.real(cpt)                   # Partie réelle du paquet d’onde
im = np.imag(cpt)                   # Partie imaginaire du paquet d’onde
b = np.zeros(nx)                    # Tableau temporaire pour la mise à jour de la densité

it = 0                              # Compteur pour l’index dans final_densite
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]          # Copie la partie imaginaire pour calculer la densité
        im[1:-1] += s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt) # Mise à jour imaginaire
        densite[i, 1:-1] = re[1:-1]**2 + b[1:-1] * im[1:-1] # Calcul de la densité
    else:
        re[1:-1] -= s * (im[2:] + im[:-2]) - 2 * im[1:-1] * (s + V[1:-1] * dt) # Mise à jour réelle

    if (i - 1) % 1000 == 0:         # Toutes les 1000 itérations
        final_densite[it, :] = densite[i, :] # Stocke la densité pour l'animation
        it += 1

# === Animation matplotlib ===
def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j, :])
    return line,

plot_title = f"Propagation du paquet d'onde (E/V₀ = {e})" # Titre du graphique
fig = plt.figure()                   # Création de la figure
line, = plt.plot([], [])             # Ligne vide pour l'animation
plt.ylim(0, np.max(final_densite) * 1.1) # Limite verticale
plt.xlim(0, max(o))                  # Limite horizontale
plt.plot(o, (V * np.max(final_densite) / abs(v0)), label="Potentiel (échelle adaptée)") # Affiche le potentiel
plt.title(plot_title)
plt.xlabel("Position x (unité arbitraire)")
plt.ylabel("Densité de probabilité |ψ(x)|²")
plt.legend()

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, blit=False, interval=100, repeat=False) # Animation
plt.show()

# Sauvegarde de l'animation au format MP4
ani.save("propagation_paquet_onde.mp4", writer="ffmpeg")
print("Animation sauvegardée sous 'propagation_paquet_onde.mp4'.")

# === Demander à l'utilisateur s'il souhaite voir les états stationnaires ===
switch = input("\nVoulez-vous afficher les états stationnaires du puits de potentiel ? (o/n) : ").lower()
if switch == 'o':

    # === Paramètres adaptés pour le second code ===
    L = 1.0                          # Longueur totale de la boîte
    N = 1000                         # Nombre de points pour la discrétisation spatiale
    dx = L / (N - 1)                 # Pas d'espace pour cette partie
    x = np.linspace(0, L, N)         # Tableau des positions spatiales

    V0 = v0                          # Profondeur du potentiel (identique à v0)
    a = bar_width                    # Largeur de la barrière
    x0 = bar_x0 + bar_width / 2      # Centre de la barrière
    n_states = int(input("Nombre d'états stationnaires à afficher (ex: 5): ")) # Nombre d’états stationnaires

    V_stationnaire = np.zeros(N)     # Potentiel pour le calcul des états stationnaires
    V_stationnaire[np.abs(x - x0) <= a / 2] = V0 # Potentiel de la barrière

    hbar2_2m = 1                     # Constante (fixée à 1 pour simplifier)
    diag = np.full(N, -2.0)          # Diagonale principale du Laplacien
    offdiag = np.full(N - 1, 1.0)    # Diagonale secondaire du Laplacien
    laplacian = (np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)) / dx**2 # Matrice Laplacienne
    H = -hbar2_2m * laplacian + np.diag(V_stationnaire) # Hamiltonien

    eigvals, eigvecs = eigh(H)       # Valeurs propres (énergies) et vecteurs propres (états stationnaires)

    # === Affichage des états stationnaires ===
    plt.figure(figsize=(10, 6))
    for n in range(n_states):
        psi_n = eigvecs[:, n]        # n-ième état stationnaire
        psi_n /= np.sqrt(np.trapz(psi_n**2, x)) # Normalisation
        plt.plot(x, psi_n**2 + eigvals[n], label=f"État {n}, E = {eigvals[n]:.2f}") # Affichage

    plt.plot(x, V_stationnaire, color='black', linestyle='--', label="Potentiel V(x)") # Potentiel

    # Calcul des bornes dynamiques pour l'axe x afin d'inclure toute la barrière
    bar_left = x0 - a / 2            # Bord gauche de la barrière
    bar_right = x0 + a / 2           # Bord droit de la barrière
    marge = 0.1 * L                  # 10% de marge de chaque côté

    x_min = max(0, bar_left - marge) # Limite gauche de l'affichage
    x_max = min(L, bar_right + marge)# Limite droite de l'affichage
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
    # Sauvegarde de la figure des états stationnaires
    plt.savefig("etats_stationnaires.png")
    print("Figure des états stationnaires sauvegardée sous 'etats_stationnaires.png'.")