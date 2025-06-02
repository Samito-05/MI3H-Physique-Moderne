import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# === Entrée utilisateur avec unités ===
v0 = float(input("Profondeur du potentiel V₀ (en unités d’énergie, ex: -4000): "))
e = float(input("Rapport E/V₀ (sans unité, ex: 5): "))
xc = float(input("Position initiale du paquet x_c (en unités de longueur, ex: 0.6): "))
sigma = float(input("Largeur (écart-type) du paquet σ (en unités de longueur, ex: 0.05): "))
nt = int(input("Durée de simulation (nombre de pas de temps, ex: 90000): "))
bar_x0 = float(input("Position de début de la barrière (en unités de longueur, ex: 0.8): "))
bar_width = float(input("Largeur de la barrière (en unités de longueur, ex: 0.1): "))


dt = 1E-7                         # Pas de temps (en unité arbitraire)
dx = 0.001                        # Pas d’espace (en unité arbitraire)
nx = int(1 / dx) * 2              # Nombre de points d’espace
nd = int(nt / 1000) + 1           # Nombre d'images de l’animation
s = dt / dx**2
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
E = e * v0                        # Énergie totale
k = math.sqrt(2 * abs(E))        # Nombre d’onde


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