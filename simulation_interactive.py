import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_user_inputs():
    def ask(prompt, default):
        val = input(f"{prompt} [default={default}]: ")
        return float(val) if val.strip() != '' else default

    v0 = ask("Entrez la hauteur de la barrière de potentiel (v0)", -4000)
    e = ask("Entrez le rapport E/V0 (e)", 5)
    xc = ask("Entrez la position initiale du paquet d'onde (xc)", 0.6)
    sigma = ask("Entrez l'écart-type du paquet d'onde (sigma)", 0.05)
    debut_barriere = ask("Entrez la position de début de la barrière", 0.8)
    largeur_barriere = ask("Entrez la largeur de la barrière", 0.1)
    return v0, e, xc, sigma, debut_barriere, largeur_barriere

def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j,:])
    return line,

def main():
    global o, line, final_densite

    v0, e, xc, sigma, debut_barriere, largeur_barriere = get_user_inputs()
    fin_barriere = debut_barriere + largeur_barriere

    dt = 1E-7
    dx = 0.001
    nx = int(1/dx)*2
    nt = 90000
    nd = int(nt/1000) + 1
    s = dt / (dx ** 2)

    E = e * v0
    k = math.sqrt(2 * abs(E))
    A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))

    o = np.linspace(0, (nx - 1) * dx, nx)
    V = np.zeros(nx)
    V[(o >= debut_barriere) & (o <= fin_barriere)] = v0

    cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))

    densite = np.zeros((nt, nx))
    densite[0,:] = np.abs(cpt[:]) ** 2
    final_densite = np.zeros((nd, nx))

    re = np.real(cpt[:])
    im = np.imag(cpt[:])
    b = np.zeros(nx)

    it = 0
    for i in range(1, nt):
        if i % 2 != 0:
            b[1:-1] = im[1:-1]
            im[1:-1] += s * (re[2:] + re[:-2] - 2 * re[1:-1]) - 2 * re[1:-1] * V[1:-1] * dt
            densite[i, 1:-1] = re[1:-1]**2 + im[1:-1]*b[1:-1]
        else:
            re[1:-1] -= s * (im[2:] + im[:-2] - 2 * im[1:-1]) - 2 * im[1:-1] * V[1:-1] * dt

    for i in range(1, nt):
        if (i - 1) % 1000 == 0:
            it += 1
            final_densite[it][:] = densite[i][:]

    plot_title = f"Barrière avec E/V0 = {e}"
    fig = plt.figure()
    line, = plt.plot([], [], label="Densité")
    plt.ylim(0, np.max(final_densite) * 1.1)
    plt.xlim(0, 2)
    plt.plot(o, V, label="Potentiel")
    plt.title(plot_title)
    plt.xlabel("x")
    plt.ylabel("Densité de probabilité de présence")
    plt.legend()

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, blit=False, interval=100, repeat=False)

    save_option = input("Souhaitez-vous sauvegarder l'animation en vidéo ? (o/n): ").strip().lower()
    if save_option == 'o':
        output_file = input("Nom du fichier de sortie (ex: animation.mp4): ").strip()
        if output_file == "":
            output_file = "simulation.mp4"
        ani.save(output_file, writer='ffmpeg', fps=10)
        print(f"Animation sauvegardée dans le fichier : {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
