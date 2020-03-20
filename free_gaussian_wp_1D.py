# free Gaussian wave packet time evolution
# to run, type "python3 free_guassian_wp_1D.py" in command line
# (after installing Python 3 with required libraries and downloading the file!)
# original Python 3 code by BA2 student Corentin Simon (2017-2018), to be improved!
# animation implemented by JF Delplace (2019-2020)
# Jean-Marc Sparenberg

from math import pi
import numpy as np  # for numerical calculations

import matplotlib.pyplot as plt  # for plotting
import matplotlib.cm as cm  # color maps for complex numbers
from matplotlib.widgets import Slider, Button  # interactive slider and button in graph
from matplotlib.animation import FuncAnimation  # to animate the slider

# Paramètres physiques

k0 = 6  # nombre d'onde central du packet gaussien
a = 1  # facteur de distribution de la gaussienne
alpha = 1  # alpha = hbar/m => v = alpha*k

t_min = 0
t_max = 10

srx = 50  # range de la position en x
x = np.linspace(-srx / 3, 2 * srx / 3, srx * 20)  # absisse de la simulation
x2 = np.linspace(0, 20, 1000)  # absisse de l'espace des fréquences

# Variables UI : mémorise l'état de l'affichage pour le fonctionnement des boutons

is_color = False
play = True
animation = None
animation_initialized = False

# Setup images boutons

ICON_PLAY_forward = plt.imread("buttons/play_button_forward_icon.png")
ICON_PLAY_back = plt.imread("buttons/play_button_back_icon.png")
ICON_PLAY = ICON_PLAY_forward
ICON_PAUSE = plt.imread("buttons/pause_button_icon.png")
ICON_MULTICOLOR = plt.imread("buttons/color_wheel.png")
ICON_TRICOLOR = plt.imread("buttons/waves.png")

# Setup Fenetre

fig = plt.figure(figsize = (15, 10))
fig.canvas.set_window_title('Free Gaussian Wave Packet - Time Evolution')

ax1 = plt.subplot2grid((2, 3), (0, 0), colspan = 2)
ax2 = plt.subplot2grid((2, 3), (1, 0), colspan = 2)

# Setup Sliders

label_size = 15  # taille des labels des sliders
x_pos, y_pos, x_span, y_span, y_spacing = 0.69, 0.85, 0.25, 0.02, 0.08  # param de position
v_max = 15  # slider du temps

ax_k = plt.axes([x_pos, y_pos, x_span, y_span])  # slider de k0
k_slider = Slider(ax_k, '$k_0$', 1, 15, valinit = k0)
k_slider.label.set_size(label_size)

ax_a = plt.axes([x_pos, y_pos - y_spacing, x_span, y_span])  # slider de a
a_slider = Slider(ax_a, '$a$', 0.01, 5, valinit = a)
a_slider.label.set_size(label_size)

ax_time = plt.axes([x_pos, y_pos - y_spacing * 2, x_span, y_span])  # slider du temps
time_slider = Slider(ax_time, '$t$', t_min, t_max, valinit = 0)
time_slider.label.set_size(label_size)

ax_speed = plt.axes([x_pos, y_pos - y_spacing * 8, x_span, y_span])  # slider vitesse de l'animation
speed_slider = Slider(ax_speed, '', v_max * -1, v_max, valinit = 0)
plt.gcf().text(x_pos, y_pos - y_spacing * 7.5, "Vitesse de l'animation", fontsize = label_size)  # label au-dessus du slider

# Setup Boutons

rax1 = plt.axes([x_pos, y_pos - y_spacing * 6, x_span, y_span * 3])  # bouton de control de la couleur
rax2 = plt.axes([x_pos, y_pos - y_spacing * 7, x_span, y_span * 3])  # bouton de control du play

colorButton = Button(rax1, '', image = ICON_MULTICOLOR, color = '1', hovercolor = 'lightgrey')
playButton = Button(rax2, '', image = ICON_PLAY, color = '1', hovercolor = 'lightgrey')

# Calcul de la fonction d'onde en fonction du temps

def psi(t):
    global x, k0, alpha, a

    v = alpha * k0
    w = (a ** 4 + ((alpha * t) ** 2) / 4)
    w_complex = a ** 2 + 1j * alpha * t / 2
    omega = (k0 ** 2) * alpha / 2
    prob_compl = np.exp(1j * (k0 * x - t * omega)) * ((pi / w_complex) ** 0.5) * np.exp(
        -((x - v * t) ** 2) / (4 * w_complex))

    return prob_compl


# Functions refresh

def update_phase(val_):
    global a
    a = a_slider.val
    y = np.exp(- (a ** 2) * (k0 - x2) ** 2)  # distribution des fréquences
    ax1.clear()
    ax1.set_title('Transformée de Fourier')
    ax1.plot(x2, y)

    fig.canvas.draw_idle()
    update_temps(0)


def update_temps(val_):  # dessin de la fonction d'onde

    probCompl = psi(time_slider.val)
    ax2.clear()
    ax2.set_xlim([-srx / 3, 2 * srx / 3])
    ax2.set_ylim([-4, 4])
    ax2.set_title('Fonction d\'onde')

    if is_color:  # affichage en couleur
        X = np.array([x, x])
        y0 = np.zeros(len(x))
        y = [abs(i) for i in probCompl]
        Y = np.array([y0, y])
        Z = np.array([probCompl, probCompl])
        C = np.angle(Z)
        ax2.pcolormesh(X, Y, C, cmap = cm.hsv, vmin = -np.pi, vmax = np.pi)
        ax2.plot(x, np.abs(probCompl), label = '$|\psi|$', color = 'black')

    else:  # affichage des parties reels et complexe
        ax2.plot(x, np.real(probCompl), label = '$\operatorname{Re}(\psi)$')
        ax2.plot(x, np.imag(probCompl), label = '$\operatorname{Im}(\psi)$')
        ax2.plot(x, np.absolute(probCompl) ** 2, label = '$\psi \psi^{\dag} $')

    ax2.legend(fontsize = 15)
    fig.canvas.draw_idle()


def update_k(val_):  # lorsqu'on change k0
    global k0
    k0 = k_slider.val
    update_phase(0)


def update_speed(val_):  # pour la modification du bouton play en fonction du signe de la vitesse d'animation
    global ICON_PLAY, play
    if speed_slider.val < 0:
        ICON_PLAY = ICON_PLAY_back
    else:
        ICON_PLAY = ICON_PLAY_forward
    if not play or not animation_initialized:
        rax2.images[0].set_data(ICON_PLAY)
    update_temps(0)


# Fonction click boutons

def on_check_color(event):  # bouton couleur pressé
    global is_color
    is_color = not is_color # change le param
    if is_color:  # change l'icone du bouton
        rax1.images[0].set_data(ICON_TRICOLOR)
    else:
        rax1.images[0].set_data(ICON_MULTICOLOR)
    update_phase(0)


def on_check_playtime(event):  # bouton play pressé
    global play, animation, animation_initialized
    if speed_slider.val == 0:  # si bouton play pressé et que vitesse d'animation est nulle --> set à 1 par défaut
        speed_slider.set_val(1)
    if not animation_initialized:  # si l'animation n'est pas encore initialisée
        animation_initialized = True
        play = True
        create_animation()
    else:  # play/pause animation
        play = not play
        if play:
            start_animation()
        else:
            stop_animation()


# Fonctions pour l'animation

def create_animation():  # attention, animation lancée par défaut à son initialisation
    global animation
    rax2.images[0].set_data(ICON_PAUSE)  # changement de l'icone du bouton play
    animation = FuncAnimation(fig, func = animation_update, frames = np.arange(0, 100, 0.1), interval = 40,
                              blit = False, repeat = True)


def start_animation():
    global animation, rax2
    rax2.images[0].set_data(ICON_PAUSE)  # changement de l'icone du bouton play
    animation.event_source.start()  # start animation


def stop_animation():
    global animation, rax2
    rax2.images[0].set_data(ICON_PLAY)  # changement de l'icone du bouton play
    animation.event_source.stop()  # stop animation


def animation_update(i):
    # Change la valeur du slider temps à chaque appel depuis l'objet FuncAnimation
    speed = speed_slider.val  # récupère la valeur de vitesse d'animation
    time = time_slider.val
    next_time = time + (speed / 100)
    if next_time <= t_min:  # recommence l'animation si elle atteint les limites temporelles fixées
        next_time = t_max
    elif next_time >= t_max:
        next_time = t_min
    time_slider.set_val(next_time)


# creation de la première frame
update_phase(0)

# association des fonctions aux sliders
time_slider.on_changed(update_temps)
a_slider.on_changed(update_phase)
k_slider.on_changed(update_k)
speed_slider.on_changed(update_speed)

# association des fonctions aux boutons
colorButton.on_clicked(on_check_color)
playButton.on_clicked(on_check_playtime)

# affichage de l'interface
plt.show()
