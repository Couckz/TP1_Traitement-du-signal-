import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
plt.close("all") #Permet d'effacer toutes les anciennes courbes faites avant (obtenir une meilleur vision)
# Partie 1 : Convolution et Transformée de Fourier

# 1/
# Initialisation
fe = 1024  # Fréquence d'échantillonnage en Hz
D = 1    # Durée du signal en s
t = np.arange(-D/2,D/2,1/fe)  # Création de l'axe temporel
T = 25/fe    # Largeur de la porte en nb d'échantillons

# Création du signal s
s = np.ones_like(t)
s[t > T] = 0
s[t < -T] = 0

# Affichage du signal
plt.figure(1)
plt.plot(t, s)
plt.title('Signal s')
plt.xlabel('temps (s)')
plt.ylabel('amplitude')
plt.show()


# 2/
f = np.fft.fftfreq(len(s), 1/fe)  # Création de l'axe des fréquences
f =  np.fft.fftshift(f)  # Déplacer les fréquences de manière centrée
S = np.fft.fft(s)  # TF du signal s
module = np.abs(S) # Calcul du module de la FFT
phase = np.angle(S) # Calcul de la phase de la FFT
# Affichage

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(f, module)
plt.xlabel('fréquences (Hz)')
plt.ylabel('spectre de s')
plt.title('Module et phase de la TF de s')


plt.subplot(2, 1, 2)
plt.plot(f, phase)
plt.xlabel('fréquences (Hz)')
plt.ylabel('phase')
plt.show()


# 3/
Y = np.fft.fftshift(S)  # Permutation avec fftshift
module = np.abs(Y)
phase = np.angle(Y)
plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(f, module)
plt.xlabel('fréquences (Hz)')
plt.ylabel('spectre de y')
plt.title('Module et phase de Y')

plt.subplot(2, 1, 2)
plt.plot(f, phase)
plt.xlabel('fréquences (Hz)')
plt.ylabel('phase')
plt.show()


# 4/
y_prime = np.fft.ifft(Y)  # TF inverse de Y
s_prime = np.fft.ifft(S)  # TF inverse de S
plt.figure(4)
plt.subplot(1, 2, 1)
plt.plot(t, y_prime)
plt.xlabel('temps s en seconde')
plt.ylabel('amplitude')
plt.title("Fft inverse de Y")


plt.subplot(1, 2, 2)
plt.plot(t, s_prime)
plt.xlabel("temps en seconde")
plt.ylabel("amplitude")
plt.title("fft inverse de S")
plt.show()


# 5/
x = np.convolve(s, s, 'same')  # Produit de convolution
plt.figure(5)
plt.plot(t, x)
plt.xlabel("temps en seconde")
plt.ylabel("amplitude")
plt.title("Produit de convolution x (s*s)")
plt.show()


# 6/
X = np.fft.fft(x)  # TF de x
S_carre = S**2  # Élévation au carré de S

plt.figure(6)
plt.plot(f, X, label='X')
plt.plot(f, S_carre, '--', label='S^2')
plt.xlabel("frequences en Hz")
plt.ylabel("Amplitudes")
plt.title("FFT de X et de S au carré")
plt.legend()
plt.show()


# Partie 2 : TF d'un signal carré

# 1/
# Initialisation
fe = 1000  # Fréquence d'échantillonnage en Hz
F = 100    # Fréquence du signal en Hz
N = 10     # Nombre de périodes
D = (1*N)/F  # Durée du signal en s
t = np.arange(0,D,1/fe) # Création de l'axe temporel

s = square(2*np.pi*F*t) # Signal carré
plt.figure(7)
plt.plot(t, s)
plt.xlabel("temps en seconde")
plt.ylabel("amplitude")
plt.title("Signal carré sur 10 périodes")
plt.show()

# 2/
nb_points_fft = 1024  # Nombre de points de la FFT
f = np.fft.fftfreq(nb_points_fft, 1/fe)  # Création de l'axe des fréquences
f =  np.fft.fftshift(f)  # Déplacer les fréquences de manière centrée

S = np.fft.fft(s, nb_points_fft)  # FFT de s
module_S = np.abs(S)
plt.figure(8)
plt.plot(f, module_S)
plt.xlabel("fréquence en Hz")
plt.ylabel("Spectre de s")
plt.title("Module de S")
plt.show()


# Partie 3 : Débruitage d'un signal par FFT

# TF et TF inverse
# Initialisation
fe = 1000  # Fréquence d'échantillonnage en Hz
F = 10   # Fréquence du signal en Hz
D = 1      # Durée du signal en s
A = 3      # Amplitude du signal
t = np.arange(0,D,1/fe)  # Création de l'axe temporel

s = A*(np.sin(2*np.pi*F*t))
plt.figure(9)
plt.plot(t, s)
plt.xlabel("Temps en seconde")
plt.ylabel("Amplitude")
plt.title("Signal sinusoidal s")
plt.show()


# 1/
f = np.fft.fftfreq(len(s), 1/fe)  # Création de l'axe des fréquences
f =  np.fft.fftshift(f)  # Déplacer les fréquences de manière centrée
S = np.fft.fft(s)  # FFT de s
Y = np.fft.fftshift(S)
module_S = np.abs(Y)
plt.figure(10)
plt.plot(f, module_S)
plt.xlabel("Frequence en Hz")
plt.ylabel("Spectre de s")
plt.title("Module de S")
plt.show()

# 2/
s_inv = np.fft.ifft(S)  # FFT inverse de S
plt.figure(11)
plt.plot(t, s_inv)
plt.xlabel("temps en seconde")
plt.ylabel("Amplitude")
plt.title("FFT inverse de S")
plt.show()


# Bruit Blanc
# 3/
bruit = np.random.rand(len(s))  # Création d'un bruit blanc centré d'amplitude 1
plt.figure(12)
plt.subplot(2, 1, 1)
plt.plot(t, bruit)
plt.xlabel("Temps en seconde")
plt.ylabel("Amplitude")
plt.title("Signal et module du bruit")

B = np.fft.fft(bruit)
module_bruit = np.abs(B)
plt.subplot(2, 1, 2)
plt.plot(f, module_bruit)
plt.xlabel("Frequence en Hz")
plt.ylabel("Spectre du bruit")
plt.show()


# Bruitage du signal
# 4/
signal_bruite = s+0.5*bruit
SIGNAL_BRUITE = np.fft.fft(signal_bruite)
module_sig = np.abs(SIGNAL_BRUITE)
plt.figure(13)
plt.subplot(2, 1, 1)
plt.plot(t, signal_bruite)
plt.xlabel("Temps en seconde")
plt.ylabel("Amplitude")
plt.title("Signal bruité et son module")

plt.subplot(2, 1, 2)
plt.plot(f, module_sig)
plt.xlabel("Frequence en Hz")
plt.ylabel("Spectre du signal bruité")
plt.show()


# Filtrage par FFT
# 5/ et 6/
M = max(0, SIGNAL_BRUITE)  # Amplitude maximale de la TF du signal bruite
seuil = 0.10  # Seuil à 10%
H = np.zeros_like(t)
H[seuil*M > M] = 0

SIGNAL_BRUITE_FILTRE = ???
signal_bruite_filtre = ???
plt.figure(14)
plt.subplot(2, 1, 1)
plt.plot(, ???)
plt.xlabel("Frequence en Hz")
plt.ylabel(???)
plt.title("FFT filtrée avec differents parametres")

plt.subplot(2, 1, 2)
plt.plot(???, ???)
plt.xlabel(???)
plt.ylabel(???)
plt.show()
