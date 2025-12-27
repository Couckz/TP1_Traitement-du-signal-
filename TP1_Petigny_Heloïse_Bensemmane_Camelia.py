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

# Création du signal s en fonction de la définition donnée
s = np.ones_like(t) 
s[t > T] = 0
s[t < -T] = 0

# Affichage du signal
plt.figure(1) #Numérotation de la figure
plt.plot(t, s) #Insertion du temps en abscisse et de l'amplitude en ordonnée
plt.title('Signal s') #Titre du signal
plt.xlabel('temps (s)') #Titre de l'axe des abscisses
plt.ylabel('amplitude') #Titre de l'axe des ordonnés
plt.show() #Affichage de la figure


# 2/
f = np.fft.fftfreq(len(s), 1/fe)  # Création de l'axe des fréquences
f =  np.fft.fftshift(f)  # Déplacer les fréquences de manière centrée
S = np.fft.fft(s)  # TF du signal s
module = np.abs(S) # Calcul du module de la FFT
phase = np.angle(S) # Calcul de la phase de la FFT

# Affichage
plt.figure(2) #Numérotation de la figure
plt.subplot(2, 1, 1) #Définition de l'organisation de la figure
plt.plot(f, module) #Affichage du module de la FFT
plt.xlabel('fréquences (Hz)') #Titre de l'axe des abscisses
plt.ylabel('spectre de s') #Titre de l'axe des ordonnées
plt.title('Module et phase de la TF de s') #Titre de la première figure


plt.subplot(2, 1, 2) #Définition de l'organisation de la figure
plt.plot(f, phase) #Affichage de la phase de la FFT
plt.xlabel('fréquences (Hz)') #Titre de l'axe des abscisses
plt.ylabel('phase') #Titre de l'axe des ordonnées
plt.show() #Affichage de la figure


# 3/
Y = np.fft.fftshift(S)  # Permutation avec fftshift
module = np.abs(Y) #Calcul du module de Y
phase = np.angle(Y) #Calcul de la phase de Y
plt.figure(3) #Numérotation de l'image
plt.subplot(2, 1, 1) #Organisation de la figure
plt.plot(f, module) #Affichage du spectre de la FFT
plt.xlabel('fréquences (Hz)') #Titre de l'axe des abscisses
plt.ylabel('spectre de y') #Titre de l'axe des ordonnées
plt.title('Module et phase de Y') #Titre de la première figure

plt.subplot(2, 1, 2) #Organisation de la figure
plt.plot(f, phase) #Affichage de la phase de la FFT
plt.xlabel('fréquences (Hz)')  #Titre de l'axe des abscisses
plt.ylabel('phase') #Titre de l'axe des ordonnées
plt.show() #Affichage de la figure


# 4/
y_prime = np.fft.ifft(Y)  # TF inverse de Y
s_prime = np.fft.ifft(S)  # TF inverse de S
plt.figure(4) #Numérotation de la figure
plt.subplot(1, 2, 1) #Organisation de la figure
plt.plot(t, y_prime) #Insertion de la courbe y_prime = f(t)
plt.xlabel('temps s en seconde') #Titre de l'axe des abscisses
plt.ylabel('amplitude')  #Titre de l'axe des ordonnées
plt.title("Fft inverse de Y") #Titre de la figure


plt.subplot(1, 2, 2) #Organisation de la figure
plt.plot(t, s_prime) #Insertion de la courbe s_prime = f(t)
plt.xlabel("temps en seconde") #Titre de l'axe des abscisses
plt.ylabel("amplitude") #Titre de l'axe des ordonnées
plt.title("fft inverse de S") #Titre de la figure
plt.show() #Affichage de la figure


# 5/
x = np.convolve(s, s, 'same')  # Produit de convolution
plt.figure(5) #Numérotation de la figure
plt.plot(t, x) #Insertion de la courbe x = f(t)
plt.xlabel("temps en seconde") #Titre de l'axe des abscisses
plt.ylabel("amplitude") #Titre de l'axe des ordonnées
plt.title("Produit de convolution x (s*s)") #Titre de la figure
plt.show() #Affichage de la figure


# 6/
X = np.fft.fft(x)  # TF de x
S_carre = S**2  # Élévation au carré de S

plt.figure(6) #Numérotation de la figure
plt.plot(f, X, label='X') #Insertion de la courbe X = f(f)
plt.plot(f, S_carre, '--', label='S^2') #Insertion de la courbe S_carre = f(f)
plt.xlabel("frequences en Hz") #Titre de l'axe des abscisses
plt.ylabel("Amplitudes") #Titre de l'axe des ordonnées
plt.title("FFT de X et de S au carré") #Titre de la figure
plt.legend() #Placement de la légende
plt.show() #Affichage de la figure


# Partie 2 : TF d'un signal carré

# 1/
# Initialisation
fe = 1000  # Fréquence d'échantillonnage en Hz
F = 100    # Fréquence du signal en Hz
N = 10     # Nombre de périodes
D = (1*N)/F  # Durée du signal en s
t = np.arange(0,D,1/fe) # Création de l'axe temporel

s = square(2*np.pi*F*t) # Signal carré
plt.figure(7) #Numérotation de la figure
plt.plot(t, s) #Insertion de la courbe s = f(t)
plt.xlabel("temps en seconde") #Titre de l'axe des abscisses
plt.ylabel("amplitude") #Titre de l'axe des ordonnées
plt.title("Signal carré sur 10 périodes") #Titre de la figure
plt.show() #Affichage de la figure

# 2/
nb_points_fft = 1024  # Nombre de points de la FFT
f = np.fft.fftfreq(nb_points_fft, 1/fe)  # Création de l'axe des fréquences
f =  np.fft.fftshift(f)  # Déplacer les fréquences de manière centrée

S = np.fft.fft(s, nb_points_fft)  # FFT de s
module_S = np.abs(S) #Calcul du module de la FFT de s
plt.figure(8) #Numérotation de la figure
plt.plot(f, module_S) #Insertion de la courbe module_S = f(f)
plt.xlabel("fréquence en Hz") #Titre de l'axe des abscisses
plt.ylabel("Spectre de s") #Titre de l'axe des ordonnées
plt.title("Module de S") #Titre de la figure
plt.show() #Affichage de la figure


# Partie 3 : Débruitage d'un signal par FFT

# TF et TF inverse
# Initialisation
fe = 1000  # Fréquence d'échantillonnage en Hz
F = 10   # Fréquence du signal en Hz
D = 1      # Durée du signal en s
A = 3      # Amplitude du signal
t = np.arange(0,D,1/fe)  # Création de l'axe temporel

s = A*(np.sin(2*np.pi*F*t))
plt.figure(9) #Numérotation de la figure
plt.plot(t, s) #Insertion de la courbe s = f(t)
plt.xlabel("Temps en seconde") #Titre de l'axe des abscisses
plt.ylabel("Amplitude") #Titre de l'axe des ordonnées
plt.title("Signal sinusoidal s") #Titre de la figure
plt.show() #Affichage de la figure 


# 1/
f = np.fft.fftfreq(len(s), 1/fe)  # Création de l'axe des fréquences
f =  np.fft.fftshift(f)  # Déplacer les fréquences de manière centrée
S = np.fft.fft(s)  # FFT de s
Y = np.fft.fftshift(S) #Recentrer les fréquences
module_S = np.abs(Y) #Calcul du module de Y
plt.figure(10) #Numérotation de la figure
plt.plot(f, module_S) #Insertion de la droite module_S = f(f)
plt.xlabel("Frequence en Hz") #Titre de l'axe des abscisses
plt.ylabel("Spectre de s") #Titre de l'axe des ordonnées
plt.title("Module de S") #Titre de la figure
plt.show() #Affichage de la figure 


# 2/
s_inv = np.fft.ifft(S)  # FFT inverse de S
plt.figure(11) #Numérotation de la figure
plt.plot(t, s_inv) #Insertion de la droite s_inv = f(t)
plt.xlabel("temps en seconde") #Titre de l'axe des abscisses
plt.ylabel("Amplitude") #Titre de l'axe des ordonnées
plt.title("FFT inverse de S") #Titre de la figure
plt.show() #Affichage de la figure 



# Bruit Blanc
# 3/
bruit = np.random.rand(len(s))  # Création d'un bruit blanc centré d'amplitude 1
plt.figure(12) #Numérotation de la figure
plt.subplot(2, 1, 1) #Organisation de la figure
plt.plot(t, bruit) #Insertion de la droite bruit=f(t)
plt.xlabel("Temps en seconde") #Titre de l'axe des abscisses
plt.ylabel("Amplitude") #Titre de l'axe des ordonnées
plt.title("Signal et module du bruit") #Titre de la figure

B = np.fft.fft(bruit) #FFT du bruit
module_bruit = np.abs(B) #Calcul du module du bruit
plt.subplot(2, 1, 2) #Organisation de la figure
plt.plot(f, module_bruit) #Insertion de la droite module_bruit=f(f)
plt.xlabel("Frequence en Hz") #Titre de l'axe des abscisses
plt.ylabel("Amplitude") #Titre de l'axe des ordonnées
plt.show() #Affichage de la figure 



# Bruitage du signal
# 4/
signal_bruite = s+0.5*bruit #Application du bruit au signal
SIGNAL_BRUITE = np.fft.fft(signal_bruite) #Calcul de la fft du signal bruité
module_sig = np.abs(SIGNAL_BRUITE) #Calcul du module du signal bruité
plt.figure(13) #Numérotation de la figure
plt.subplot(2, 1, 1) #Organisation de la figure
plt.plot(t, signal_bruite) #Insertion de la droite signal_bruite = f(t)
plt.xlabel("Temps en seconde") #Titre de l'axe des abscisses
plt.ylabel("Amplitude") #Titre de l'axe des ordonnées
plt.title("Signal bruité et son module") #Titre de la figure

plt.subplot(2, 1, 2) #Organisation de la figure
plt.plot(f, module_sig) #Insertion de la droite module_sig = f(f)
plt.xlabel("Frequence en Hz") #Titre de l'axe des abscisses
plt.ylabel("Spectre du signal bruité") #Titre de l'axe des ordonnées
plt.show() #Affichage de la figure 



# Filtrage par FFT
# 5/ et 6/
# FFT du signal bruité
M = max(module_sig)  # Amplitude maximale de la TF du signal bruite
seuil = 0.1  # Seuil à 10%
H = np.zeros_like(SIGNAL_BRUITE) #Creation d'un tableau de 0
H[module_sig > seuil*M] = 1 #Definition du filtre

SIGNAL_BRUITE_FILTRE = SIGNAL_BRUITE*H #Application du filtre au signal bruité
signal_filtre = np.fft.ifft(SIGNAL_BRUITE_FILTRE) #Determiner le signal dans le domaine temporel

plt.figure(14) #Numérotation de la figure
plt.subplot(2, 1, 1) #Organisation de la figure
plt.plot(f, SIGNAL_BRUITE_FILTRE) #Insertion de la droite SIGNAL_BRUITE_FILTRE = f(f)
plt.xlabel("Frequence en Hz") #Titre de l'axe des abscisses
plt.ylabel("Amplitude") #Titre de l'axe des ordonnées
plt.title("FFT filtrée avec differents parametres et signal d'origine") #Titre de la figure

plt.subplot(2, 1, 2) #Organisation de la figure
plt.plot(f, signal_filtre) #Insertion de la droite signal_filtre = f(f)
plt.xlabel("Fréquence en Hz") #Titre de l'axe des abscisses
plt.ylabel("Amplitude") #Titre de l'axe des ordonnées
plt.show() #Affichage de la figure 

