# -*- coding: utf-8 -*-

import lire_log
import matplotlib. pyplot as plt
import numpy as np
import scipy.signal as sg


############################# Importation des données ##########################################""

file_name_ahrs = 'Log/ahrs1_log_2023-01-19_13_34_14.log'
_, acc, gyr, mag, temp, rate, T = lire_log.log(file_name_ahrs)

# borne_inf = 27000
# borne_sup = 51000

aZ = acc[:,2] #rajouter borne_inf et borne_sup si limitation dans le temps
T = np.array(T)

################################## Paramètres ##################################

# Setting standard filter requierments
order = 2
fs = 50
cutoff = 0.8
gain = 100

#################### Calcul du spectre du signal : transformée de Fourier ###########

sp = np.fft.fft(aZ)
freq = np.fft.fftfreq(len(T))


plt.figure()
plt.plot(freq, np.sqrt(sp.real**2 + sp.imag**2))
# plt.plot(freq, np.log(np.sqrt(sp.real**2 + sp.imag**2)))
plt.title('Transformée de Fourier')

# Choix des/ de la fréquence de coupure :

## ToDo

# Liste des filtres

def butter_passe_bande(fc_inf,fc_sup, fs, order):
    nyq = 0.5 * fs
    b, a = sg.butter(order, (fc_inf/nyq, fc_sup/nyq), btype='bandpass', analog=False)
    return b, a

def butter_passe_bande_filtre(data, fc_inf, fc_sup, order):
    b, a = butter_passe_bande(fc_inf, fc_sup, fs, order=order)
    y = sg.lfilter(b, a, data)
    return y

# Application des différents filtres


######## Filtre Butterworth passe-bande ########

# Setting standard filter requirements.
order = 2
fs = 50
fc_inf, fc_sup = 0.1 , 0.8

#Calcul de la réponse et du filtre
b, a = butter_passe_bande(fc_inf, fc_sup, fs, order)
w, h = sg.freqz(b, a, worN=8000)
y = sg.lfilter(b, a , aZ)

# Visualisation
plt.figure('pb')
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b') # Filtre
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Bandpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

plt.subplot(2, 1, 2) ### Réponse au filtre
plt.plot(T, (aZ - np.mean(aZ))*100, 'b-', label='data')
plt.plot(T, (y-np.mean(y))*100, 'g-', linewidth=2, label='filtered data')

plt.xlabel('Time [sec]')
plt.grid()
plt.legend()
