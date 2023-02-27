# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:30:33 2022

@author: oscar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from datetime import datetime, timedelta, date, time
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import scipy.fft as scf
from scipy import signal
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import gpxpy
from geopy import distance
import matplotlib as mpl

#plt.style.use('seaborn-deep')

def read_log(log_file):


    col_names = ['rtcDate','ms','Ahrs_id','aX','aY','aZ','gX','gY','gZ','mX','mY','mZ','imu_degC','output_Hz','']
    df = pd.read_csv(log_file,sep =';|,',skiprows = 30,index_col= False, names = col_names,usecols = col_names[:-1],parse_dates=['rtcDate'], engine = 'python')
    df['rtcDate'] = df['rtcDate'] + pd.to_timedelta(df['ms'],unit = 'ms')
    df.drop('ms',axis = 1, inplace = True)
   
    
    return df


def get_nav_info(df):    
 
    timestamp = df['rtcDate'].dt.hour * 3600 + df['rtcDate'].dt.minute * 60 + df['rtcDate'].dt.second + df['rtcDate'].dt.microsecond * 10**-6
    acc = df[['aX','aY','aZ']]*0.01
    gyr = df[['gX','gY','gZ']]*np.pi/180
    mag = df[['mX','mY','mZ']]
    
    return timestamp,acc,gyr,mag

def get_gps_info(file_name):     ##Extract the time and speed for each point of each tracks
    gpx_file = open(file_name, 'r')
    gpx = gpxpy.parse(gpx_file)
    t = []
    v = []
    c = 0
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if c == 0:
                    t.append(point.time)
                    v.append(0)
                else:
                    d = distance.distance((previous_point.latitude, previous_point.longitude), (point.latitude, point.longitude)).m
                    t.append(point.time)
                    v.append(d/(t[c]-t[c-1]).total_seconds())
                previous_point = point
                c += 1
    return t,v
    
def plot_accelero(date,acc,timestamp,time_sec,T,fe,distance,height,dist_recherche,Mer):
    acc = acc - np.mean(acc)
    peaks, _ = signal.find_peaks(acc,distance = distance, height = height*np.max(acc))
    time_sec = time_sec.values
    

    plt.figure()
    plt.plot(time_sec,acc)
    plt.xlabel('Temps (s)')
    plt.ylabel(r'Accélération  ($m^2 .s^{-1}$)')
    
    if Mer:
        plt.plot(time_sec[peaks],acc.values[peaks],'x')

    else:
        metrics = precision_recall(timestamp['total_sec'].values, time_sec[peaks],dist_recherche) 
        plt.plot(time_sec[peaks],acc.values[peaks],'x',label = f"RMSE = {round(metrics[-1],2)}s, F1 = {round(metrics[-2],2)}")
        [plt.vlines(x,0,np.max(acc), color = 'g') for x in timestamp['total_sec']]
        plt.legend()
    
    
    # minute = mdates.MinuteLocator(interval = 2)
    # second = mdates.SecondLocator(bysecond = np.arange(0,60,10)) ##decalage a faire bysecond
    # hour = mdates.HourLocator()
    # formatter = mdates.DateFormatter('%H:%M:%S')
    
    # fig, ax = plt.subplots()  
    # ax.plot(date,acc)
    #     # [ax.axvline(x, color = 'g') for x in timestamp['timestamp']]
    #     # ax.xaxis.set_major_locator(minute)
    #     # ax.xaxis.set_major_formatter(formatter)
    #     # ax.xaxis.set_minor_locator(second)
    
    
    # # datemin = date.loc[date.dt.minute == 34].iloc[0]
    # # datemax = date.loc[date.dt.minute == 42].iloc[0]
    # #ax.set_xlim(datemin, datemax)
    # fig.autofmt_xdate()
    # plt.ylabel('accélaration verticale (m.s-2)')
    # plt.xlabel('temps')
    # plt.show()
    
    return time_sec[peaks]


def read_timestamp(timestamp_file,df):
    timestamp = pd.read_csv(timestamp_file,sep = ';',index_col = False,names = ['timestamp'])
    timestamp['timestamp'] = pd.to_datetime(timestamp['timestamp'], format = '%H:%M:%S')

    timestamp['date'] = df['rtcDate'].dt.date.iloc[0]
    timestamp['timestamp'] = pd.to_datetime(timestamp['date'].astype(str) + ' ' + timestamp['timestamp'].dt.time.astype(str))
    timestamp.drop('date',axis = 1, inplace = True)
    timestamp['total_sec'] = timestamp['timestamp'].dt.hour * 3600 +  timestamp['timestamp'].dt.minute * 60 + timestamp['timestamp'].dt.second
    
    return timestamp

def acc_filter_treshold(accZ,treshold):
    accZ = accZ - accZ.mean()
    accZ.loc[(accZ < treshold) & (accZ > - treshold) ] = 0  
    return accZ

def acc_filter_fft(accZ,df):
    
    ###ajouter plage temporelle sur l acc pour visualisr les vagues ou non
    freq = df['output_Hz'].iloc[0]
    N = len(accZ)
    
    accZ = accZ - accZ.mean()
    
    accZ_fft = scf.fft(accZ, norm = 'forward')
    accZ_fft = np.abs(accZ_fft) **2

    
    accZ_fftfreq = scf.fftfreq(N,1/freq)
    
    i = accZ_fftfreq>0
    accZ_fftfreq,accZ_fft= accZ_fftfreq[i],accZ_fft[i]
    
    plt.figure()
    plt.plot(accZ_fftfreq,accZ_fft)
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency (Hz)')
    
    peaks, _ = signal.find_peaks(accZ_fft,distance = 100, height = 0.00001 )
    peaks = peaks[np.argpartition(accZ_fft[peaks], -2)[-2:]]
    peaks = peaks[::-1]

    half_height = signal.peak_widths(accZ_fft, peaks, rel_height=0.5)
    
    c = ['green','orange']
    for i in range(len(peaks)):
        plt.plot(accZ_fftfreq[peaks[i]],accZ_fft[peaks[i]],'x',color = c[i],label = f'f={round(accZ_fftfreq[peaks[i]],2)}Hz')

    #plt.legend()
    #plt.plot(accZ_fftfreq[peaks],accZ_fft[peaks],'x')
    #plt.hlines(y = half_height[1], xmin =  accZ_fftfreq[half_height[2].astype(np.int32)], xmax = accZ_fftfreq[half_height[3].astype(np.int32)] ,color="C2")
    
    
    
    height =  signal.peak_widths(accZ_fft, peaks, rel_height= 0.995)
    #plt.hlines(y = height[1], xmin =  accZ_fftfreq[height[2].astype(np.int32)], xmax = accZ_fftfreq[height[3].astype(np.int32)] ,color="C2")
    
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('Amplitude')
    # plt.figure()
    # plt.plot(accZ_fft)
    # plt.plot(peaks,accZ_fft[peaks],'x')
    # plt.hlines(*half_height[1:] ,color="C3")
    
    
    #plt.xlim(right = 0.001)
    
    return accZ_fftfreq,accZ_fft,peaks



def acc_filter_stft(accZ,df):
    plt.figure()
    freq = df['output_Hz'].iloc[0]
    f,t,Zxx = signal.stft(accZ, freq, nperseg=256)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, shading='gouraud')
    plt.ylim(0,.5)

    plt.title('STFT Magnitude')
    
    plt.ylabel('Frequency [Hz]')
    
    plt.xlabel('Time [sec]')

    plt.show()

    return f,t,Zxx

def butter_filter(acc,f,time_sec,timestamp,T,fe,distance,height,dist_recherche,Mer):

    b,a = signal.butter(N = 1, Wn = f, btype='bandpass', analog=True, output='ba')
    sos = signal.butter(N = 1, Wn = f, btype='bandpass', analog=False, output='sos', fs = fe)
    
    w, h = signal.freqs(b, a)
    filtered = signal.sosfilt(sos,acc)
    filtered = filtered[50:]
    time_sec = time_sec[50:]
    
    plt.figure()
    plt.semilogx(w, 20 * np.log10(abs(h)))

    plt.title('Butterworth filter frequency response')
    
    plt.xlabel('Frequency [radians / second]')
    
    plt.ylabel('Amplitude [dB]')
    
    
    peaks, _ = signal.find_peaks(filtered,distance = distance, height = height*np.max(filtered))

    plt.figure()
    plt.plot(time_sec,filtered)
    plt.xlabel('temps (s)')
    plt.ylabel('Amplitude')
    
    
    if Mer:
        plt.plot(time_sec.values[peaks],filtered[peaks],'x')

    else:
        metrics = precision_recall(timestamp['total_sec'].values, time_sec.values[peaks],dist_recherche)
        [plt.vlines(x,0,np.max(filtered), color = 'g') for x in timestamp['total_sec']]
        plt.plot(time_sec.values[peaks],filtered[peaks],'x',label = f'RMSE={round(metrics[-1],2)}, F1={round(metrics[-2],2)}')
        #plt.legend()


    return filtered,time_sec.values[peaks]
    
    
    
    
def periodogram(acc,time_sec,t,timestamp,fe,distance,height,dist_recherche,Mer):

    plt.figure()
    density = []
    box_size =  (int(np.sqrt(10*t*fe)))**2
    for i in range(box_size,acc.size):
        f, pxx = signal.periodogram(acc[i-box_size:i], fs = fe, window = 'boxcar', nfft = None)
        density.append(pxx)
    
    density = np.array(density)
    #density[density == 0] = np.NaN
    #density = np.log(density)
    #density[density == -np.inf] = -100
    #density[density == np.NaN] = -100
    
    
    time_sec = time_sec[box_size:].values
    plt.imshow(density, aspect='auto',vmin = 0, extent = (f[0],f[-1],time_sec[0],time_sec[-1]))
    plt.colorbar()
    plt.ylabel('Temps (s)')
    plt.xlabel('Fréquence (Hz)')
    plt.title(r'DSP ($\frac{m^2}{s^4 Hz}$)')
    
    
    power = np.sum(density, axis = 0)
    peaks, _ = signal.find_peaks(power,distance = 5, height = 0.4*np.max(power))
    half_height = signal.peak_widths(power, peaks, rel_height=0.3)
    
    plt.figure()
    c = ['green','orange']
    plt.plot(f,power)
    for i in range(len(peaks)):
        plt.plot(f[peaks[i]],power[peaks[i]],'x',color = c[i],label = f'f={round(f[peaks[i]],2)}Hz')
        plt.hlines(y = half_height[1][i], xmin =  f[int(half_height[2][i])], xmax = f[int(half_height[3][i])+1] ,color = c[i], label = f'f=[{round(f[int(half_height[2][i])],2)}, {round(f[int(half_height[3][i])+1],2)}]Hz \n')

    plt.xlabel('fréquence (Hz)')
    plt.ylabel(r'$\sum_t DSP(t)$    $(\frac{m^2}{s^4 Hz})$')
    plt.legend()

    
 
    plt.figure()
    power = np.sum(density, axis = 1)
    peaks, _ = signal.find_peaks(power,distance = distance, height = height*np.max(power))

    plt.plot(time_sec,power)
    plt.xlabel('Temps (s)')
    plt.ylabel(r'$\sum_f DSP(f)$    $(\frac{m^2}{s^4 Hz})$')
    
    if Mer:
        plt.plot(time_sec[peaks],power[peaks],'x')

    else:
        metrics = precision_recall(time_sec[peaks],timestamp['total_sec'],dist_recherche)
        [plt.vlines(x,0,np.max(power), color = 'g') for x in timestamp['total_sec']]
        plt.plot(time_sec[peaks],power[peaks],'x', label = f"RMSE = {round(metrics[-1],2)}s, F1 = {round(metrics[-2],2)}")
        #plt.legend()
        
    #peaks = peaks[:-1]

    return density,power,time_sec[peaks]

def welch(acc,time_sec,T,t,timestamp,fe,distance,height,dist_recherche,Mer):
    
    plt.figure()
    density = []
    box_size =  (int(np.sqrt(10*t*fe)))**2
    nperseg = (int(np.sqrt(2*t*fe)))**2


    for i in range(box_size,acc.size):
        f,pxx = signal.welch(acc[i-box_size:i],fs = fe,nperseg = nperseg, window = 'boxcar')
        density.append(pxx)

    density = np.array(density)
    time_sec = time_sec[box_size:].values
    

    plt.imshow(density, aspect='auto', extent = (f[0],f[-1],time_sec[0],time_sec[-2]))
    plt.colorbar()
    plt.ylabel('Temps (s)')
    plt.xlabel('Fréquence (Hz)')
    plt.title(r'DSP ($\frac{m^2}{s^4 Hz}$)')
    
    plt.figure()
    power = np.sum(density,axis=0)
    plt.plot(f,power)
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel(r'$\sum_t DSP(t)$    $(\frac{m^2}{s^4 Hz})$')
    
    
    
    power = np.sum(density, axis = 1)
    peaks, _ = signal.find_peaks(power,distance = distance, height = height*np.max(power))
    plt.figure()
    plt.plot(time_sec,power)
    if Mer:
        plt.plot(time_sec[peaks],power[peaks],'x')

    else:
        metrics = precision_recall(time_sec[peaks],timestamp['total_sec'],dist_recherche)
        plt.plot(time_sec[peaks],power[peaks],'x',label = f'RMSE={round(metrics[-1],2)}s, F1={round(metrics[-2],2)}')
        [plt.vlines(x,0,np.max(power), color = 'g') for x in timestamp['total_sec']]
        #plt.legend()


    plt.xlabel('Temps (s)')
    plt.ylabel(r'$\sum_f DSP(f)$    $(\frac{m^2}{s^4 Hz})$')
    


            
    return density,power, time_sec[peaks]
    
 
def precision_recall(detection_list,truth_list,dist):
    detection_mean = []
    TP_timestamp = []
    TP_detection = []
    FP = []
    FN = []
    for truth in truth_list:
        A = ( detection_list >= truth - T) & (detection_list <= truth + T)
        if np.any(A):
            TP_timestamp.append(truth)
            detection_mean.append(np.mean(detection_list[A]))
        else:
            FN.append(truth)

    for detection in detection_list:
        B = (truth_list >= detection - T) & (truth_list <= detection + T)
        if np.any(B):
            TP_detection.append(detection)
        else:
            FP.append(detection)
         
    #print(len(detection_mean))
    #print(len(TP_timestamp))
    P = len(TP_detection)/(len(TP_detection)+len(FP))
    R = len(TP_detection)/(len(TP_detection)+len(FN))
    F1 = 2*P*R/(P+R)

    # print(f"TP_timestamp: {TP_timestamp}, ({len(TP_timestamp)})")
    # print(f"detection_mean:{detection_mean},({len(detection_mean)})")
    # print(f"TP_detection:{TP_detection},({len(TP_detection)}) \n")
    RMSE = mean_squared_error(TP_timestamp,detection_mean)
    return TP_timestamp,TP_detection,FP,FN,P,R,F1,RMSE
 
if __name__ == '__main__':
    
    Mer = False

    treshold = 1.5

    
    
    log_file = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_08_02\\sillage1_ahrs2_log_2023-02-08_14_29_14.log'  ##timestamp correcte mais bon frequencage
    
    if Mer:
        log_file= 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_rade2\\sillage1_ahrs1_log_2023-02-17_12_58_03.log'  ##sortie en mer
    #log_file_3 = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_10_13\\ahrs1_log_2022-10-13_09_34_42.log'  ##bon timestamps 1 problème de frequencage
    #log_file_4 = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_10_12\\ahrs1_log_2022-10-12_12_36_08.log' ##idem
    
    gps_file = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_08_02\\2023-02-08 15_30_37.gpx'
    #gps_file = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_19_01_23\\2023-01-19 14_39_15.gpx'
    
    
    timestamp_file = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_08_02\\timestamp_2022-08-02.txt'
    

    df = read_log(log_file)

    
    Date = df['rtcDate'].dt.date.iloc[0]
    
    t_gps,v_gps = get_gps_info(gps_file)
    t_ini,t_end = t_gps[0],t_gps[-1]
    #t_ini,t_end = datetime.combine(Date, time(0,0,0)),datetime.combine(Date, time(8,10,0))
    
    if Mer:
        t_ini,t_end = datetime.combine(Date, time(14,23,0)),datetime.combine(Date, time(14,27,0))
 
    
    df = df.loc[(df['rtcDate'] > np.datetime64(t_ini)) & (df['rtcDate'] < np.datetime64(t_end)) ]

    
    
    time_sec,acc,gyr,mag = get_nav_info(df)
    
    fft = acc_filter_fft(acc['aZ'].values,df)

    timestamp = read_timestamp(timestamp_file,df)


    #acc['aZ'] = acc_filter_treshold(acc['aZ'], treshold)
    
    #plt.close('all')
    
    f_sill,f_wave = fft[0][fft[-1]]
    
    if Mer:
        f_wave, f_sill = 1,0.6
        
        
    if  abs(f_wave - f_sill) >= 0.1:       
        f1 = [f_sill-0.1, f_sill + 0.1]
        f2 = [f_wave-0.1, f_wave + 0.1]
        
    else: 
        f1 = [f_sill - abs(f_sill-f_wave), f_sill + abs(f_sill-f_wave)]
        f2 = [f_wave - abs(f_sill-f_wave), f_wave + abs(f_sill-f_wave)]
    

    t = 1/f_sill
    T = 20*t
    fe = df['output_Hz'].iloc[0]
    distance = T*fe
    height = 0.3
    dist_recherche =T
    #f,t,Zxx = acc_filter_stft(acc['aZ'].values,df)
    
    detection_time_naif = plot_accelero(df['rtcDate'],acc['aZ'],timestamp,time_sec,T,fe,distance,height,dist_recherche,Mer)    
    filtered,detection_time_butter = butter_filter(acc['aZ'].values,f1,time_sec,timestamp,T,fe,distance,height,dist_recherche,Mer)
    density,power,detection_time_perio = periodogram(acc['aZ'].values,time_sec,t,timestamp,fe,distance,height,dist_recherche,Mer)
    density,power, detection_time_welch = welch(acc['aZ'].values,time_sec,T,t,timestamp,fe,distance,height,dist_recherche,Mer)

    metrics_naif = precision_recall(detection_time_naif,timestamp['total_sec'],dist_recherche)
    metrics_butter = precision_recall(detection_time_butter,timestamp['total_sec'],dist_recherche)
    metrics_perio = precision_recall(detection_time_perio,timestamp['total_sec'],dist_recherche)
    metrics_welch = precision_recall(detection_time_welch,timestamp['total_sec'],dist_recherche)
    
    
    # f, pxx = signal.periodogram(acc['aZ'].values, fs = df['output_Hz'].iloc[0], window = 'boxcar', nfft = None)
    # plt.plot(f,pxx)
    # plt.xlabel('fréquence (Hz)')
    # plt.ylabel(r"Densité spectrale de puissance ($\frac{m^2}{s^4 Hz}$)")
    # peaks, _ = signal.find_peaks(pxx,distance = 100, height = 0.1)
    # peaks = peaks[np.argpartition(pxx[peaks], -2)[-2:]]
    # for peak in peaks:
    #     plt.scatter(f[peak],pxx[peak],marker = 'x', s = 20, c='green', label=f'f={round(f[peak],2)}Hz')
    # plt.legend()