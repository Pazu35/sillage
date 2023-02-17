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
    
def plot_accelero(date,acc,timestamp):
   
    minute = mdates.MinuteLocator(interval = 2)
    second = mdates.SecondLocator(bysecond = np.arange(0,60,10)) ##decalage a faire bysecond
    hour = mdates.HourLocator()
    formatter = mdates.DateFormatter('%H:%M:%S')
    
    fig, ax = plt.subplots()  
    ax.plot(date,acc)
        # [ax.axvline(x, color = 'g') for x in timestamp['timestamp']]
        # ax.xaxis.set_major_locator(minute)
        # ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_locator(second)
    
    
    # datemin = date.loc[date.dt.minute == 34].iloc[0]
    # datemax = date.loc[date.dt.minute == 42].iloc[0]
    #ax.set_xlim(datemin, datemax)
    fig.autofmt_xdate()
    plt.ylabel('accélaration verticale (m.s-2)')
    plt.xlabel('temps')
    plt.show()

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
    

    half_height = signal.peak_widths(accZ_fft, peaks, rel_height=0.5)
    
    
    plt.plot(accZ_fftfreq[peaks],accZ_fft[peaks],'x')
    plt.hlines(y = half_height[1], xmin =  accZ_fftfreq[half_height[2].astype(np.int32)], xmax = accZ_fftfreq[half_height[3].astype(np.int32)] ,color="C2")
    
    
    
    height =  signal.peak_widths(accZ_fft, peaks, rel_height= 0.995)
    plt.hlines(y = height[1], xmin =  accZ_fftfreq[height[2].astype(np.int32)], xmax = accZ_fftfreq[height[3].astype(np.int32)] ,color="C2")
    
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

def butter_filter(acc,f,time_sec,timestamp,T):

    b,a = signal.butter(N = 1, Wn = f, btype='bandpass', analog=True, output='ba')
    sos = signal.butter(N = 1, Wn = f, btype='bandpass', analog=False, output='sos', fs = df['output_Hz'].iloc[0])
    
    w, h = signal.freqs(b, a)
    filtered = signal.sosfilt(sos,acc)
    
    
    plt.figure()
    plt.semilogx(w, 20 * np.log10(abs(h)))

    plt.title('Butterworth filter frequency response')
    
    plt.xlabel('Frequency [radians / second]')
    
    plt.ylabel('Amplitude [dB]')
    
    plt.figure()
    plt.plot(time_sec,filtered)
    plt.xlabel('temps (s)')
    plt.ylabel('Amplitude')
    [plt.vlines(x,0,np.max(filtered), color = 'g') for x in timestamp['total_sec']]
    
    peaks, _ = signal.find_peaks(filtered,distance = 250, height = 0.2)
    #print(timestamp['total_sec'].values, time_sec.values[peaks])
    metrics = precision_recall(timestamp['total_sec'].values, time_sec.values[peaks],T)

    plt.plot(time_sec.values[peaks],filtered[peaks],'x', label = f'RMSE={round(metrics[-1],2)}, F1={round(metrics[-2],2)}')
    plt.legend()
    return filtered,time_sec.values[peaks]
    
    
    
    
def periodogram(acc,time_sec,T,timestamp):
    plt.figure()
    density = []
    
    box_size =  (int(np.sqrt(10*T))+1)**2
    box_size = 121
    for i in range(box_size,acc.size):
        f, pxx = signal.periodogram(acc[i-box_size:i], fs = df['output_Hz'].iloc[0], window = 'boxcar', nfft = None)
        density.append(pxx)
    
    density = np.array(density)
    #density[density == 0] = np.NaN
    #density = np.log(density)
    #density[density == -np.inf] = -100
    #density[density == np.NaN] = -100
    
    
    time_sec = time_sec[box_size:].values
    print((f[0],f[-1],time_sec[0],time_sec[-1]))
    plt.imshow(density, aspect='auto',vmin = 0, extent = (f[0],f[-1],time_sec[0],time_sec[-1]))
    plt.colorbar()
    plt.figure()
    power = np.sum(density, axis = 1)

    plt.plot(time_sec,power)
    plt.xlabel('temps (s)')
    plt.ylabel('Energie (J)')
    [plt.vlines(x,0,np.max(power), color = 'g') for x in timestamp['total_sec']]
    
    peaks, _ = signal.find_peaks(power,distance = 100, height = 0.5)
    rmse = mean_squared_error(timestamp['total_sec'], time_sec[peaks])
    
    plt.plot(time_sec[peaks],power[peaks],'x', label = f'RMSE = {rmse}')
    plt.legend()
    #plt.xticks(label = time_sec)


    plt.figure()
    power = np.sum(density, axis = 0)
    plt.plot(f,power)
    plt.xlabel('fréquence (Hz)')
    plt.ylabel('Energie (J)')
    
    
    return density,power,peaks

def welch(acc,time_sec,T,t,timestamp):
    wave_lenght = 2*T*df['output_Hz'].iloc[0]
    
    plt.figure()
    density = []
    box_size =  (int(np.sqrt(10*T))+1)**2
    nperseg = (int(np.sqrt(10*t))+1)**2
    
    box_size = 121
    nperseg = 36
    for i in range(box_size,acc.size):
        f,pxx = signal.welch(acc[i-box_size:i],fs = df['output_Hz'].iloc[0],nperseg = nperseg, window = 'boxcar')
        density.append(pxx)

    density = np.array(density)
    time_sec = time_sec[box_size:].values
    
    power = np.sum(density, axis = 1)
    
    peaks, _ = signal.find_peaks(power,distance = wave_lenght, height = 0.1*np.max(power))
    
    metrics = precision_recall(time_sec[peaks],timestamp['total_sec'],T)
    
    plt.imshow(density, aspect='auto', extent = (f[0],f[-1],time_sec[0],time_sec[-2]))
    plt.colorbar()
    plt.xlabel('frequence (Hz)')
    plt.ylabel('temps (s)')
    
    plt.figure()
    plt.plot(time_sec,power)


    plt.plot(time_sec[peaks],power[peaks],'x',label = f'RMSE={round(metrics[-1],2)}, F1={round(metrics[-2],2)}')
    plt.xlabel('temps (s)')
    plt.ylabel('Energie (J)')
    plt.legend()
    
    [plt.vlines(x,0,np.max(power), color = 'g') for x in timestamp['total_sec']]

            
    return density,power, time_sec[peaks]
    
 
def precision_recall(detection_list,truth_list,T):
    TP_timestamp = []
    TP_detection = []
    FP = []
    FN = []
    for truth in truth_list:
        A = ( detection_list >= truth - T) & (detection_list <= truth + T)
        if np.any(A):
            TP_timestamp.append(truth)
        else:
            FN.append(truth)

    for detection in detection_list:
        B = (truth_list >= detection - T) & (truth_list <= detection + T)
        if np.any(B):
            TP_detection.append(detection)
        else:
            FP.append(detection)
            
    P = len(TP_detection)/(len(TP_detection)+len(FP))
    R = len(TP_detection)/(len(TP_detection)+len(FN))
    F1 = 2*P*R/(P+R)
    
    RMSE = mean_squared_error(TP_timestamp,TP_detection)
    return TP_timestamp,TP_detection,FP,FN,P,R,F1,RMSE
 
if __name__ == '__main__':
    
    treshold = 1.5

    
    log_file_1 = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_08_02\\sillage1_ahrs1_log_2023-02-08_14_29_14.log'  ##timestamp correcte mais bon frequencage
    log_file_2 = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_19_01_23\\ahrs1_log_2023-01-19_13_34_14.log'  ##sortie en mer
    #log_file_3 = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_10_13\\ahrs1_log_2022-10-13_09_34_42.log'  ##bon timestamps 1 problème de frequencage
    #log_file_4 = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_10_12\\ahrs1_log_2022-10-12_12_36_08.log' ##idem
    
    gps_file = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_08_02\\2023-02-08 15_30_37.gpx'
    #gps_file = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_19_01_23\\2023-01-19 14_39_15.gpx'
    
    
    timestamp_file = 'C:\\Users\\oscar\\Documents\\3A\\guerledan\\log\\log_08_02\\timestamp_2022-08-02.txt'
    

    df = read_log(log_file_1)
    
    Date = df['rtcDate'].dt.date.iloc[0]
    
    t_gps,v_gps = get_gps_info(gps_file)
    t_ini,t_end = t_gps[0],t_gps[-1]
    #t_ini,t_end = datetime.combine(Date, time(8,0,0)),datetime.combine(Date, time(8,10,0))
    # t_ini_2,t_end_2 = datetime.combine(Date, time(14,2,0)),datetime.combine(Date, time(14,4,0))

    df = df.loc[(df['rtcDate'] > np.datetime64(t_ini)) & (df['rtcDate'] < np.datetime64(t_end)) ]
    # df_1 = df.loc[(df['rtcDate'] > np.datetime64(t_ini_1)) & (df['rtcDate'] < np.datetime64(t_end_1)) ]
    # df_2 = df.loc[(df['rtcDate'] > np.datetime64(t_ini_2)) & (df['rtcDate'] < np.datetime64(t_end_2)) ]    
    
    
    time_sec,acc,gyr,mag = get_nav_info(df)
    
    fft = acc_filter_fft(acc['aZ'].values,df)
    plt.close('all')
    
    timestamp = read_timestamp(timestamp_file,df)


    #acc['aZ'] = acc_filter_treshold(acc['aZ'], treshold)
    
    plot_accelero(df['rtcDate'],acc['aZ'],timestamp)
    
    f_wave,f_sill = fft[0][fft[-1]]
    if  abs(f_wave - f_sill) >= 0.1:       
        f1 = [f_sill-0.1, f_sill + 0.1]
        f2 = [f_wave-0.1, f_wave + 0.1]
        
    else: 
        f1 = [f_sill - abs(f_sill-f_wave), f_sill + abs(f_sill-f_wave)]
        f2 = [f_wave - abs(f_sill-f_wave), f_wave + abs(f_sill-f_wave)]
    
    
    t = 1/f_sill
    T = 13*t
    #f,t,Zxx = acc_filter_stft(acc['aZ'].values,df)
    
    
    filtered,detection_time_butter = butter_filter(acc['aZ'].values,f1,time_sec,timestamp,T)

    #density,power,peaks = periodogram(acc['aZ'].values,time_sec,T,timestamp)
    density,power, detection_time_welch = welch(acc['aZ'].values,time_sec,T,t,timestamp)

    TP_timestamp,TP_detection,FP,FN,P,R,F1,RMSE = precision_recall(detection_time_welch,timestamp['total_sec'],T)

    