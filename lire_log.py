# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import datetime
import gpxpy
import gpxpy.gpx
# from geopy import distance
import matplotlib.dates as mdates
import scipy.signal as sg

###### Importation, traitement et visualisation des données GPS #####








# def replace_tzinfo(dt):
#     return dt.replace(tzinfo=datetime.timezone(dt.utcoffset()))

# def gps(gpx):
# 	gpx = gpxpy.parse(gpx_file)
# 	t_gps = []
# 	v_gps = []
# 	c = 0
# 	for track in gpx.tracks:
# 		for segment in track.segments:
# 			for point in segment.points:
# 				point.time = replace_tzinfo(point.time)
# 				if c == 0:
# 					t_gps.append(point.time)
# 					v_gps.append(0)
# 				else:
# 					d = distance.distance((previous_point.latitude, previous_point.longitude), (point.latitude, point.longitude)).m
# 					t_gps.append(point.time)
# 					v_gps.append(d/(t_gps[c]-t_gps[c-1]).total_seconds())
# 				previous_point = point
# 				c += 1

# 	fig, ax = plt.subplots()
# 	plt.plot_date(t_gps, v_gps, '-')
# 	fig.autofmt_xdate()


#### Importation des données AHRS #####



def log(file_name):
	time = []
	acc = []
	gyr = []
	mag = []
	temp = []
	rate = []
	T = []
	f = open(file_name, 'r')

	count = 0
	for line in f:

		l = line.strip()
		# print("First step l :", l)
		l = l.split(';')
		# print("Second step l :", l)
		date = l[0].split(' ')
		l = l[2].split(',')
		# print("Date : ", date)
		# print("Data : ", l)
		# print("===========\n")
		if count < 30:
			# print(l)
			count +=1
		else :
			# print(l)
			intermediate = date[1].split(':')
			t_i = float(intermediate[0])*3600 + float(intermediate[1])*60 + float(intermediate[2].replace(",", "."))
			tps = datetime.datetime.strptime(date[1], '%H:%M:%S,%f')
			T.append(tps)
			time.append(t_i)
			acc.append([float(l[0]), float(l[1]), float(l[2])])
			gyr.append([float(l[3]), float(l[4]), float(l[5])])
			mag.append([float(l[6]), float(l[7]), float(l[8])])
			temp.append(float(l[9]))
			rate.append(float(l[10]))


	return np.array(time), np.array(acc)*0.01, np.array(gyr)*np.pi/180, np.array(mag), np.array(temp), np.array(rate), T

def traitement_ahrs(file_name_ahrs, borne_inf, borne_sup, axe = 2):
	t, acc, gyr, mag, temp, rate, T = log(file_name_ahrs)
	aZ = acc[borne_inf:borne_sup,axe]
	T = T[borne_inf:borne_sup]

	## Detection des vagues importantes

	mean = np.mean(aZ)
	aZ -= mean
	std = np.std(aZ)
	print(std)
	seuil = 3.3 * std
	aZ_simp = []
	temps_vagues_importante = []
	num_vagues_imp = []
	vb_qd_vague = []

	for i in range(len(aZ)):
		if np.abs(aZ[i])<seuil :
			aZ_simp.append(0)
		else:
			aZ_simp.append(aZ[i])
			num_vagues_imp+=[i]
			temps_vagues_importante.append(t[i])



	#Detection des pics des vagues importantes pour chaque sillage

	aZ_VI = []
	lst_tps = []
	lst_index = []
	for i in num_vagues_imp:
		if aZ_simp[i-1] == 0:
			aZ_VI.append(aZ_simp[i])
			lst_tps.append(T[i])
			lst_index.append(i)


	### Détection des différents groupes:
	dic_grp = {}
	i = 0
	id = 1
	while i < len(aZ_VI):
		dic_grp['Mesure' + str(id)] = [lst_tps[i]]
		j = 1
		while i+j < len(aZ_VI)  and lst_index[i+j]-lst_index[i]< 60:
			dic_grp['Mesure' + str(id)].append(lst_tps[i+j])
			j+=1

		id+= 1
		i+=j
	print("Groupe " + file_name_ahrs + str(axe) +":", dic_grp)
	moyenne = {}
	delta = {}
	for e in dic_grp.keys():
		delta[e] = []

		for i in range(len(dic_grp[e])-1):
			dt = dic_grp[e][i+1]-dic_grp[e][i]
			delta[e].append(round(dt.seconds + dt.microseconds*1e-6, 3))
		#moyenne[e] = np.mean(delta[e])
	print("Nombre de groupe: " + str(id - 1))
	print("Delta " + file_name_ahrs + ":", delta)









	return aZ, aZ_simp, aZ_VI, T, lst_tps, file_name_ahrs


def traitemet_ahrs_butterworth(file_name_ahrs, borne_inf, borne_sup, Te):
	_, acc, gyr, mag, temp, rate, T = log(file_name_ahrs)
	aZ = acc[borne_inf:borne_sup, 2]
	T = T[borne_inf:borne_sup]
	tps = T[-1] - T[0]
	tps = tps.seconds + tps.microseconds*1e-6
	mean = np.mean(aZ)
	tf = np.fft.fft(aZ-mean)
	spectre = np.abs(tf)*2/len(tf)
	freq = np.fft.fftfreq(aZ.size, d=Te)
	plt.figure(figsize=(10, 4))
	plt.plot(freq, spectre, 'r')
	plt.xlabel('f')
	plt.ylabel('A')
	#plt.axis([-0.1, fe / 2, 0, spectre.max()])
	plt.grid()




def plot_ahrs(T, aZ, aZ_simp, aZ_VI, lst_tps, lst_tps_vrai, file_name_ahrs, axe = 2):
	fig, ax = plt.subplots()
	fig.autofmt_xdate()
	ax.xaxis.axis_date()
	myFmt = mdates.DateFormatter('%H:%M:%S')
	ax.xaxis.set_major_formatter(myFmt)
	plt.plot_date(T, np.abs(aZ), '-')
	plt.plot_date(T, np.abs(aZ_simp), '-')
	plt.title(file_name_ahrs + str(axe))
	tps = []
	for e in lst_tps_vrai:
		el = datetime.datetime.strptime(e, '%H:%M:%S.%f')
		tps.append(el)
		plt.axvline(x=el)
	plt.plot_date(lst_tps, np.abs(aZ_VI), color='b', label="Vague")
	plt.legend()
	plt.show()

if __name__ == '__main__':
	#gpx_file = open('guerledan\\Log\\2022-10-12-14_59_56.gpx', 'r')
	fe =50
	borne_inf = 0
	borne_sup = 4403
	file_name_ahrs1 = 'ahrs1_log_2023-01-19_13_34_14.log'
	file_name_ahrs2 = 'ahrs2_log_2022-10-14_07_44_50.log'
	file_name_ahrs3 = 'ahrs3_log_2022-10-14_07_44_50.log'
	file_name_ahrs4 = 'ahrs4_log_2022-10-14_07_44_50.log'

	# lst_tps_vrai = ['07:54:54.82', '07:55:38.65', '07:56:20.48', '07:56:59.63', '07:57:30.27', '07:58:10.93', '07:58:48.31', '07:59:20.84', '08:00:00.39', '08:00:35.97', '08:01:06.07', '08:01:40.59', '08:02:14.72', '08:02:55.85', '08:03:29.77', '08:04:05.97', '08:04:36.77', '08:05:12.87']

#borneinf, sup 12/10 28568, 50367
#lst_tps_vrai_12/10/22 = "13:00:41,79", "13:01:35,09", "13:02:20,27", "13:03:11,06", "13:04:11,76", "13:04:53,14",
						# "13:05:45,89", "13:06:30,60", "13:07:24,33",
						# "13:08:11,05", "13:08:49,48", "13:10:11,31", "13:11:37,94", "13:12:50,64", "13:14:17,99",
						# "13:15:43,02", "13:17:20,85"

	aZ, aZ_simp, aZ_VI, T, lst_index, file_name_ahrs1 = traitement_ahrs(file_name_ahrs1, borne_inf, borne_sup )
	plt.plot(T, aZ)
	plot_ahrs(T,aZ, aZ_simp, aZ_VI, lst_index, lst_tps_vrai, file_name_ahrs1)

	# aZ, aZ_simp, aZ_VI, T, lst_index, file_name_ahrs1 = traitement_ahrs(file_name_ahrs1, borne_inf, borne_sup, axe =0)
	# plot_ahrs(T, aZ, aZ_simp, aZ_VI, lst_index, lst_tps_vrai, file_name_ahrs1, axe =0)
	#
	# aZ, aZ_simp, aZ_VI, T, lst_index, file_name_ahrs1 = traitement_ahrs(file_name_ahrs1, borne_inf, borne_sup, axe=1)
	# plot_ahrs(T, aZ, aZ_simp, aZ_VI, lst_index, lst_tps_vrai, file_name_ahrs1, axe=1)

	# aZ, aZ_simp, aZ_VI, T, lst_index, file_name_ahrs2 = traitement_ahrs(file_name_ahrs2, borne_inf, borne_sup)
	# plot_ahrs(T,aZ, aZ_simp, aZ_VI, lst_index, lst_tps_vrai, file_name_ahrs2)
	#
	# aZ, aZ_simp, aZ_VI, T, lst_index, file_name_ahrs3 = traitement_ahrs(file_name_ahrs3, borne_inf, borne_sup)
	# plot_ahrs(T,aZ, aZ_simp, aZ_VI, lst_index, lst_tps_vrai,file_name_ahrs3)
	#
	# aZ, aZ_simp, aZ_VI, T, lst_index, file_name_ahrs4 = traitement_ahrs(file_name_ahrs4, borne_inf, borne_sup)
	# plot_ahrs(T,aZ, aZ_simp, aZ_VI, lst_index, lst_tps_vrai, file_name_ahrs4)

	# traitemet_ahrs_butterworth(file_name_ahrs3, borne_inf,borne_sup, fe)